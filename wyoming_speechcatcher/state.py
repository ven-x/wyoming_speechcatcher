"""Model state management for wyoming-speechcatcher.

Implements the 3-layer model loading strategy:
  1. Preload cache (in-memory dict, populated at startup)
  2. Local data directories (--data-dir, repeatable)
  3. espnet_model_zoo cache (--cache-dir, default ~/.cache/espnet)
  4. Automatic download via speechcatcher.load_model()

The module is importable even when the optional ``speechcatcher``
dependency is not installed (e.g. in CI or for unit tests). Only
actual model loading requires it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .models import TAGS, MODELS, LANG_FOR_TAG

if TYPE_CHECKING:
    from speechcatcher.speech2text_streaming import Speech2TextStreaming

_LOGGER = logging.getLogger(__name__)

# Optional dependency — import lazily so state.py is testable without torch.
try:
    import speechcatcher
    _HAS_SPEECHCATCHER = True
except ImportError:  # pragma: no cover
    speechcatcher = None  # type: ignore[assignment]
    _HAS_SPEECHCATCHER = False


class State:
    """Holds loaded models and resolves them via the 3-layer search.

    Args:
        args: Parsed argparse namespace. Expected attributes:
            ``model`` (str), ``language`` (str),
            ``model_for_language`` (List[List[str]]),
            ``data_dir`` (List[str]), ``cache_dir`` (str),
            ``beam_size`` (int), ``ctc_weight`` (float),
            ``decoder`` (str), ``penalty`` (float), ``disable_bbd`` (bool).
            Compute device/dtype sind fest CPU-only (cpu/float32).
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.models: Dict[str, "Speech2TextStreaming"] = {}
        self.data_dirs: List[Path] = [
            Path(d).expanduser() for d in getattr(args, "data_dir", [])
        ]
        self.cache_dir: Path = Path(
            getattr(args, "cache_dir", "/share/wyoming-speechcatcher")
        ).expanduser()
        # --model-for-language overrides: {language: short_tag}
        self.model_for_language: Dict[str, str] = {}
        for pair in getattr(args, "model_for_language", []):
            if len(pair) == 2:
                lang, tag = pair
                self.model_for_language[lang] = tag
        # --allowed-models whitelist (AUDIT-001). None means: no
        # restriction (e.g. when State is used outside the CLI).
        allowed = getattr(args, "allowed_models", None)
        self.allowed_models: Optional[set] = set(allowed) if allowed is not None else None

        # AUDIT-003 (Variante B): Lock-Registry pro Modell für
        # exklusiven Zugriff über die komplette Äußerung.
        self._locks: Dict[str, asyncio.Lock] = {}

    def resolve_model_name(self, language: Optional[str], model_name: Optional[str]) -> str:
        """Resolve the effective short tag for a request.

        Priority:
          1. ``model_name`` if given and valid (must match ``language`` if both given)
          2. ``--model-for-language`` override for the language
          3. Default model for the language (first entry in ``MODELS``)
          4. Global default ``self.args.model`` as fallback

        Raises:
            ValueError: If the resolved tag is unknown.
        """
        if model_name is not None:
            if model_name not in TAGS:
                raise ValueError(f"Unknown model name: {model_name!r}")
            if language is not None:
                expected_lang = LANG_FOR_TAG.get(model_name)
                if expected_lang != language:
                    _LOGGER.warning(
                        "Model %s is for language %s, but language %s was requested",
                        model_name, expected_lang, language,
                    )
            return model_name

        lang = language or getattr(self.args, "language", "de")
        if lang in self.model_for_language:
            tag = self.model_for_language[lang]
            if tag not in TAGS:
                raise ValueError(f"Unknown model in --model-for-language: {tag!r}")
            return tag

        if lang in MODELS:
            # Bug-Fix: --model (self.args.model) hat Vorrang vor dem
            # Sprach-Default (MODELS[lang][0]), wenn es zur angefragten
            # Sprache passt. Ohne diesen Check wurde immer das erste
            # Modell der Sprache (M) verwendet — ein Modellwechsel via
            # --model oder HA-Option hatte keine Wirkung.
            global_model = getattr(self.args, "model", None)
            if global_model is not None and global_model in TAGS:
                model_lang = LANG_FOR_TAG.get(global_model)
                if model_lang == lang:
                    return global_model
            return MODELS[lang][0]

        return getattr(self.args, "model", "de_streaming_transformer_m")

    def _load_from_data_dir(self, model_name: str) -> Optional["Speech2TextStreaming"]:
        """Try to load a model from any configured --data-dir.

        Looks for ``<data_dir>/<model_name>/valid.acc.best.pth``.
        Returns a Speech2TextStreaming instance or None if not found.
        """
        if not _HAS_SPEECHCATCHER:
            return None

        from speechcatcher.speech2text_streaming import Speech2TextStreaming

        for data_dir in self.data_dirs:
            model_dir = data_dir / model_name
            checkpoint = model_dir / "valid.acc.best.pth"
            if checkpoint.exists():
                _LOGGER.info("Loading model %s from data-dir %s", model_name, model_dir)
                return Speech2TextStreaming(
                    model_dir=str(model_dir),
                    beam_size=getattr(self.args, "beam_size", 5),
                    ctc_weight=getattr(self.args, "ctc_weight", 0.3),
                    device="cpu",
                    dtype="float32",
                    use_bbd=not getattr(self.args, "disable_bbd", False),
                )
        return None

    def _load_via_speechcatcher(self, model_name: str) -> "Speech2TextStreaming":
        """Download the model and construct a ``Speech2TextStreaming``.

        Uses the espnet_model_zoo cache (``--cache-dir``); downloads from
        HuggingFace only if the model is not already cached.

        ``speechcatcher.load_model()`` is NOT used for the native decoder: its
        internal checkpoint search only knows ``valid.acc.best.pth`` /
        ``ave_6best`` / ``ave`` / ``model.pth`` / ``checkpoint.pth``, but the
        current HF model repos ship the checkpoint as
        ``valid.acc.ave_10best.pth`` → ``FileNotFoundError``. We resolve the
        checkpoint ourselves and symlink a recognized name before constructing
        the model. The ESPnet decoder path is unaffected and still delegates to
        ``speechcatcher.load_model()``.
        """
        if not _HAS_SPEECHCATCHER:
            raise ImportError(
                "speechcatcher is required to load models. "
                "Install it with: pip install speechcatcher"
            )

        tag = TAGS[model_name]
        decoder = getattr(self.args, "decoder", "native")

        if decoder == "espnet":
            from espnet_model_zoo.downloader import ModelDownloader
            from espnet_streaming_decoder.asr_inference_streaming import (
                Speech2TextStreaming as ESPnetStreaming,
            )

            _LOGGER.info("Resolving model %s (espnet, cache-dir=%s)",
                         model_name, self.cache_dir)
            downloader = ModelDownloader(str(self.cache_dir))
            info = downloader.download_and_unpack(tag, quiet=True)

            config_path = info.get("asr_train_config") or info.get("train_config")
            model_path = info.get("asr_model_file") or info.get("model_file")
            if not config_path or not model_path:
                raise ValueError(
                    f"Could not find config/model paths in info: {sorted(info)}"
                )

            _LOGGER.info("Loading model %s (espnet) from %s", model_name, model_path)
            return ESPnetStreaming(
                asr_train_config=str(config_path),
                asr_model_file=str(model_path),
                device="cpu",
                token_type=None,
                bpemodel=None,
                maxlenratio=0.0,
                minlenratio=0.0,
                beam_size=getattr(self.args, "beam_size", 20),
                ctc_weight=getattr(self.args, "ctc_weight", 0.5),
                lm_weight=0.0,
                penalty=getattr(self.args, "penalty", 0.0),
                nbest=1,
                disable_repetition_detection=True,
                decoder_text_length_limit=0,
                encoded_feat_length_limit=0,
            )

        # Native decoder: Download + Checkpoint-Namens-Fix + direkte Konstruktion.
        from espnet_model_zoo.downloader import ModelDownloader
        from speechcatcher.speech2text_streaming import Speech2TextStreaming

        _LOGGER.info("Resolving model %s (cache-dir=%s)", model_name, self.cache_dir)
        downloader = ModelDownloader(str(self.cache_dir))
        info = downloader.download_and_unpack(tag, quiet=True)

        model_dir: Optional[Path] = None
        for key in ("asr_model_file", "asr_train_config", "model_file", "train_config"):
            value = info.get(key)
            if value:
                model_dir = Path(value).parent
                break
        if model_dir is None:
            raise ValueError(
                f"Could not determine model directory from info: {sorted(info)}"
            )

        recognized = (
            "valid.acc.best.pth", "valid.acc.ave_6best.pth", "valid.acc.ave.pth",
            "model.pth", "checkpoint.pth",
        )
        if not any((model_dir / name).exists() for name in recognized):
            pth_files = sorted(model_dir.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth checkpoint found in {model_dir}")
            link = model_dir / "valid.acc.best.pth"
            if not link.exists():
                link.symlink_to(pth_files[0].name)
                _LOGGER.info(
                    "Checkpoint %s -> symlink %s", pth_files[0].name, link.name
                )

        _LOGGER.info("Loading model %s from %s", model_name, model_dir)
        return Speech2TextStreaming(
            model_dir=str(model_dir),
            beam_size=getattr(self.args, "beam_size", 5),
            ctc_weight=getattr(self.args, "ctc_weight", 0.3),
            device="cpu",
            dtype="float32",
            use_bbd=not getattr(self.args, "disable_bbd", False),
        )

    def get_model(
        self,
        language: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "Speech2TextStreaming":
        """Return a loaded model, using the 3-layer search.

        Layers:
          0. In-memory preload cache (``self.models``)
          1. ``--data-dir`` directories (local, no download)
          2. ``--cache-dir`` via espnet_model_zoo (local cache, no download if present)
          3. Automatic download via speechcatcher.load_model()

        Args:
            language: Requested language code (e.g. ``de``). Used to
                resolve the default model if ``model_name`` is omitted.
            model_name: Explicit short tag. Takes precedence over
                language-based resolution.

        Returns:
            A ready-to-use ``Speech2TextStreaming`` instance.

        Raises:
            ValueError: If the resolved model name is invalid.
            ImportError: If speechcatcher is not installed.
            RuntimeError: If loading fails (network, corrupt files, etc.).
        """
        resolved = self.resolve_model_name(language, model_name)

        # AUDIT-001: reject models that are not on the whitelist before
        # any loading / download is attempted.
        if self.allowed_models is not None and resolved not in self.allowed_models:
            raise ValueError(
                f"Model {resolved!r} is not in the allowed models whitelist "
                f"({sorted(self.allowed_models)}). "
                "Use --allowed-models to extend the whitelist."
            )

        # Layer 0: preload cache
        if resolved in self.models:
            _LOGGER.debug("Model %s found in preload cache", resolved)
            return self.models[resolved]

        # Layer 1: --data-dir
        model = self._load_from_data_dir(resolved)
        if model is not None:
            self.models[resolved] = model
            return model

        # Layer 2+3: cache-dir / download via speechcatcher.load_model()
        try:
            model = self._load_via_speechcatcher(resolved)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {resolved!r}: {exc}"
            ) from exc

        self.models[resolved] = model
        return model

    def get_lock(self, model_name: str) -> asyncio.Lock:
        """Return the asyncio.Lock for a given model name.

        Creates the lock on first access. The lock is used to serialize
        access to the shared model instance across concurrent connections
        (AUDIT-003, Variante B).

        Args:
            model_name: Resolved short tag (e.g. ``de_streaming_transformer_m``).

        Returns:
            The asyncio.Lock for this model.
        """
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]

    def model_available_locally(self, short_tag: str) -> bool:
        """Return True if the model checkpoint is available locally.

        Checks (in order):
          1. Any ``--data-dir``: ``<data_dir>/<short_tag>/valid.acc.best.pth``
          2. ``--cache-dir`` (espnet_model_zoo): ``<cache_dir>/<hf_tag>/valid.acc.best.pth``

        No network access, no download. Used by ``build_info()`` to advertise
        per-model ``installed`` truthfully (AUDIT-005).

        Args:
            short_tag: Model short tag (e.g. ``de_streaming_transformer_m``).

        Returns:
            True if the checkpoint exists locally.
        """
        checkpoint_name = "valid.acc.best.pth"
        for data_dir in self.data_dirs:
            if (data_dir / short_tag / checkpoint_name).exists():
                return True
        hf_tag = TAGS.get(short_tag)
        if hf_tag:
            # huggingface_hub-Cache: cache_dir/models--<org>--<repo>/snapshots/<rev>/...
            cache_root = self.cache_dir / ("models--" + hf_tag.replace("/", "--"))
            if cache_root.exists() and any(cache_root.glob("**/*.pth")):
                return True
        return False

    def preload(
        self,
        language: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Load a model into the preload cache at startup.

        If ``language`` is omitted, the default language from ``args`` is
        used. If ``model_name`` is omitted, the model for the language is
        resolved (respecting ``--model-for-language`` overrides).

        Raises:
            ValueError: If the model cannot be resolved.
            RuntimeError: If loading fails.
        """
        resolved = self.resolve_model_name(language, model_name)
        if resolved in self.models:
            _LOGGER.debug("Model %s already preloaded", resolved)
            return

        _LOGGER.info("Preloading model %s ...", resolved)
        # get_model() stores into self.models
        self.get_model(language=language, model_name=resolved)
        _LOGGER.info("Preloaded model %s", resolved)
