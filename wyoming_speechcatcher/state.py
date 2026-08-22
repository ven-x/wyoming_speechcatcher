from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from .models import TAGS, MODELS, LANG_FOR_TAG

if TYPE_CHECKING:
    from speechcatcher.speech2text_streaming import Speech2TextStreaming

_LOGGER = logging.getLogger(__name__)

try:
    import speechcatcher
    _HAS_SPEECHCATCHER = True
except ImportError:
    speechcatcher = None
    _HAS_SPEECHCATCHER = False


class State:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.models: Dict[str, "Speech2TextStreaming"] = {}
        self.data_dirs: List[Path] = [
            Path(d).expanduser() for d in getattr(args, "data_dir", [])
        ]
        self.cache_dir: Path = Path(
            getattr(args, "cache_dir", "/share/wyoming-speechcatcher")
        ).expanduser()
      
        self.model_for_language: Dict[str, str] = {}
      
        for pair in getattr(args, "model_for_language", []):
            if len(pair) == 2:
                lang, tag = pair
                self.model_for_language[lang] = tag
              
        allowed = getattr(args, "allowed_models", None)
        self.allowed_models: Optional[set] = set(allowed) if allowed is not None else None
        self._locks: Dict[str, asyncio.Lock] = {}

    def resolve_model_name(self, language: Optional[str], model_name: Optional[str]) -> str:
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
            global_model = getattr(self.args, "model", None)
            if global_model is not None and global_model in TAGS:
                model_lang = LANG_FOR_TAG.get(global_model)
                if model_lang == lang:
                    return global_model
            return MODELS[lang][0]

        raise ValueError(
            f"Unsupported language {lang!r}: no speechcatcher model "
            f"available (supported: {sorted(MODELS)})"
        )

    def _load_from_data_dir(self, model_name: str) -> Optional["Speech2TextStreaming"]:
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
                beam_size=getattr(self.args, "beam_size", 5),
                ctc_weight=getattr(self.args, "ctc_weight", 0.3),
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
        resolved = self.resolve_model_name(language, model_name)
      
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
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]

    def model_available_locally(self, short_tag: str) -> bool:
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
        resolved = self.resolve_model_name(language, model_name)
      
        if resolved in self.models:
            _LOGGER.debug("Model %s already preloaded", resolved)
            return

        _LOGGER.info("Preloading model %s ...", resolved)
        # get_model() stores into self.models
        self.get_model(language=language, model_name=resolved)
        _LOGGER.info("Preloaded model %s", resolved)
