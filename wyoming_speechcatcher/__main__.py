"""Command-line interface and server entry point for wyoming-speechcatcher.

Provides the full CLI argument parser for the Wyoming Speechcatcher ASR
server and the actual server startup:

- ``build_parser()`` defines every CLI parameter documented in STATUS.md
  (``--uri``, model/language selection, decoder settings, storage,
  Wyoming feature flags, phase-2 sentence-correction stubs, ``--debug``).
- ``main()`` configures logging, preloads the default language model(s)
  and runs a Wyoming :class:`~wyoming.server.AsyncServer` with
  :class:`~wyoming_speechcatcher.handler.SpeechcatcherEventHandler`.
- ``run()`` is the console-script entry point; it catches
  :class:`KeyboardInterrupt` so Ctrl-C shuts the server down cleanly.

The module is importable without torch / speechcatcher installed — both
are only required when the server actually starts. This keeps ``--help``
usable on bare systems.
"""

import argparse
import asyncio
import logging
import sys
from functools import partial

from wyoming.server import AsyncServer

from . import __version__
from .handler import SpeechcatcherEventHandler
from .models import MODELS, model_choices, language_choices
from .state import State

_LOGGER = logging.getLogger(__name__)

MODEL_CHOICES = model_choices()
LANGUAGE_CHOICES = language_choices()


def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="wyoming-speechcatcher",
        description=(
            "Wyoming protocol server for Speechcatcher ASR "
            "(EspNet2 streaming transformer models)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # --- Server ---
    parser.add_argument(
        "--uri",
        default="stdio://",
        help="Wyoming server URI (stdio://, tcp://host:port, unix://path). "
        "Default: stdio://",
    )

    # --- Model / language ---
    parser.add_argument(
        "--model",
        default="de_streaming_transformer_m",
        choices=MODEL_CHOICES,
        help="Model short name. Default: de_streaming_transformer_m",
    )
    parser.add_argument(
        "--model-for-language",
        nargs=2,
        action="append",
        metavar=("LANGUAGE", "MODEL"),
        default=[],
        help="Override model for a language, e.g. "
        "--model-for-language de de_streaming_transformer_xl (repeatable)",
    )
    parser.add_argument(
        "--allowed-models",
        action="append",
        default=None,
        metavar="MODEL",
        help="Whitelist of model short names that may be loaded at runtime "
        "(repeatable). Default: all models referenced by --model, "
        "--model-for-language and --preload-language. "
        "Requests for non-whitelisted models are rejected with a clear error.",
    )
    parser.add_argument(
        "--language",
        default="de",
        choices=LANGUAGE_CHOICES,
        help="Default language. Default: de",
    )
    parser.add_argument(
        "--preload-language",
        action="append",
        default=None,
        metavar="LANGUAGE",
        help="Language(s) to preload at server start (repeatable). "
        "Default: the default language is preloaded. "
        "Use --preload-language '' to disable preloading.",
    )

    # --- Decoder / inference ---
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Decoder beam size (2-20). Default: 5",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.3,
        help="CTC weight (0.0-1.0). Default: 0.3",
    )
    parser.add_argument(
        "--decoder",
        default="espnet",
        choices=["native", "espnet"],
        help="Decoder implementation. Default: espnet",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of PyTorch intraop threads on CPU (1-32). "
        "Default: 1",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.0,
        help="Insertion penalty / length bonus (-1.0 to 1.0). "
        "ESPnet decoder only. Default: 0.0",
    )
    parser.add_argument(
        "--disable-bbd",
        action="store_true",
        help="Disable Block Boundary Detection (native decoder only)",
    )

    # --- Model storage ---
    parser.add_argument(
        "--cache-dir",
        default="/share/wyoming-speechcatcher",
        help="Model cache directory. Default: /share/wyoming-speechcatcher",
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory with pre-installed/offline models (repeatable). "
        "Checked BEFORE --cache-dir. For Docker: mount as volume, e.g. "
        "-v /host/models:/models",
    )

    # --- Wyoming features ---
    parser.add_argument(
        "--stream-transcript",
        action="store_true",
        help="Stream partial transcripts (transcript-chunk events)",
    )
    parser.add_argument(
        "--external-vad",
        action="store_true",
        help="Require external VAD (for wake-word setups)",
    )

    # --- Sentence correction (Phase 2, accepted but not yet active) ---
    parser.add_argument(
        "--sentences-dir",
        default=None,
        help="YAML sentence templates directory (optional, Phase 2)",
    )
    parser.add_argument(
        "--correct-sentences",
        type=float,
        default=None,
        metavar="CUTOFF",
        help="Sentence correction with score cutoff (optional, Phase 2)",
    )
    parser.add_argument(
        "--limit-sentences",
        action="store_true",
        help="Only allow template sentences (optional, Phase 2)",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Detect unknown words (optional, Phase 2)",
    )

    # --- Logging ---
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def resolve_preload_languages(args: argparse.Namespace) -> list:
    """Determine which languages to preload at server start.

    Returns a list of language codes. An empty list means: no preloading
    (user passed ``--preload-language ''``).
    """
    if args.preload_language is None:
        # Not specified → preload the default language.
        return [args.language]
    # Explicit list; empty strings disable preloading for that entry.
    return [lang for lang in args.preload_language if lang]


def resolve_allowed_models(args: argparse.Namespace) -> list:
    """Determine the whitelist of models that may be loaded at runtime.

    If ``--allowed-models`` was given explicitly, that list is returned
    (duplicates removed, order preserved). Otherwise the default is
    derived from every model referenced by ``--model``,
    ``--model-for-language`` and the preloaded languages
    (``--preload-language`` / default language).

    Returns a list of model short tags.
    """
    if args.allowed_models is not None:
        # Explicit whitelist — deduplicate, keep order.
        seen = set()
        return [m for m in args.allowed_models if not (m in seen or seen.add(m))]

    allowed = {args.model}
    for pair in getattr(args, "model_for_language", []):
        if len(pair) == 2:
            allowed.add(pair[1])
    for lang in resolve_preload_languages(args):
        allowed.update(MODELS.get(lang, []))
    return sorted(allowed)


async def run_server(args: argparse.Namespace) -> None:
    """Preload model(s) and run the Wyoming AsyncServer."""
    # Avoid PyTorch thread-pool overhead inside the asyncio event loop
    # (STATUS.md "Technische Fallstricke").
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(max(1, int(getattr(args, "num_threads", 1))))
    except ImportError:
        # torch arrives with speechcatcher on the target system; without
        # it State.get_model() will fail later with a clear message.
        pass

    # Resolve the model whitelist before State is built so the effective
    # set is visible in the log (AUDIT-001: prevent foreign-triggered
    # downloads / RAM allocation).
    args.allowed_models = resolve_allowed_models(args)
    _LOGGER.info("Allowed models: %s", ", ".join(args.allowed_models))

    state = State(args)
    # Preload loop: runs synchronously BEFORE server.run() accepts any
    # client connection. Startup blocking is therefore harmless — there
    # are no other coroutines to starve yet (AUDIT-002). Moving this into
    # an executor would not change observable behaviour (server start is
    # delayed either way until models are in RAM), so we keep the simple
    # sequential loop and document the decision here.
    for language in resolve_preload_languages(args):
        state.preload(language)

    server = AsyncServer.from_uri(args.uri)
    handler_factory = partial(SpeechcatcherEventHandler, args, state)

    # Optional: HA-Auto-Discovery via mDNS (Zeroconf). Nur fuer tcp://
    # sinnvoll; benoetigt das wyoming[zeroconf]-Extra.
    try:
        from urllib.parse import urlparse

        from wyoming.zeroconf import HomeAssistantZeroconf

        parsed = urlparse(args.uri)
        if parsed.scheme == "tcp" and parsed.port:
            zeroconf = HomeAssistantZeroconf(port=parsed.port)
            await zeroconf.register_server()
            _LOGGER.info(
                "Zeroconf (mDNS) discovery enabled on port %s", parsed.port
            )
    except ImportError:
        _LOGGER.info(
            "zeroconf not installed (wyoming[zeroconf]) — mDNS discovery disabled"
        )
    except Exception:  # pragma: no cover - defensive
        _LOGGER.exception("Failed to enable Zeroconf discovery")

    _LOGGER.info("Starting Wyoming Speechcatcher server at %s", args.uri)
    await server.run(handler_factory)


def main() -> None:
    """Parse CLI arguments and run the server."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped (KeyboardInterrupt)")


def run() -> None:
    """Console-script entry point (see pyproject [project.scripts])."""
    try:
        main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
