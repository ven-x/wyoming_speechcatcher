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
        return [args.language]
    return [lang for lang in args.preload_language if lang]


def resolve_allowed_models(args: argparse.Namespace) -> list:
    if args.allowed_models is not None:
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
    try:
        import torch

        torch.set_num_threads(max(1, int(getattr(args, "num_threads", 1))))
    except ImportError:
        pass
      
    args.allowed_models = resolve_allowed_models(args)
    _LOGGER.info("Allowed models: %s", ", ".join(args.allowed_models))

    state = State(args)
    for language in resolve_preload_languages(args):
        state.preload(language)

    server = AsyncServer.from_uri(args.uri)
    handler_factory = partial(SpeechcatcherEventHandler, args, state)
  
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
    except Exception:
        _LOGGER.exception("Failed to enable Zeroconf discovery")

    _LOGGER.info("Starting Wyoming Speechcatcher server at %s", args.uri)
    await server.run(handler_factory)


def main() -> None:
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
    try:
        main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
