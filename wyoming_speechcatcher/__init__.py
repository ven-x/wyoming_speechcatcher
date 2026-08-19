"""Wyoming protocol server for Speechcatcher ASR.

Provides a Wyoming-compatible speech-to-text server using
Speechcatcher (EspNet2 streaming transformer models) as backend.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    # Works for any pip-installed package (wheel, sdist, editable).
    __version__ = version("wyoming-speechcatcher")
except PackageNotFoundError:
    # Fallback for direct source execution (e.g. python -m wyoming_speechcatcher
    # from a source checkout without installation).
    _VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()

__all__ = ["__version__"]
