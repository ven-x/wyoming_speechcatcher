from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("wyoming-speechcatcher")
except PackageNotFoundError:
    _VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()

__all__ = ["__version__"]
