"""Model Landscape."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("model-landscape")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
