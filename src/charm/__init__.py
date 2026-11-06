"""charm - head model generation from MRI."""

from importlib.metadata import version as _version

__version__ = _version("charm")

__all__ = ["SubjectFiles", "Templates", "__version__", "run"]

_LAZY = {
    "run": ("charm.pipeline.run", "run"),
    "SubjectFiles": ("charm.subject", "SubjectFiles"),
    "Templates": ("charm.resources", "Templates"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def __dir__():
    return sorted(__all__)
