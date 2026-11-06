"""Logging for charm.

Importing this module has no side effects beyond creating a logger with a
NullHandler. Handlers, levels and the excepthook are installed explicitly by
the CLI, so importing charm as a library never touches global logging state.
"""

import logging
from contextlib import contextmanager

logger = logging.getLogger("charm")
logger.addHandler(logging.NullHandler())

_HTML_PREAMBLE = """<HTML><HEAD><TITLE>charm report</TITLE></HEAD><BODY><pre>"""
_HTML_EPILOGUE = """</pre></BODY></HTML>"""


def setup_cli_logging(level: int = logging.INFO) -> logging.Handler:
    """Attach a stream handler to the charm logger. Called only from the CLI."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[ %(name)s ] %(levelname)s: %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return handler


@contextmanager
def log_to_file(fname, mode: str = "a"):
    """Tee the charm logger into an HTML log file for the duration of the block.

    Unlike the SimNIBS original, the closing tags are written from a finally
    block, so an exception mid-pipeline no longer leaves the file unterminated.
    """
    with open(fname, mode) as f:
        f.write(_HTML_PREAMBLE)

    handler = logging.FileHandler(fname, mode="a")
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # A library caller has not been through setup_cli_logging, and the file
    # should hold the full log regardless.
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)
        handler.close()
        with open(fname, "a") as f:
            f.write(_HTML_EPILOGUE)


@contextmanager
def log_uncaught_exceptions():
    """Record an exception escaping the block in the log, then re-raise it.

    An excepthook would run too late: by then log_to_file has already detached
    the file handler.
    """
    try:
        yield
    except Exception as exc:
        logger.debug("Traceback", exc_info=True)
        logger.critical("Uncaught exception", exc_info=(type(exc), exc, None))
        raise


def worker_initializer() -> None:
    """Restore console logging in a spawned worker of a CLI run."""
    setup_cli_logging()


def console_logging_enabled() -> bool:
    return any(type(h) is logging.StreamHandler for h in logger.handlers)
