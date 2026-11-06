import logging

import pytest

from charm.logging import log_to_file, log_uncaught_exceptions, logger


def test_log_file_is_closed_and_cleaned(tmp_path):
    fn = tmp_path / "log.html"
    with log_to_file(fn):
        logger.debug("visible text")
    text = fn.read_text()
    assert "visible text" in text
    assert text.endswith("</pre></BODY></HTML>")
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_uncaught_exception_reaches_the_log_file(tmp_path):
    fn = tmp_path / "log.html"
    with pytest.raises(ValueError), log_to_file(fn), log_uncaught_exceptions():
        raise ValueError("boom")
    text = fn.read_text()
    assert "Uncaught exception" in text
    assert "boom" in text
    assert text.endswith("</pre></BODY></HTML>")
