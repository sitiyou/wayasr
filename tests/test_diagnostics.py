import logging

from wayasr import diagnostics
from wayasr.alignment import aligned_cut
from types import SimpleNamespace


def test_alignment_failure_details_are_logged_only_to_file(tmp_path, monkeypatch, capsys):
    logger = logging.Logger("test-alignment", level=logging.ERROR)
    logger.propagate = False
    monkeypatch.setattr(diagnostics, "logger", logger)
    monkeypatch.setattr(diagnostics, "user_state_path", lambda *args, **kwargs: tmp_path)
    try:
        try:
            aligned_cut([SimpleNamespace(text="hello", start_time=-1, end_time=3)],
                        "hello.", "hello.", 2)
        except ValueError as error:
            diagnostics.log_alignment_failure(error)
        content = (tmp_path / "alignment.log").read_text()
        assert "Forced alignment failed:" in content
        assert "word[0]='hello'" in content
        assert "start=-1.0, end=3.0, audio_duration=2" in content
        assert "previous_start=0.0, previous_end=0.0" in content
        assert "Traceback" in content
        assert capsys.readouterr() == ("", "")
        handler, = logger.handlers
        assert handler.maxBytes == 2 * 1024 * 1024
        assert handler.backupCount == 2
    finally:
        for handler in logger.handlers:
            handler.close()


def test_log_directory_failure_is_silent(tmp_path, monkeypatch, capsys):
    logger = logging.Logger("test-alignment")
    monkeypatch.setattr(diagnostics, "logger", logger)
    path = tmp_path / "not-a-directory"
    path.write_text("occupied")
    monkeypatch.setattr(diagnostics, "user_state_path", lambda *args, **kwargs: path)
    diagnostics.log_alignment_failure(ValueError("invalid timestamps"))
    assert capsys.readouterr() == ("", "")


def test_log_write_failure_is_silent(tmp_path, monkeypatch, capsys):
    handler = diagnostics.FileOnlyHandler(tmp_path / "alignment.log")
    logger = logging.Logger("test-alignment")
    logger.addHandler(handler)

    def fail(record):
        raise OSError("disk full")

    monkeypatch.setattr(handler, "shouldRollover", fail)
    try:
        logger.error("failure")
        assert capsys.readouterr() == ("", "")
    finally:
        handler.close()
