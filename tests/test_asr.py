import io
import threading
import wave
from unittest.mock import Mock

import pytest

from wayasr.client import LlamaASR
from wayasr.live import AudioSegmenter, SubtitleOutput, TranscriptionWorker


def test_tui_renders_literal_text_and_metrics():
    from rich.console import Console
    from wayasr.tui import ASRView

    stream = io.StringIO()
    console = Console(file=stream, width=100, height=30, color_system=None)
    view = ASRView("monitor", "Chinese", "http://localhost:8087", console=console)
    view.sentence("[red]literal transcript[/red]")
    view.partial("稳定文本", "待确认文本")
    view.audio(True, 2.5)
    view.metrics(10, 0.5, 2)
    view.show_error("test error")
    console.print(view.render())
    rendered = stream.getvalue()
    for expected in ("WayASR", "[red]literal transcript[/red]", "稳定文本", "待确认文本",
                     "Speech", "2.5s", "Queued 2", "RTF 0.050", "test error"):
        assert expected in rendered


def test_subtitle_output_routes_to_tui_without_terminal_escapes(capsys):
    view = Mock()
    output = SubtitleOutput(view)
    output.update("你好世界")
    output.update("你好朋友")
    view.partial.assert_called_with("你好", "朋友")
    output.update("第一句。第二句", final=True, end=False)
    view.sentence.assert_called_with("第一句。")
    view.partial.assert_called_with("第二句")
    output.flush()
    view.sentence.assert_called_with("第二句")
    assert capsys.readouterr().out == ""


def test_wav_request():
    client = LlamaASR(language="Chinese")
    client.session = Mock()
    client.session.post.return_value.json.return_value = {"text": "language Chinese<asr_text>你好"}
    pcm = b"\x01\x00" * 512
    assert client.transcribe(pcm) == "你好"
    call = client.session.post.call_args
    assert call.args[0].endswith("/v1/audio/transcriptions")
    assert call.kwargs["data"]["language"] == "Chinese"
    with wave.open(io.BytesIO(call.kwargs["files"]["file"][1]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(512) == pcm


def test_empty_and_invalid_audio():
    client = LlamaASR()
    client.session = Mock()
    assert client.transcribe(b"") == ""
    with pytest.raises(ValueError):
        client.transcribe(b"x")
    client.session.post.assert_not_called()


def test_http_error():
    client = LlamaASR()
    client.session = Mock()
    client.session.post.return_value.raise_for_status.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        client.transcribe(b"\x00\x00")


def test_worker_keeps_finals_and_coalesces_partials():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def transcribe(audio):
        calls.append(audio)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return "text"

    client = Mock()
    client.transcribe.side_effect = transcribe
    worker = TranscriptionWorker(client)
    try:
        worker.submit(b"aa", 0)
        assert entered.wait(5)
        worker.submit(b"bb", 0)
        worker.submit(b"cc", 0, final=True)
        worker.submit(b"dd", 1, final=True)
        worker.submit(b"ee", 2)
        worker.submit(b"ff", 2)
        release.set()
    finally:
        release.set()
        worker.close()
    assert calls[:3] == [b"aa", b"cc", b"dd"]
    assert b"bb" not in calls
    assert b"ee" not in calls
    client.close.assert_called_once()


def test_segmenter_splits_at_recent_pause_without_losing_audio():
    segmenter = AudioSegmenter(100, lookback_chunks=40)
    frames = [index.to_bytes(2, "little") for index in range(100)]
    for index, frame in enumerate(frames):
        result = segmenter.push(frame, not 70 <= index < 75)
    audio, overlap, end = result
    assert audio == b"".join(frames[:75])
    assert overlap == 0
    assert not end
    assert segmenter.finish() == (b"".join(frames[75:]), 0, True)


def test_segmenter_hard_cut_preserves_overlap():
    segmenter = AudioSegmenter(32)
    frames = [index.to_bytes(2, "little") for index in range(40)]
    for frame in frames[:32]:
        result = segmenter.push(frame, True)
    assert result == (b"".join(frames[:32]), 0, False)
    for frame in frames[32:]:
        assert segmenter.push(frame, True) is None
    assert segmenter.finish() == (b"".join(frames[24:]), 8, True)


def test_overlap_alone_is_not_submitted_again():
    segmenter = AudioSegmenter(32)
    for _ in range(32):
        result = segmenter.push(b"aa", True)
    assert result is not None
    assert segmenter.finish() is None


def test_overlap_does_not_remove_part_of_english_word(capsys):
    output = SubtitleOutput()
    output.update("something", final=True)
    capsys.readouterr()
    output.update("thingamajig", final=True, overlap=8)
    assert "thingamajig" in capsys.readouterr().out


def test_segmenter_silence_and_short_noise():
    segmenter = AudioSegmenter(100)
    for _ in range(3):
        segmenter.push(b"aa", True)
    for _ in range(15):
        assert segmenter.push(b"zz", False) is None
    assert not segmenter.frames
    for _ in range(5):
        segmenter.push(b"bb", True)
    for _ in range(15):
        result = segmenter.push(b"zz", False)
    assert result == (b"bb" * 5 + b"zz" * 15, 0, True)


def test_subtitles_stability_and_sentence_continuation(capsys):
    output = SubtitleOutput()
    output.update("你好世界")
    capsys.readouterr()
    output.update("你好朋友")
    assert "你好\033[2m朋友" in capsys.readouterr().out
    output.update("这是一个", final=True, end=False)
    assert output.pending == "这是一个"
    capsys.readouterr()
    output.update("完整的句子。下一句", final=True, end=False)
    assert "这是一个完整的句子。\n" in capsys.readouterr().out
    assert output.pending == "下一句"
    output.flush()
    assert "下一句\n" in capsys.readouterr().out


def test_overlap_dedup_is_bounded_and_only_for_overlapped_audio(capsys):
    output = SubtitleOutput()
    output.update("前面接着春夏秋冬", final=True)
    output.update("春夏秋冬然后继续。", final=True, overlap=8)
    assert "春夏秋冬然后" not in capsys.readouterr().out
    output.update("好的好的", final=True)
    output.update("好的好的", final=True, overlap=8)
    assert "好的好的" in capsys.readouterr().out
    output.update("春夏秋冬", final=True)
    capsys.readouterr()
    output.update("春夏秋冬", final=True, overlap=0)
    assert "春夏秋冬" in capsys.readouterr().out


def test_inflight_partial_is_not_displayed_after_finalization(capsys):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def transcribe(audio):
        calls.append(audio)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            return "stale partial"
        return "final text."

    client = Mock()
    client.transcribe.side_effect = transcribe
    worker = TranscriptionWorker(client)
    try:
        worker.submit(b"aa", 0)
        assert entered.wait(5)
        worker.submit(b"bb", 0, final=True)
    finally:
        release.set()
        worker.close()
    captured = capsys.readouterr().out
    assert "stale partial" not in captured
    assert "final text." in captured
