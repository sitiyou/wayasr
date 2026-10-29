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


@pytest.mark.parametrize("previous,current,expected", [
    ("这个PCU是根据一定的系数换算出来的。", "算出来的，这里有个PCU的换算系数。", "这里有个PCU的换算系数。"),
    ("不超过一千三百五十辆呢。", "五十辆呢，这条路的通行效率就视为能够满足。", "这条路的通行效率就视为能够满足。"),
    ("通过二十辆以上的情况。", "样的情况，所以说在一环这个地图中呢。", "样的情况，所以说在一环这个地图中呢。"),
    ("采访的是工作室的一位高管。", "位高管，这个文章的标题是。", "这个文章的标题是。"),
    ("追加大量建筑制作。", "建筑制作文章中提到了。", "文章中提到了。"),
    ("前面说春夏秋冬。", "春夏，秋冬，然后继续。", "然后继续。"),
    ("a thing.", "thingamajig", "thingamajig"),
    ("something.", "thing next", "thing next"),
    ("hello world.", "world, next", "next"),
])
def test_overlap_dedup_ignores_punctuation_without_splitting_words(previous, current, expected):
    view = Mock()
    output = SubtitleOutput(view)
    output.update(previous, final=True)
    view.reset_mock()
    output.update(current, final=True, overlap=16)
    assert "".join(call.args[0] for call in view.sentence.call_args_list) == expected


def test_overlap_dedup_is_applied_to_partial_and_final():
    view = Mock()
    output = SubtitleOutput(view)
    output.update("采访了一位高管。", final=True)
    view.reset_mock()
    output.update("位高管，文章的标题", overlap=16)
    assert view.partial.call_args.args == ("", "文章的标题")
    output.update("位高管，文章的标题。", final=True, overlap=16)
    view.sentence.assert_called_once_with("文章的标题。")


def test_aligned_sentence_preserves_unconfirmed_tail_until_next_recognition():
    view, overlay = Mock(), Mock()
    output = SubtitleOutput(view, overlay)
    output.update("第一句。第二句还在说")
    output.update("第一句。", final=True, end=False, preserve_tail=True)
    for sink in (view, overlay):
        sink.sentence.assert_called_once_with("第一句。")
        sink.partial.assert_called_with("第二句还在说")
    assert output.pending == ""
    output.update("第二句换了说法")
    view.partial.assert_called_with("第二句", "换了说法")
    output.update("第二句说完了。", final=True)
    assert [call.args[0] for call in view.sentence.call_args_list] == ["第一句。", "第二句说完了。"]
    view.partial.assert_called_with("")


@pytest.mark.parametrize("end,preserve_tail,final_text", [
    (True, True, "第一句。"),
    (False, False, "第一句。"),
    (False, True, "另一句。"),
])
def test_unconfirmed_tail_is_not_kept_for_finished_or_mismatched_text(end, preserve_tail, final_text):
    view = Mock()
    output = SubtitleOutput(view)
    output.update("第一句。后面的临时内容")
    view.reset_mock()
    output.update(final_text, final=True, end=end, preserve_tail=preserve_tail)
    view.partial.assert_called_with("")
    assert output.pending == ""


def test_aligned_tail_survives_overlap_deduplication():
    view = Mock()
    output = SubtitleOutput(view)
    output.update("前面的建筑制作。", final=True)
    output.update("建筑制作文章提到了。后文还在说", overlap=16)
    output.update("建筑制作文章提到了。", final=True, end=False,
                  overlap=16, preserve_tail=True)
    view.sentence.assert_called_with("文章提到了。")
    view.partial.assert_called_with("后文还在说")
    output.update("后文还在说然后继续")
    view.partial.assert_called_with("后文还在说", "然后继续")


def test_plain_mode_does_not_print_asr_metrics(capsys):
    client = Mock()
    client.transcribe.return_value = "识别结果。"
    worker = TranscriptionWorker(client)
    worker.submit(b"aa" * 512 * 31, 0, final=True)
    worker.close()
    captured = capsys.readouterr()
    assert "识别结果。" in captured.out
    assert captured.err == ""


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
