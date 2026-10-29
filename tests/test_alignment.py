import io
import math
import wave
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wayasr import alignment
from wayasr.alignment import ForcedAligner, aligned_cut
from wayasr.live import AlignedCut, AudioSegmenter, TranscriptionWorker


def spans(*values):
    return [SimpleNamespace(text=text, start_time=start, end_time=end) for text, start, end in values]


def test_word_alignment_maps_stable_sentence_to_audio_time():
    items = spans(("Hello", 0, 0.5), ("world", 0.6, 1.0), ("Next", 1.1, 2.0))
    assert aligned_cut(items, "Hello world. Next", "Hello world.", 3) == 32


def test_character_alignment_maps_chinese_sentence_to_audio_time():
    items = spans(("你", 0, 0.2), ("好", 0.2, 0.4), ("再", 0.5, 0.7), ("见", 0.7, 0.9))
    assert aligned_cut(items, "你好。再见", "你好。", 1) == 13


@pytest.mark.parametrize("items,text,prefix,duration", [
    (spans(("wrong", 0, 1)), "hello.", "hello.", 2),
    (spans(("hello", 0, 1)), "hello. next", "hello.", 2),
    (spans(("hellonext", 0, 1)), "hello.next", "hello.", 2),
    (spans(("hello", -1, 1)), "hello.", "hello.", 2),
    (spans(("hello", 1, 0.5)), "hello.", "hello.", 2),
    (spans(("hello", 0, math.nan)), "hello.", "hello.", 2),
    (spans(("hello", 0, math.inf)), "hello.", "hello.", 2),
    (spans(("hello", 0, 0)), "hello.", "hello.", 2),
    (spans(("hello", 1, 2), ("next", 0, 1)), "hello. next", "hello.", 3),
    (spans(("hello", 0, 1)), "hello.", "other.", 2),
    (spans(("", 0, 1)), "hello.", "hello.", 2),
])
def test_invalid_alignment_is_rejected(items, text, prefix, duration):
    with pytest.raises(ValueError):
        aligned_cut(items, text, prefix, duration)


@pytest.mark.parametrize("end", [2.001, 2.032, 3, 100])
def test_alignment_clamps_boundary_to_complete_audio_frames(end):
    assert aligned_cut(spans(("hello", 0, end)), "hello.", "hello.", 2) == 62


@pytest.mark.parametrize("chunks,end", [(329, 10.56), (392, 12.56)])
def test_aligner_accepts_boundary_beyond_audio_end(monkeypatch, chunks, end):
    session = Mock()
    session.post.return_value.json.return_value = {
        "words": [{"word": "hello", "start": 0, "end": end}]
    }
    monkeypatch.setattr(alignment.requests, "Session", lambda: session)
    client = ForcedAligner("http://aligner", "model", "English")
    try:
        assert client.cut(b"aa" * 512 * chunks, "hello.", "hello.") == chunks
    finally:
        client.close()


def test_out_of_range_tail_does_not_change_earlier_sentence_boundary():
    items = spans(("hello", 0, 0.5), ("next", 1, 3), ("tail", 4, 5))
    assert aligned_cut(items, "hello. next tail", "hello.", 2) == 16


def test_clamping_does_not_hide_invalid_timestamp_order():
    items = spans(("hello", 0, 4), ("next", 3, 3.5))
    with pytest.raises(ValueError, match="Invalid alignment timestamps"):
        aligned_cut(items, "hello. next", "hello.", 2)


def test_aligner_uploads_full_snapshot_and_maps_sentence_boundary(monkeypatch):
    session = Mock()
    session.post.return_value.json.return_value = {"words": [
        {"word": "你", "start": 0, "end": 0.2},
        {"word": "好", "start": 0.2, "end": 0.9},
        {"word": "后", "start": 1.0, "end": 1.1},
        {"word": "文", "start": 1.2, "end": 1.4},
    ]}
    monkeypatch.setattr(alignment.requests, "Session", lambda: session)
    client = ForcedAligner("http://aligner/", "model", "Chinese", timeout=42)
    audio = b"aa" * 512 * 200
    try:
        assert client.cut(audio, "你好。后文", "你好。") == 29
        call = session.post.call_args
        assert call.args == ("http://aligner/v1/audio/alignments",)
        assert call.kwargs["data"] == {"model": "model", "language": "Chinese", "text": "你好。后文"}
        assert call.kwargs["timeout"] == 42
        name, data, content_type = call.kwargs["files"]["file"]
        assert (name, content_type) == ("audio.wav", "audio/wav")
        with wave.open(io.BytesIO(data)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            assert wav.readframes(wav.getnframes()) == audio
        session.post.return_value.raise_for_status.assert_called_once()
    finally:
        client.close()
    session.close.assert_called_once()


def test_aligner_http_timeout_is_reported(monkeypatch):
    session = Mock()
    session.post.side_effect = alignment.requests.Timeout("timed out")
    monkeypatch.setattr(alignment.requests, "Session", lambda: session)
    client = ForcedAligner("http://aligner", "model", "Chinese", timeout=0.05)
    try:
        with pytest.raises(alignment.requests.Timeout):
            client.cut(b"aa" * 512 * 200, "你好。", "你好。")
    finally:
        client.close()
    session.close.assert_called_once()


def test_aligned_cut_can_commit_sentence_shorter_than_soft_window():
    segmenter = AudioSegmenter(200)
    frames = [index.to_bytes(2, "little") for index in range(170)]
    for frame in frames:
        segmenter.push(frame, True)
    assert segmenter.commit(60) is None
    assert segmenter.commit(60, aligned=True) == (b"".join(frames[:60]), 0, False)
    assert segmenter.frames == frames[44:]
    assert segmenter.finish() == (b"".join(frames[44:]), 16, True)


def test_aligned_cut_cannot_retain_more_context_than_committed_audio():
    segmenter = AudioSegmenter(200)
    for _ in range(170):
        segmenter.push(b"aa", True)
    assert segmenter.commit(5, aligned=True) is None
    assert len(segmenter.frames) == 170


def test_failed_alignment_does_not_fall_back(monkeypatch):
    log_failure = Mock()
    monkeypatch.setattr("wayasr.live.log_alignment_failure", log_failure)
    aligner = Mock()
    aligner.cut.side_effect = ValueError("invalid timestamps")
    worker = TranscriptionWorker(Mock(), view=Mock(), aligner=aligner)
    audio = b"aa" * 512 * 170
    try:
        previous = (0, "第一句。第二句")
        worker.request_cut(audio, "第一句。第二句还在", 0, previous)
        worker.request_cut(audio, "第一句。第二句还在继续", 0, previous)
        aligner.cut.assert_called_once_with(audio, "第一句。第二句还在", "第一句。")
        assert worker.take_cut(0) is None
        worker.request_cut(audio, "第一句。", 0, (0, "第一句。"))
        assert worker.take_cut(0) is None
        worker.view.show_error.assert_not_called()
        log_failure.assert_called_once_with(aligner.cut.side_effect)
    finally:
        worker.close()
    aligner.close.assert_called_once()


def test_worker_aligns_stable_sentence_before_changing_tail():
    aligner = Mock()
    aligner.cut.return_value = 60
    worker = TranscriptionWorker(Mock(), view=Mock(), aligner=aligner)
    audio = b"aa" * 512 * 170
    try:
        worker.request_cut(audio, "第一句。正在说第二句", 0, (0, "第一句。正在说"))
        assert worker.take_cut(0) == AlignedCut(60, "第一句。")
        aligner.cut.assert_called_once_with(audio, "第一句。正在说第二句", "第一句。")
    finally:
        worker.close()


def test_inflight_alignment_cannot_cut_a_finalized_segment():
    entered, release = threading.Event(), threading.Event()
    aligner = Mock()

    def cut(*args):
        entered.set()
        assert release.wait(5)
        return 60

    aligner.cut.side_effect = cut
    client = Mock()
    client.transcribe.side_effect = ["第一句。后文", "第一句。后文变化"]
    worker = TranscriptionWorker(client, view=Mock(), aligner=aligner)
    try:
        worker.previous_partial = (0, "第一句。")
        worker.submit(b"aa" * 512 * 170, 0, can_commit=True)
        assert entered.wait(5)
        worker.submit(b"aa" * 512 * 180, 0, final=True)
    finally:
        release.set()
        worker.close()
    assert worker.take_cut(0) is None


@pytest.mark.parametrize("fails", [False, True])
def test_slow_alignment_does_not_block_transcription_or_finalization(fails, monkeypatch):
    monkeypatch.setattr("wayasr.live.log_alignment_failure", Mock())
    entered, release, finalized = threading.Event(), threading.Event(), threading.Event()
    aligner = Mock()

    def cut(*args):
        entered.set()
        assert release.wait(5)
        if fails:
            raise ValueError("Invalid alignment timestamps")
        return 60

    aligner.cut.side_effect = cut
    client, view = Mock(), Mock()
    client.transcribe.return_value = "第一句。后文"
    view.sentence.side_effect = lambda text: finalized.set()
    worker = TranscriptionWorker(client, view=view, aligner=aligner)
    audio = b"aa" * 512 * 170
    try:
        worker.previous_partial = (0, "第一句。")
        worker.submit(audio, 0, can_commit=True)
        assert entered.wait(5)
        worker.submit(audio, 0, final=True)
        assert finalized.wait(5)
        assert client.transcribe.call_count == 2
        assert not release.is_set()
    finally:
        release.set()
        worker.close()
    assert worker.take_cut(0) is None
    aligner.close.assert_called_once()


def test_aligned_final_uses_stable_text_without_retranscribing_audio():
    client, view = Mock(), Mock()
    worker = TranscriptionWorker(client, view=view)
    worker.submit(b"aa" * 512 * 60, 0, final=True, end=False, final_text="第一句。")
    worker.close()
    client.transcribe.assert_not_called()
    view.sentence.assert_called_once_with("第一句。")


@pytest.mark.parametrize("options", [["--partial-interval", "0"], ["--language", "Chinese", "--partial-interval", "0"]])
def test_aligner_requires_partial_recognition(options):
    from click.testing import CliRunner
    from wayasr.cli import main

    from wayasr.config import save_config

    save_config({"backend": "audio.cpp", "url": "http://audio", "live": {"fa": True}})
    result = CliRunner().invoke(main, ["live", *options])
    assert result.exit_code == 1
    assert "Forced alignment requires a positive --partial-interval." in result.output
