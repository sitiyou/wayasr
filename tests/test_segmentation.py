import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from wayasr import live
from wayasr.live import AudioSegmenter, TranscriptionWorker


def test_soft_cut_at_long_pause_after_five_seconds():
    segmenter = AudioSegmenter(468)
    frames = [index.to_bytes(2, "little") for index in range(156)]
    for index, frame in enumerate(frames):
        result = segmenter.push(frame, index < 146)
        if index < 155:
            assert result is None
    assert result == (b"".join(frames), 0, False)
    assert segmenter.frames == []
    assert segmenter.finish() is None


def test_short_pause_without_stable_punctuation_does_not_cut():
    segmenter = AudioSegmenter(468)
    for _ in range(156):
        assert segmenter.push(b"aa", True) is None
    for _ in range(4):
        assert segmenter.push(b"zz", False) is None
    assert segmenter.at_boundary
    assert len(segmenter.frames) == 160


def test_long_pause_before_soft_threshold_still_waits_for_endpoint():
    segmenter = AudioSegmenter(468)
    for _ in range(30):
        assert segmenter.push(b"aa", True) is None
    for _ in range(14):
        assert segmenter.push(b"zz", False) is None
    assert not segmenter.at_boundary
    assert segmenter.push(b"zz", False) == (b"aa" * 30 + b"zz" * 15, 0, True)


def test_async_commit_preserves_audio_captured_after_snapshot():
    segmenter = AudioSegmenter(200, soft_chunks=20)
    frames = [index.to_bytes(2, "little") for index in range(30)]
    for index, frame in enumerate(frames[:20]):
        assert segmenter.push(frame, index < 16) is None
    assert segmenter.at_boundary
    for frame in frames[20:]:
        assert segmenter.push(frame, True) is None
    assert not segmenter.at_boundary
    assert segmenter.commit(20) == (b"".join(frames[:20]), 0, False)
    assert segmenter.frames == frames[20:]
    assert segmenter.voiced == [True] * 10
    assert segmenter.finish() == (b"".join(frames[20:]), 0, True)


@pytest.mark.parametrize("cut", [0, 19, 20, 41])
def test_invalid_or_voiced_commit_leaves_audio_untouched(cut):
    segmenter = AudioSegmenter(200, soft_chunks=20)
    for _ in range(40):
        segmenter.push(b"aa", True)
    assert segmenter.commit(cut) is None
    assert segmenter.frames == [b"aa"] * 40
    assert segmenter.voiced == [True] * 40


def test_default_hard_cut_keeps_half_second_of_audio_context():
    segmenter = AudioSegmenter(200)
    frames = [index.to_bytes(2, "little") for index in range(205)]
    for frame in frames[:200]:
        result = segmenter.push(frame, True)
    assert result == (b"".join(frames[:200]), 0, False)
    for frame in frames[200:]:
        assert segmenter.push(frame, True) is None
    assert segmenter.finish() == (b"".join(frames[184:]), 16, True)


@pytest.mark.parametrize("first,second,boundary,second_segment,expected", [
    ("第一句。", "第一句。", True, 0, True),
    ("First sentence.", "First sentence.", True, 0, True),
    ("他说“结束。”", "他说“结束。”", True, 0, True),
    ("第一句，", "第一句，", True, 0, False),
    ("第一句。", "第二句。", True, 0, False),
    ("3.14", "3.14", True, 0, False),
    ("第一句。", "第一句。", False, 0, False),
    ("第一句。", "第一句。", True, 1, False),
])
def test_worker_requires_stable_sentence_end_and_vad_boundary(
        first, second, boundary, second_segment, expected):
    displayed = threading.Event()
    view = Mock()
    view.partial.side_effect = lambda *args: displayed.set()
    client = Mock()
    client.transcribe.side_effect = [first, second]
    worker = TranscriptionWorker(client, view=view)
    try:
        worker.submit(b"aa" * 512 * 156, 0, boundary=True)
        assert displayed.wait(5)
        with worker.condition:
            assert worker.take_cut(0) is None
        displayed.clear()
        worker.submit(b"bb" * 512 * 160, second_segment, boundary=boundary)
        assert displayed.wait(5)
        with worker.condition:
            assert worker.take_cut(second_segment) == (160 if expected else None)
    finally:
        worker.close()


def test_finalized_inflight_partial_cannot_request_a_cut():
    entered = threading.Event()
    release = threading.Event()
    client = Mock()

    def transcribe(audio):
        entered.set()
        assert release.wait(5)
        return "完整句子。"

    client.transcribe.side_effect = transcribe
    worker = TranscriptionWorker(client, view=Mock())
    try:
        with worker.condition:
            worker.previous_partial = (0, "完整句子。")
        worker.submit(b"aa" * 512 * 156, 0, boundary=True)
        assert entered.wait(5)
        worker.submit(b"bb" * 512 * 160, 0, final=True)
    finally:
        release.set()
        worker.close()
    assert worker.take_cut(0) is None


def test_capture_forwards_pause_snapshot_and_preserves_new_audio(monkeypatch):
    numpy = SimpleNamespace(frombuffer=Mock(return_value=MagicMock()), float32="float32")
    torch = SimpleNamespace(set_num_threads=Mock(), from_numpy=Mock())
    vad = Mock()
    vad.return_value.item.side_effect = [0.9] * 156 + [0.1] * 4 + [0.9] * 10
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "silero_vad", SimpleNamespace(load_silero_vad=lambda **kw: vad))
    client, proc, worker = Mock(url="http://localhost:8087"), Mock(), Mock()
    frames = [index.to_bytes(2, "little") * 512 for index in range(170)]
    proc.stdout.read.side_effect = frames + [b""]
    worker.take_cut.side_effect = [None] * 160 + [160] + [None] * 9
    monkeypatch.setattr(live, "LlamaASR", lambda *args: client)
    monkeypatch.setattr(live.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(live, "TranscriptionWorker", lambda *args, **kwargs: worker)
    args = SimpleNamespace(plain=True, url="http://localhost:8087", language=None,
                           timeout=120, source="test-source", max_segment=15, partial_interval=1000)
    live.capture(args)
    calls = worker.submit.call_args_list
    assert len(calls) == 3
    assert calls[0].args == (b"".join(frames[:160]), 0)
    assert calls[0].kwargs == {"overlap": 0, "boundary": True}
    assert calls[1].args == (b"".join(frames[:160]), 0)
    assert calls[1].kwargs == {"final": True, "overlap": 0, "end": False}
    assert calls[2].args == (b"".join(frames[160:]), 1)
    assert calls[2].kwargs == {"final": True, "overlap": 0, "end": True}
    worker.close.assert_called_once()
    proc.terminate.assert_called_once()
    proc.stdout.close.assert_called_once()


def test_old_segment_cut_requests_are_discarded():
    worker = TranscriptionWorker(Mock(), view=Mock())
    try:
        worker.cuts.put((0, 160))
        worker.cuts.put((1, 170))
        assert worker.take_cut(1) == 170
        assert worker.take_cut(1) is None
    finally:
        worker.close()
