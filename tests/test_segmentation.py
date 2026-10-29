import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from wayasr import live
from wayasr.live import AlignedCut, AudioSegmenter, TranscriptionWorker


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
    assert segmenter.can_commit
    assert len(segmenter.frames) == 160


def test_long_pause_before_soft_threshold_still_waits_for_endpoint():
    segmenter = AudioSegmenter(468)
    for _ in range(30):
        assert segmenter.push(b"aa", True) is None
    for _ in range(14):
        assert segmenter.push(b"zz", False) is None
    assert not segmenter.can_commit
    assert segmenter.push(b"zz", False) == (b"aa" * 30 + b"zz" * 15, 0, True)


def test_async_commit_preserves_audio_captured_after_snapshot():
    segmenter = AudioSegmenter(200, soft_chunks=20)
    frames = [index.to_bytes(2, "little") for index in range(30)]
    for index, frame in enumerate(frames[:20]):
        assert segmenter.push(frame, index < 16) is None
    assert segmenter.can_commit
    for frame in frames[20:]:
        assert segmenter.push(frame, True) is None
    assert segmenter.can_commit
    assert segmenter.commit(20) == (b"".join(frames[:20]), 0, False)
    assert segmenter.frames == frames[4:]
    assert segmenter.voiced == [True] * 12 + [False] * 4 + [True] * 10
    assert segmenter.finish() == (b"".join(frames[4:]), 16, True)


def test_punctuation_commit_does_not_require_silence():
    segmenter = AudioSegmenter(200, soft_chunks=20)
    frames = [index.to_bytes(2, "little") for index in range(30)]
    for frame in frames:
        assert segmenter.push(frame, True) is None
    assert segmenter.commit(20) == (b"".join(frames[:20]), 0, False)
    assert segmenter.finish() == (b"".join(frames[4:]), 16, True)


@pytest.mark.parametrize("cut", [0, 19, 41])
def test_invalid_commit_leaves_audio_untouched(cut):
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


@pytest.mark.parametrize("first,second,can_commit,second_segment,expected", [
    ("第一句。", "第一句。", True, 0, True),
    ("First sentence.", "First sentence.", True, 0, True),
    ("他说“结束。”", "他说“结束。”", True, 0, True),
    ("第一句，", "第一句，", True, 0, False),
    ("第一句。", "第二句。", True, 0, False),
    ("3.14", "3.14", True, 0, False),
    ("第一句。后面还有内容", "第一句。后面还有内容", True, 0, False),
    ("第一句。", "第一句。第二句。", True, 0, False),
    ("第一句。", "第一句。", False, 0, False),
    ("第一句。", "第一句。", True, 1, False),
])
def test_worker_requires_stable_trailing_sentence_end_and_eligible_snapshot(
        first, second, can_commit, second_segment, expected):
    displayed = threading.Event()
    view = Mock()
    view.partial.side_effect = lambda *args: displayed.set()
    client = Mock()
    client.transcribe.side_effect = [first, second]
    worker = TranscriptionWorker(client, view=view)
    completed = threading.Event()
    request_cut = worker.request_cut

    def complete_cut(*args):
        try:
            request_cut(*args)
        finally:
            completed.set()

    worker.request_cut = complete_cut
    try:
        worker.submit(b"aa" * 512 * 156, 0, can_commit=True)
        assert displayed.wait(5)
        assert completed.wait(5)
        with worker.condition:
            assert worker.take_cut(0) is None
        displayed.clear()
        completed.clear()
        worker.submit(b"bb" * 512 * 160, second_segment, can_commit=can_commit)
        assert displayed.wait(5)
        if can_commit:
            assert completed.wait(5)
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
        worker.submit(b"aa" * 512 * 156, 0, can_commit=True)
        assert entered.wait(5)
        worker.submit(b"bb" * 512 * 160, 0, final=True)
    finally:
        release.set()
        worker.close()
    assert worker.take_cut(0) is None


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("silent_chunks", [0, 10])
@pytest.mark.parametrize("max_segment", [15, 5.12])
def test_capture_prioritizes_punctuation_over_vad_and_hard_limit(monkeypatch, silent_chunks, max_segment, aligned):
    numpy = SimpleNamespace(frombuffer=Mock(return_value=MagicMock()), float32="float32")
    vad = Mock(side_effect=[0.9] * (160 - silent_chunks) + [0.1] * silent_chunks + [0.9] * 10)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "wayasr.vad", SimpleNamespace(SileroVAD=lambda: vad))
    client, proc, worker = Mock(url="http://localhost:8087"), Mock(), Mock()
    frames = [index.to_bytes(2, "little") * 512 for index in range(170)]
    proc.stdout.read.side_effect = frames + [b""]
    cut = AlignedCut(60, "第一句。") if aligned else 159
    worker.take_cut.side_effect = [None] * 159 + [cut] + [None] * 10
    monkeypatch.setattr(live, "time", SimpleNamespace(monotonic=lambda: proc.stdout.read.call_count * 0.032))
    monkeypatch.setattr(live, "create_client", lambda *args: client)
    monkeypatch.setattr(live.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(live, "TranscriptionWorker", lambda *args, **kwargs: worker)
    args = SimpleNamespace(plain=True, url="http://localhost:8087", language=None,
                           timeout=120, source="test-source", max_segment=max_segment, partial_interval=159 * 0.032)
    live.capture(args)
    calls = worker.submit.call_args_list
    assert len(calls) == 3
    assert calls[0].args == (b"".join(frames[:159]), 0)
    assert calls[0].kwargs == {"overlap": 0, "can_commit": True}
    committed = 60 if aligned else 159
    assert calls[1].args == (b"".join(frames[:committed]), 0)
    options = {"final_text": "第一句。"} if aligned else {}
    assert calls[1].kwargs == {"final": True, "overlap": 0, "end": False, **options}
    assert calls[2].args == (b"".join(frames[committed - 16:]), 1)
    assert calls[2].kwargs == {"final": True, "overlap": 16, "end": True}
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
