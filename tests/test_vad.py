import numpy as np
import pytest

from wayasr.vad import SileroVAD


def test_bundled_model_runs_without_torch():
    vad = SileroVAD()
    assert vad.session.get_providers() == ["CPUExecutionProvider"]
    for _ in range(5):
        probability = vad(np.zeros(512, dtype=np.float32))
        assert 0 <= probability < 0.5
    assert vad.state.shape == (2, 1, 128)
    assert vad.state.dtype == np.float32
    assert vad.context.shape == (1, 64)


def test_vad_preserves_context_and_state(monkeypatch):
    vad = SileroVAD()
    state = np.ones((2, 1, 128), dtype=np.float32)
    calls = []

    def run(outputs, inputs):
        calls.append(inputs)
        return np.array([[0.75]], dtype=np.float32), state

    monkeypatch.setattr(vad.session, "run", run)
    samples = np.arange(512, dtype=np.float32) / 512
    assert vad(samples) == 0.75
    assert vad(samples) == 0.75
    np.testing.assert_array_equal(calls[0]["input"][0, :64], np.zeros(64))
    np.testing.assert_array_equal(calls[1]["input"][0, :64], samples[-64:])
    assert calls[1]["state"] is state
    assert calls[1]["sr"].dtype == np.int64


@pytest.mark.parametrize("shape", [(256,), (1024,), (1, 512)])
def test_vad_rejects_wrong_frame_shape(shape):
    with pytest.raises(ValueError, match="512 samples"):
        SileroVAD()(np.zeros(shape, dtype=np.float32))
