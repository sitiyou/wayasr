import wave
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from wayasr.audio import read_audio


def test_pcm_wav_is_read_without_conversion(tmp_path, monkeypatch):
    import librosa

    path = tmp_path / "audio.wav"
    pcm = b"\x00\x80\xff\x7f" * 128
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    decode = Mock(side_effect=AssertionError("Must not convert PCM WAV"))
    monkeypatch.setattr(librosa, "load", decode)
    assert read_audio(path) == pcm
    decode.assert_not_called()


@pytest.mark.parametrize("suffix,subtype", [(".wav", "PCM_24"), (".flac", "PCM_16"), (".ogg", "VORBIS")])
def test_audio_is_resampled_and_downmixed(tmp_path, suffix, subtype):
    path = tmp_path / ("audio" + suffix)
    rate = 48000
    signal = 0.5 * np.sin(2 * np.pi * 440 * np.arange(rate // 10) / rate)
    sf.write(path, np.column_stack((signal, signal)), rate, subtype=subtype)
    before = path.read_bytes()
    pcm = read_audio(path)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert len(samples) == 1600
    assert 14000 < np.max(samples) < 18000
    assert path.read_bytes() == before


def test_unsupported_audio_has_actionable_error(tmp_path):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Cannot decode audio.*libsndfile"):
        read_audio(path)


def test_float_audio_is_clipped_not_wrapped(tmp_path):
    path = tmp_path / "float.wav"
    sf.write(path, np.array([-2, -1, 0, 1, 2], dtype=np.float32), 16000, subtype="FLOAT")
    assert np.frombuffer(read_audio(path), dtype="<i2").tolist() == [-32768, -32768, 0, 32767, 32767]
