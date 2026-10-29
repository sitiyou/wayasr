from pathlib import Path
import wave


def read_audio(path):
    path = Path(path).expanduser()
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000):
                return wav.readframes(wav.getnframes())
    except (wave.Error, EOFError):
        pass

    import librosa
    import numpy as np
    import soundfile as sf

    try:
        with sf.SoundFile(path) as source:
            samples, _ = librosa.load(source, sr=16000, mono=True, dtype=np.float32)
    except (sf.LibsndfileError, ValueError) as error:
        raise ValueError(f"Cannot decode audio: {path}. Use a format supported by libsndfile, such as WAV, FLAC, Ogg or MP3.") from error
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2").tobytes()
