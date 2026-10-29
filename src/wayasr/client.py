import io
import wave

import requests


def encode_wav(pcm):
    if len(pcm) % 2:
        raise ValueError("Expected mono 16-bit PCM at 16000 Hz")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return buffer.getvalue()


class LlamaASR:
    def __init__(self, url="http://127.0.0.1:8087", language=None, timeout=120, model=None):
        self.url = url.rstrip("/")
        self.language = language
        self.model = model
        self.detected_language = None
        self.detect_language = False
        self.timeout = timeout
        self.session = requests.Session()

    def transcribe(self, pcm):
        if not pcm:
            return ""
        audio = encode_wav(pcm)
        data = {"response_format": "json", "temperature": "0", "max_tokens": "512"}
        if self.model:
            data["model"] = self.model
        if self.language:
            data["language"] = self.language
        response = self.session.post(
            self.url + ("/v1/audio/transcriptions/details" if self.detect_language else "/v1/audio/transcriptions"),
            files={"file": ("audio.wav", audio, "audio/wav")},
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        self.detected_language = result.get("language")
        text = result["text"]
        return text.split("<asr_text>", 1)[-1].strip()

    def close(self):
        self.session.close()

