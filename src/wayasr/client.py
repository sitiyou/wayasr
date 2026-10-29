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


class ASRClient:
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
        data = self.request_data()
        if self.language:
            data["language"] = self.language
        response = self.session.post(
            self.url + self.request_path(),
            files={"file": ("audio.wav", audio, "audio/wav")},
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        self.detected_language = result.get("language")
        text = result["text"]
        return self.clean_text(text)

    def clean_text(self, text):
        return text.strip()

    def close(self):
        self.session.close()


class LlamaASR(ASRClient):
    def request_data(self):
        data = {"response_format": "json", "temperature": "0", "max_tokens": "512"}
        if self.model:
            data["model"] = self.model
        return data

    def request_path(self):
        return "/v1/audio/transcriptions"

    def clean_text(self, text):
        return text.split("<asr_text>", 1)[-1].strip()


class AudioCppASR(ASRClient):
    def request_data(self):
        if not self.model:
            raise ValueError("audio.cpp requires an ASR model ID")
        return {"model": self.model}

    def request_path(self):
        return "/v1/audio/transcriptions/details" if self.detect_language else "/v1/audio/transcriptions"


def create_client(backend, url, language=None, timeout=120, model=None):
    if backend == "llama.cpp":
        return LlamaASR(url, language, timeout, model)
    if backend == "audio.cpp":
        return AudioCppASR(url, language, timeout, model or "qwen3-asr")
    raise ValueError(f"Unknown backend: {backend}")

