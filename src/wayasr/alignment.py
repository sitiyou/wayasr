import math
from types import SimpleNamespace

import requests

from .client import encode_wav


def normalized_text(text):
    return "".join(char for char in text if char.isalnum() or char == "'")


def aligned_cut(items, text, prefix, duration):
    target = normalized_text(prefix)
    if not target or not text.startswith(prefix):
        raise ValueError("Invalid alignment prefix")
    aligned = ""
    boundary = None
    previous_start = previous_end = 0.0
    for item in items:
        start, end = float(item.start_time), float(item.end_time)
        if (not math.isfinite(start) or not math.isfinite(end)
                or not previous_start <= start <= end <= duration
                or end < previous_end):
            raise ValueError("Invalid alignment timestamps")
        token = normalized_text(item.text)
        if not token:
            raise ValueError("Empty alignment token")
        aligned += token
        if aligned == target:
            boundary = end
        previous_start, previous_end = start, end
    if aligned != normalized_text(text) or boundary is None or boundary <= 0:
        raise ValueError("Alignment does not match the transcript boundary")
    return math.ceil(boundary / 0.032)


class ForcedAligner:
    def __init__(self, url, model, language, timeout=120):
        self.url = url.rstrip("/")
        self.model = model
        self.language = language
        self.timeout = timeout
        self.session = requests.Session()

    def cut(self, audio, text, prefix):
        if not self.language:
            raise ValueError("ASR did not return a detected language.")
        response = self.session.post(
            self.url + "/v1/audio/alignments",
            files={"file": ("audio.wav", encode_wav(audio), "audio/wav")},
            data={"model": self.model, "language": self.language, "text": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        items = [SimpleNamespace(text=word["word"], start_time=word["start"], end_time=word["end"])
                 for word in response.json()["words"]]
        cut = aligned_cut(items, text, prefix, len(audio) / 32000)
        if not 0 < cut <= len(audio) // 1024:
            raise ValueError("Forced aligner returned an invalid audio boundary")
        return cut

    def close(self):
        self.session.close()
