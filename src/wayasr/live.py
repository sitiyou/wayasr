import shutil
from collections import deque
import queue
import re
import subprocess
import sys
import threading
import time
from typing import NamedTuple

from .client import create_client


SENTENCE_BOUNDARY = re.compile(r"[。！？]+[”’\"]*|(?<!\d)[.!?]+[\"']*(?=\s|$)")


class AlignedCut(NamedTuple):
    chunks: int
    text: str


class AudioSegmenter:
    def __init__(self, max_chunks, lookback_chunks=94, overlap_chunks=16, soft_chunks=156):
        self.max_chunks = max_chunks
        self.soft_chunks = soft_chunks
        self.lookback_chunks = min(lookback_chunks, max_chunks // 2)
        self.overlap_chunks = min(overlap_chunks, max_chunks // 4)
        self.preroll = deque(maxlen=8)
        self.frames = []
        self.voiced = []
        self.silence = 0
        self.overlap = 0

    def push(self, raw, voiced, defer_fallback=False):
        if not self.frames:
            if not voiced:
                self.preroll.append(raw)
                return None
            self.frames = list(self.preroll)
            self.voiced = [False] * len(self.frames)
            self.preroll.clear()
        self.frames.append(raw)
        self.voiced.append(voiced)
        self.silence = 0 if voiced else self.silence + 1
        if not defer_fallback:
            return self.fallback()
        return None

    def fallback(self):
        if self.silence >= 15:
            return self.finish()
        if len(self.frames) >= self.soft_chunks and self.silence >= 10:
            return self._cut(len(self.frames))
        if len(self.frames) < self.max_chunks:
            return None
        candidates = []
        start = max(1, len(self.frames) - self.lookback_chunks)
        run = 0
        for index in range(start, len(self.frames)):
            run = 0 if self.voiced[index] else run + 1
            if run >= 4:
                candidates.append((run, index + 1))
        cut = max(candidates)[1] if candidates else len(self.frames)
        overlap = 0 if candidates else self.overlap_chunks
        return self._cut(cut, overlap)

    @property
    def can_commit(self):
        return len(self.frames) >= self.soft_chunks

    def commit(self, cut, aligned=False):
        minimum = max(self.overlap + 5, self.overlap_chunks + 1) if aligned else self.soft_chunks
        if (not minimum <= cut <= len(self.frames)
                or sum(self.voiced[self.overlap:cut]) < 5):
            return None
        return self._cut(cut, self.overlap_chunks)

    def _cut(self, cut, overlap=0):
        result = self._result(cut, False)
        self.frames = self.frames[cut - overlap:]
        self.voiced = self.voiced[cut - overlap:]
        self.overlap = overlap
        self.silence = 0
        for voiced in reversed(self.voiced):
            if voiced:
                break
            self.silence += 1
        return result

    def _result(self, cut, end):
        if sum(self.voiced[self.overlap:cut]) < 5:
            return None
        return b"".join(self.frames[:cut]), self.overlap, end

    def finish(self):
        result = self._result(len(self.frames), True)
        self.frames = []
        self.voiced = []
        self.silence = self.overlap = 0
        return result


class SubtitleOutput:
    def __init__(self, view=None, overlay=None):
        self.view = view
        self.views = tuple(sink for sink in (view, overlay) if sink is not None)
        self.previous = ""
        self.pending = ""
        self.last_final = ""

    def update(self, text, final=False, overlap=0, end=True):
        if overlap:
            limit = min(len(text), len(self.last_final), max(1, int(overlap * 0.032 * 24)))
            for size in range(limit, 3, -1):
                prefix = text[:size]
                if (self.last_final.endswith(prefix)
                        and len(set(prefix.strip())) >= 3
                        and (not prefix[-1].isascii() or prefix[-1].isspace()
                             or not prefix[-1].isalnum())):
                    text = text[size:].lstrip()
                    break
        if not final:
            stable = 0
            for left, right in zip(self.previous, text):
                if left != right:
                    break
                stable += 1
            self.previous = text
            for view in self.views:
                view.partial(self.pending + text[:stable], text[stable:])
            if not self.view:
                sys.stdout.write("\r\033[K" + self.pending + text[:stable]
                                 + "\033[2m" + text[stable:] + "\033[0m")
        else:
            self.previous = ""
            self.last_final = text
            separator = " " if self.pending and text and self.pending[-1].isascii() and text[0].isascii() else ""
            combined = self.pending + separator + text
            self.pending = ""
            if not self.view:
                sys.stdout.write("\r\033[K")
            start = 0
            for match in SENTENCE_BOUNDARY.finditer(combined):
                self.sentence(combined[start:match.end()].strip())
                start = match.end()
            self.pending = combined[start:].lstrip()
            if end:
                self.flush()
            else:
                for view in self.views:
                    view.partial(self.pending)
                if not self.view:
                    sys.stdout.write(self.pending)
        if not self.view:
            sys.stdout.flush()

    def sentence(self, text):
        for view in self.views:
            view.sentence(text)
        if not self.view:
            sys.stdout.write(text + "\n")

    def flush(self):
        if self.pending:
            if not self.view:
                sys.stdout.write("\r\033[K")
            self.sentence(self.pending)
            self.pending = ""
        for view in self.views:
            view.partial("")
        if not self.view:
            sys.stdout.flush()


class TranscriptionWorker:
    def __init__(self, client, max_pending=16, view=None, overlay=None, aligner=None):
        self.client = client
        self.aligner = aligner
        self.alignment_attempt = None
        self.finals = queue.Queue(maxsize=max_pending)
        self.partial = None
        self.cuts = queue.SimpleQueue()
        self.previous_partial = None
        self.condition = threading.Condition()
        self.closed = False
        self.segment = 0
        self.finalized_segment = -1
        self.view = view
        self.output = SubtitleOutput(view, overlay)
        self.thread = threading.Thread(target=self.run, name="asr-inference")
        self.thread.start()

    def submit(self, audio, segment, final=False, overlap=0, end=True, can_commit=False, final_text=None):
        with self.condition:
            self.segment = segment
            if final:
                self.partial = None
                self.finals.put_nowait((audio, segment, overlap, end, can_commit, final_text))
                self.finalized_segment = segment
            else:
                self.partial = (audio, segment, overlap, end, can_commit, final_text)
            self.condition.notify()

    def take_cut(self, segment):
        while True:
            try:
                candidate, cut = self.cuts.get_nowait()
            except queue.Empty:
                return None
            if candidate == segment:
                return cut

    def request_cut(self, audio, text, segment, previous):
        if previous is None or previous[0] != segment:
            return
        stable = 0
        for left, right in zip(previous[1], text):
            if left != right:
                break
            stable += 1
        ends = [match.end() for match in SENTENCE_BOUNDARY.finditer(text) if match.end() <= stable]
        if not ends:
            return
        prefix = text[:ends[-1]]
        cut = None
        if self.aligner is not None and self.alignment_attempt != (segment, prefix):
            self.alignment_attempt = (segment, prefix)
            try:
                if not self.client.language:
                    self.aligner.language = self.client.detected_language
                cut = AlignedCut(self.aligner.cut(audio, text, prefix), prefix)
            except Exception as error:
                message = f"Forced alignment failed: {error}"
                if self.view:
                    self.view.show_error(message)
                else:
                    print(message, file=sys.stderr)
        if self.aligner is None and cut is None and previous[1] == text and len(prefix) == len(text):
            cut = len(audio) // 1024
        if cut is not None:
            with self.condition:
                if segment == self.segment and segment > self.finalized_segment and not self.closed:
                    self.cuts.put((segment, cut))

    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.closed or not self.finals.empty() or self.partial is not None
                    )
                    if not self.finals.empty():
                        audio, segment, overlap, end, can_commit, final_text = self.finals.get_nowait()
                        final = True
                    elif self.partial is not None:
                        audio, segment, overlap, end, can_commit, final_text = self.partial
                        self.partial = None
                        final = False
                    else:
                        return
                started = time.perf_counter()
                if self.view:
                    self.view.set_status("Transcribing")
                try:
                    text = final_text if final and final_text is not None else self.client.transcribe(audio)
                    with self.condition:
                        stale = segment != self.segment or segment <= self.finalized_segment or self.closed
                        if final or not stale:
                            self.output.update(text, final, overlap, end)
                            previous = self.previous_partial
                            self.previous_partial = None if final else (segment, text)
                    if not final and not stale and can_commit:
                        self.request_cut(audio, text, segment, previous)
                    if final and final_text is None:
                        duration = len(audio) / 32000
                        elapsed = time.perf_counter() - started
                        if self.view:
                            self.view.metrics(duration, elapsed, self.finals.qsize())
                        else:
                            print(f"[ASR] audio={duration:.2f}s inference={elapsed:.2f}s RTF={elapsed / duration:.3f}", file=sys.stderr)
                except Exception as error:
                    if self.view:
                        self.view.show_error(error)
                    else:
                        print(f"ASR error: {error}", file=sys.stderr)
                finally:
                    if self.view:
                        self.view.set_status("Finishing" if self.closed else "Listening")
        finally:
            try:
                self.output.flush()
            finally:
                try:
                    self.client.close()
                finally:
                    if self.aligner is not None:
                        self.aligner.close()

    def close(self):
        with self.condition:
            self.closed = True
            self.partial = None
            self.condition.notify()
        self.thread.join()


def run(args):
    if args.partial_interval < 0 or not 1 <= args.max_segment <= 25:
        raise ValueError("partial interval must be nonnegative; max segment must be 1–25 seconds")
    if getattr(args, "aligner_url", None) and not args.partial_interval:
        raise ValueError("Forced alignment requires a positive --partial-interval.")
    if not shutil.which("parec"):
        raise ValueError("live capture requires parec (Linux PulseAudio or PipeWire-Pulse)")

    if args.overlay:
        from .overlay import WaylandOverlay

        with WaylandOverlay(args.overlay_font_size, args.overlay_duration) as overlay:
            capture(args, overlay)
    else:
        capture(args)


def capture(args, overlay=None):
    import numpy as np
    from .vad import SileroVAD

    vad = SileroVAD()
    view = None
    if not args.plain and sys.stdout.isatty():
        from .tui import ASRView

        view = ASRView(args.source, args.language, args.url)
    client = create_client(getattr(args, "backend", "llama.cpp"), args.url, args.language, args.timeout,
                           getattr(args, "asr_model", None))
    client.detect_language = bool(getattr(args, "aligner_url", None) and not args.language)
    aligner = None
    if getattr(args, "aligner_url", None):
        from .alignment import ForcedAligner

        aligner = ForcedAligner(args.aligner_url, args.aligner_model, args.language, args.timeout)
    try:
        response = client.session.get(client.url + "/health", timeout=10)
        response.raise_for_status()
        proc = subprocess.Popen([
            "parec", "--format=s16le", "--rate=16000", "--channels=1",
            "--latency-msec=20", "-d", args.source,
        ], stdout=subprocess.PIPE)
    except BaseException:
        client.close()
        if aligner is not None:
            aligner.close()
        raise
    worker = TranscriptionWorker(client, view=view, overlay=overlay, aligner=aligner)
    segmenter = AudioSegmenter(int(args.max_segment * 16000 / 512))
    segment = 0
    last_partial = time.monotonic()

    def submit_final(result, final_text=None):
        nonlocal segment, last_partial
        if result is not None:
            audio, overlap, end = result
            options = {"final_text": final_text} if final_text is not None else {}
            worker.submit(audio, segment, final=True, overlap=overlap, end=end, **options)
        segment += 1
        last_partial = time.monotonic()

    if not view:
        print("Listening; Ctrl+C to stop.", file=sys.stderr)
    try:
        if view:
            view.live.start(refresh=True)
        while True:
            raw = proc.stdout.read(1024)
            if len(raw) != 1024:
                if raw:
                    if view:
                        view.show_error("Discarding incomplete audio frame")
                    else:
                        print("Discarding incomplete audio frame", file=sys.stderr)
                break
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            probability = vad(samples)
            voiced = probability >= 0.5
            segmenter.push(raw, voiced, defer_fallback=True)
            cut = worker.take_cut(segment)
            final_text = None
            if isinstance(cut, AlignedCut):
                result = segmenter.commit(cut.chunks, aligned=True)
                if result is not None:
                    final_text = cut.text
            else:
                result = segmenter.commit(cut) if cut is not None else None
            if result is None:
                result = segmenter.fallback()
            if view:
                view.audio(voiced, len(segmenter.frames) * 0.032)
            if result is not None:
                submit_final(result, final_text)
            elif (args.partial_interval and sum(segmenter.voiced[segmenter.overlap:]) >= 5
                  and time.monotonic() - last_partial >= args.partial_interval):
                worker.submit(b"".join(segmenter.frames), segment, overlap=segmenter.overlap,
                              can_commit=segmenter.can_commit)
                last_partial = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        try:
            if view:
                view.set_status("Finishing")
            submit_final(segmenter.finish())
        finally:
            try:
                worker.close()
            finally:
                if view:
                    view.set_status("Stopped")
                    view.live.stop()

