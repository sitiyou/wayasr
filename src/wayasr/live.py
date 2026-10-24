import shutil
from collections import deque
import queue
import re
import subprocess
import sys
import threading
import time

from .client import LlamaASR


SENTENCE_BOUNDARY = re.compile(r"[。！？]+[”’\"]*|(?<!\d)[.!?]+[\"']*(?=\s|$)")


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

    def push(self, raw, voiced):
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
    def at_boundary(self):
        return len(self.frames) >= self.soft_chunks and self.silence >= 4

    def commit(self, cut):
        if (not self.soft_chunks <= cut <= len(self.frames)
                or any(self.voiced[cut - 4:cut])
                or sum(self.voiced[self.overlap:cut]) < 5):
            return None
        return self._cut(cut)

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
    def __init__(self, client, max_pending=16, view=None, overlay=None):
        self.client = client
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

    def submit(self, audio, segment, final=False, overlap=0, end=True, boundary=False):
        with self.condition:
            self.segment = segment
            if final:
                self.partial = None
                self.finals.put_nowait((audio, segment, overlap, end, boundary))
                self.finalized_segment = segment
            else:
                self.partial = (audio, segment, overlap, end, boundary)
            self.condition.notify()

    def take_cut(self, segment):
        while True:
            try:
                candidate, cut = self.cuts.get_nowait()
            except queue.Empty:
                return None
            if candidate == segment:
                return cut

    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.closed or not self.finals.empty() or self.partial is not None
                    )
                    if not self.finals.empty():
                        audio, segment, overlap, end, boundary = self.finals.get_nowait()
                        final = True
                    elif self.partial is not None:
                        audio, segment, overlap, end, boundary = self.partial
                        self.partial = None
                        final = False
                    else:
                        return
                started = time.perf_counter()
                if self.view:
                    self.view.set_status("Transcribing")
                try:
                    text = self.client.transcribe(audio)
                    with self.condition:
                        stale = segment != self.segment or segment <= self.finalized_segment or self.closed
                        if final or not stale:
                            self.output.update(text, final, overlap, end)
                            if final:
                                self.previous_partial = None
                            else:
                                if (boundary and self.previous_partial == (segment, text)
                                        and any(match.end() == len(text)
                                                for match in SENTENCE_BOUNDARY.finditer(text))):
                                    self.cuts.put((segment, len(audio) // 1024))
                                self.previous_partial = (segment, text)
                    if final:
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
            self.output.flush()
            self.client.close()

    def close(self):
        with self.condition:
            self.closed = True
            self.partial = None
            self.condition.notify()
        self.thread.join()


def run(args):
    if args.partial_interval < 0 or not 1 <= args.max_segment <= 25:
        raise ValueError("partial interval must be nonnegative; max segment must be 1–25 seconds")
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
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    vad = load_silero_vad(onnx=True)
    view = None
    if not args.plain and sys.stdout.isatty():
        from .tui import ASRView

        view = ASRView(args.source, args.language, args.url)
    client = LlamaASR(args.url, args.language, args.timeout)
    try:
        response = client.session.get(client.url + "/health", timeout=10)
        response.raise_for_status()
        proc = subprocess.Popen([
            "parec", "--format=s16le", "--rate=16000", "--channels=1",
            "--latency-msec=20", "-d", args.source,
        ], stdout=subprocess.PIPE)
    except BaseException:
        client.close()
        raise
    worker = TranscriptionWorker(client, view=view, overlay=overlay)
    segmenter = AudioSegmenter(int(args.max_segment * 16000 / 512))
    segment = 0
    last_partial = time.monotonic()

    def submit_final(result):
        nonlocal segment, last_partial
        if result is not None:
            audio, overlap, end = result
            worker.submit(audio, segment, final=True, overlap=overlap, end=end)
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
            probability = float(vad(torch.from_numpy(samples), 16000).item())
            voiced = probability >= 0.5
            result = segmenter.push(raw, voiced)
            if view:
                view.audio(voiced, len(segmenter.frames) * 0.032)
            if result is not None:
                submit_final(result)
            else:
                cut = worker.take_cut(segment)
                committed = segmenter.commit(cut) if cut is not None else None
                if committed is not None:
                    submit_final(committed)
                elif (args.partial_interval and sum(segmenter.voiced[segmenter.overlap:]) >= 5
                      and ((segmenter.at_boundary and segmenter.silence == 4)
                           or time.monotonic() - last_partial >= args.partial_interval)):
                    worker.submit(b"".join(segmenter.frames), segment, overlap=segmenter.overlap,
                                  boundary=segmenter.at_boundary)
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

