from collections import deque
import threading

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class ASRView:
    def __init__(self, source, language, url, console=None):
        self.console = console or Console()
        self.lock = threading.Lock()
        self.history = deque(maxlen=100)
        self.stable = ""
        self.pending = ""
        self.status = "Listening"
        self.speech = False
        self.buffer_seconds = 0.0
        self.queue_size = 0
        self.timing = "Waiting for transcription"
        self.error = ""
        self.details = f"{language or 'Auto'} · {source} · {url}"
        self.live = Live(get_renderable=self.render, console=self.console, refresh_per_second=8)

    def set_status(self, status):
        with self.lock:
            self.status = status

    def audio(self, speech, seconds):
        with self.lock:
            self.speech = speech
            self.buffer_seconds = seconds

    def partial(self, stable, pending=""):
        with self.lock:
            self.stable = stable
            self.pending = pending

    def sentence(self, text):
        with self.lock:
            self.history.append(text)
            self.stable = self.pending = ""

    def metrics(self, duration, elapsed, queue_size):
        with self.lock:
            self.timing = f"Audio {duration:.2f}s · Inference {elapsed:.2f}s · RTF {elapsed / duration:.3f}"
            self.queue_size = queue_size

    def show_error(self, error):
        with self.lock:
            self.error = str(error)

    def render(self):
        with self.lock:
            height = max(3, min(12, self.console.size.height - 13))
            history = Text("\n".join(list(self.history)[-height:]))
            current = Text(self.stable)
            current.append(self.pending, style="dim")
            status = Text(f"{self.status} · {'Speech' if self.speech else 'Silence'} · Buffer {self.buffer_seconds:.1f}s · Queued {self.queue_size}", style="cyan")
            items = [
                Panel(Group(status, Text(self.details, style="dim")), title="WayASR", border_style="cyan"),
                Panel(history, title="Transcript", height=height + 2),
                Panel(current, title="Live · dim text may change", height=4),
                Text(self.timing, style="green"),
            ]
            if self.error:
                items.append(Text(self.error, style="red"))
            items.append(Text("Ctrl+C stop and finish pending transcription", style="dim"))
            return Group(*items)
