import os
import sys
import time
import unicodedata


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,sans-serif,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,480,480,64,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_text(text, pending="", max_chars=22):
    output = []
    column = 0
    for part, color in ((text, ""), (pending, r"{\c&HAAAAAA&}")):
        if not part:
            continue
        if color:
            output.append(color)
        part = (part.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
                .replace("\r\n", "\n").replace("\r", "\n")
                .replace("\t", " ").replace("\x00", ""))
        for character in part:
            if character == "\n":
                output.append(r"\N")
                column = 0
                continue
            if unicodedata.category(character) not in ("Mn", "Me", "Cf"):
                if column >= max_chars:
                    output.append(r"\N")
                    column = 0
                column += 1
            output.append(character)
    return "".join(output)


class WaylandOverlay:
    def __init__(self, font_size=42, duration=8):
        if sys.platform != "linux" or sys.version_info < (3, 14):
            raise ValueError("Wayland overlay requires Linux and Python 3.14+")
        if not os.environ.get("WAYLAND_DISPLAY"):
            raise ValueError("Wayland overlay requires a running Wayland session (WAYLAND_DISPLAY)")
        if not 12 <= font_size <= 120 or not 1 <= duration <= 60:
            raise ValueError("Overlay font size must be 12–120; duration must be 1–60 seconds")
        try:
            from asslayer import ApplicationController
        except ImportError as error:
            raise ValueError(f"Cannot load asslayer; install '.[live,wayland]' and its native libraries: {error}") from error

        self.controller = ApplicationController(namespace="wayasr")
        self.header = ASS_HEADER.format(font_size=font_size)
        self.max_chars = max(1, 960 // font_size)
        self.end_time = f"0:{duration // 60:02d}:{duration % 60:02d}.00"
        self.final_text = ""

    def __enter__(self):
        try:
            self.controller.set_ass_input(self.header)
            self.controller.set_status(paused=False, hidden=False, locked=True, target_fps=30,
                                       start_time_ms=int(time.monotonic() * 1000))
            self.controller.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.controller.stop()
        finally:
            self.controller.join()

    def partial(self, stable, pending=""):
        if stable or pending:
            text = ass_text(stable, pending, self.max_chars)
        else:
            text = ass_text(self.final_text, max_chars=self.max_chars)
        self.render(text)

    def sentence(self, text):
        self.final_text = text
        self.render(ass_text(text, max_chars=self.max_chars))

    def render(self, text):
        content = self.header
        if text:
            content += f"Dialogue: 0,0:00:00.00,{self.end_time},Default,,0,0,0,,{text}\n"
        self.controller.set_ass_input(content)
        self.controller.set_status(start_time_ms=int(time.monotonic() * 1000))
