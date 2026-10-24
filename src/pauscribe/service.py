import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .cli import server_command


UNIT = "pauscribe.service"


def quote_argument(value):
    if any(character in value for character in "\n\r\x00"):
        raise ValueError("service arguments cannot contain newlines or NUL")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def render_unit(args):
    command = server_command(args)
    command[0] = str(Path(command[0]).absolute())
    lines = [
        "[Unit]",
        "Description=Pauscribe local transcription server",
        "After=default.target",
        "",
        "[Service]",
        "Type=simple",
        "ExecStart=" + " ".join(quote_argument(value) for value in command),
        "Restart=on-failure",
        "RestartSec=3",
        "TimeoutStopSec=20",
        "KillMode=control-group",
    ]
    if args.clear_device_visibility:
        lines.append("UnsetEnvironment=CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL")
    return "\n".join(lines + ["", "[Install]", "WantedBy=default.target", ""])


def manage(args):
    if sys.platform != "linux" or not shutil.which("systemctl"):
        raise ValueError("service management requires Linux with systemd and a running user manager")
    subprocess.run(["systemctl", "--user", "show-environment"], check=True, stdout=subprocess.DEVNULL)
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser()
    if not config.is_absolute():
        raise ValueError("XDG_CONFIG_HOME must be an absolute path")
    unit = config / "systemd" / "user" / UNIT
    if args.action == "install":
        content = render_unit(args)
        unit.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=unit.parent, delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(content)
                file.flush()
                temporary.chmod(0o644)
                os.replace(temporary, unit)
            finally:
                temporary.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", UNIT], check=True)
        subprocess.run(["systemctl", "--user", "restart", UNIT], check=True)
        print(f"Installed and started {unit}")
        if any(unit.with_name(UNIT + ".d").glob("*.conf")):
            print("Existing systemd drop-ins remain active and may override these settings.")
        print(f"Status: systemctl --user status {UNIT}")
    else:
        if not unit.exists() and not unit.is_symlink():
            print(f"{UNIT} is not installed at {unit}")
            return 0
        subprocess.run(["systemctl", "--user", "disable", "--now", UNIT], check=True)
        unit.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        print(f"Removed {unit}; models and any custom drop-ins were kept")
    return 0
