from pathlib import Path
import shutil
import subprocess
import sys

from .cli import server_command
from .config import config_home, save_server_config, write_text_atomic


UNIT = "wayasr.service"


def quote_argument(value):
    if any(character in value for character in "\n\r\x00"):
        raise ValueError("service arguments cannot contain newlines or NUL")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def render_unit(args):
    command = server_command(args)
    command[0] = str(Path(command[0]).absolute())
    lines = [
        "[Unit]",
        "Description=WayASR local transcription server",
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
    unit = config_home() / "systemd" / "user" / UNIT
    if args.action == "install":
        content = render_unit(args)
        write_text_atomic(unit, content, mode=0o644)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", UNIT], check=True)
        subprocess.run(["systemctl", "--user", "restart", UNIT], check=True)
        save_server_config(args, shutil.which(args.server_bin))
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
