from functools import wraps
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import wave

import click
import requests

from .client import LlamaASR


MODEL_NAME = "Qwen3-ASR-1.7B-Q8_0.gguf"
MMPROJ_NAME = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"


def server_options(command):
    options = [
        click.option("--server-bin", default="llama-server", show_default=True),
        click.option("--model-dir", type=click.Path(path_type=Path), default="models", envvar="PAUSCRIBE_MODEL_DIR", show_default=True),
        click.option("--model", type=click.Path(path_type=Path), help="Override the ASR GGUF path."),
        click.option("--mmproj", type=click.Path(path_type=Path), help="Override the matching projector path."),
        click.option("--device", help="llama.cpp device: ROCm0, CUDA0, Vulkan0, or none for CPU."),
        click.option("--gpu-layers", default="99", show_default=True),
        click.option("--flash-attn", type=click.Choice(["on", "off", "auto"]), default="auto", show_default=True),
        click.option("--host", default="127.0.0.1", show_default=True),
        click.option("--port", type=click.IntRange(1, 65535), default=8087, show_default=True),
        click.option("--clear-device-visibility", is_flag=True, help="Clear GPU visibility filters only for the server."),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def client_options(command):
    for option in reversed([
        click.option("--url", default="http://127.0.0.1:8087", envvar="PAUSCRIBE_URL", show_default=True),
        click.option("--language", help="Omit to detect the language automatically."),
        click.option("--timeout", type=click.FloatRange(min=0, min_open=True), default=120, show_default=True),
    ]):
        command = option(command)
    return command


def command_errors(command):
    @wraps(command)
    def wrapped(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except ModuleNotFoundError as error:
            raise click.ClickException(f"Missing dependency: {error.name}. Reinstall with the live extra.") from error
        except (OSError, ValueError, wave.Error, requests.RequestException, subprocess.CalledProcessError) as error:
            raise click.ClickException(str(error)) from error
    return wrapped


def server_command(args):
    binary = shutil.which(args.server_bin)
    if not binary:
        raise ValueError(f"llama-server not found: {args.server_bin}; install llama.cpp or use --server-bin")
    model = (args.model or args.model_dir / MODEL_NAME).expanduser().resolve()
    mmproj = (args.mmproj or args.model_dir / MMPROJ_NAME).expanduser().resolve()
    for path in (model, mmproj):
        if not path.is_file():
            raise ValueError(f"model file not found: {path}")
    command = [binary, "--model", str(model), "--mmproj", str(mmproj),
               "--n-gpu-layers", "0" if args.device == "none" else args.gpu_layers,
               "--flash-attn", args.flash_attn, "--ctx-size", "4096", "--parallel", "1",
               "--host", args.host, "--port", str(args.port)]
    if args.device:
        command.extend(["--device", args.device, "--mmproj-device", args.device])
    return command


def serve(args):
    command = server_command(args)
    env = os.environ.copy()
    if args.clear_device_visibility:
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            env.pop(name, None)
    proc = subprocess.Popen(command, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """Pause-aware local transcription with llama.cpp."""


@main.command("serve")
@server_options
@command_errors
def serve_command(**options):
    """Start a llama.cpp audio server."""
    raise click.exceptions.Exit(serve(SimpleNamespace(**options)))


@main.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@client_options
@command_errors
def transcribe(audio, url, language, timeout):
    """Transcribe a mono 16 kHz, 16-bit PCM WAV file."""
    with wave.open(str(audio.expanduser()), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("WAV must be mono, 16-bit PCM, 16000 Hz; convert with ffmpeg first")
        pcm = wav.readframes(wav.getnframes())
    client = LlamaASR(url, language, timeout)
    try:
        click.echo(client.transcribe(pcm))
    finally:
        client.close()


@main.command()
@client_options
@click.option("--source", default="@DEFAULT_MONITOR@", show_default=True)
@click.option("--plain", is_flag=True, help="Disable the Rich live display.")
@click.option("--partial-interval", type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option("--max-segment", type=click.FloatRange(1, 25), default=15.0, show_default=True)
@command_errors
def live(**options):
    """Live Linux system audio or microphone captions."""
    from .live import run

    run(SimpleNamespace(**options))


@main.group()
def service():
    """Manage the systemd user service without sudo."""


@service.command("install")
@server_options
@command_errors
def install_service(**options):
    """Install, enable, and start (or restart) the user service."""
    from .service import manage

    manage(SimpleNamespace(action="install", **options))


@service.command("uninstall")
@command_errors
def uninstall_service():
    """Stop, disable, and remove the user service; keep model files."""
    from .service import manage

    manage(SimpleNamespace(action="uninstall"))
