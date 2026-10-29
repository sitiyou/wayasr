from functools import wraps
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import wave

import click
import requests

from .client import LlamaASR
from .config import config_path, default_model_dir, load_config, save_config, server_config


MODEL_NAME = "Qwen3-ASR-1.7B-Q8_0.gguf"
MMPROJ_NAME = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"


def server_options(command):
    options = [
        click.option("--server-bin", default="llama-server", show_default=True),
        click.option("--model-dir", type=click.Path(path_type=Path), default=default_model_dir, envvar="WAYASR_MODEL_DIR", show_default="user data directory/models"),
        click.option("--model", type=click.Path(path_type=Path), help="Override the ASR GGUF path."),
        click.option("--mmproj", type=click.Path(path_type=Path), help="Override the matching projector path."),
        click.option("--device", help="llama.cpp device: ROCm0, CUDA0, Vulkan0, or none for CPU."),
        click.option("--gpu-layers", default="99", show_default=True),
        click.option("--flash-attn", type=click.Choice(["on", "off", "auto"]), default="auto", show_default=True),
        click.option("--host", default="127.0.0.1", show_default=True),
        click.option("--port", type=click.IntRange(1, 65535), default=8087, show_default=True),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def client_options(command):
    for option in reversed([
        click.option("--url", default="http://127.0.0.1:8087", envvar="WAYASR_URL", show_default=True),
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


AUDIO_ASR_MODEL = "Qwen3-ASR-1.7B Q8_0"
AUDIO_ALIGNER_MODEL = "Qwen3-ForcedAligner-0.6B Q8_0"


def resolve_audio_model(model, *, aligner=False):
    if Path(model).is_absolute():
        return model

    from huggingface_hub import hf_hub_download

    preset = AUDIO_ALIGNER_MODEL if aligner else AUDIO_ASR_MODEL
    if model != preset:
        raise ValueError(f"Unknown model: {model}. Select a preset or a local model with 'wayasr config init'.")
    filename = (
        "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf"
        if aligner else "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf"
    )
    return hf_hub_download(repo_id="audio-cpp/audio.cpp-gguf", filename=filename)


def server_command(args):
    config = load_config()
    settings = config.get("server", {})
    if config.get("backend", "llama.cpp") == "audio.cpp":
        import json
        from .config import write_text_atomic

        binary = shutil.which(args.server_bin)
        if not binary:
            raise ValueError(f"audiocpp_server not found: {args.server_bin}")
        required = ["asr_path"]
        if config.get("live", {}).get("fa", True):
            required.append("aligner_path")
        settings = settings.copy()
        for name in required:
            default = AUDIO_ASR_MODEL if name == "asr_path" else AUDIO_ALIGNER_MODEL
            settings[name] = resolve_audio_model(settings.get(name) or default, aligner=name == "aligner_path")
        models = [{"id": settings.get("asr_model", "qwen3-asr"), "family": "qwen3_asr",
                   "path": str(Path(settings["asr_path"]).expanduser().resolve()), "task": "asr", "mode": "offline"}]
        if config.get("live", {}).get("fa", True):
            models.append({"id": settings.get("aligner_model", "qwen3-align"), "family": "qwen3_forced_aligner",
                           "path": str(Path(settings["aligner_path"]).expanduser().resolve()), "task": "align", "mode": "offline"})
        for model in models:
            if not Path(model["path"]).exists():
                raise ValueError(f"model file not found: {model['path']}")
        path = config_path().with_name("audio-server.json")
        write_text_atomic(path, json.dumps({"host": args.host, "port": args.port,
                          "backend": settings.get("audio_backend", "cpu"), "lazy_load": True,
                          "device": settings.get("audio_device", 0),
                          "idle_unload_ms": settings.get("idle_seconds", 180) * 1000, "models": models}, indent=2) + "\n")
        return [binary, "--config", str(path)]
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
    idle = settings.get("idle_seconds", 180)
    if idle:
        command.extend(["--sleep-idle-seconds", str(idle)])
    return command


def serve(args):
    command = server_command(args)
    env = os.environ.copy()
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
@click.pass_context
@command_errors
def main(ctx):
    """WayASR: pause-aware local transcription."""
    config = load_config()
    client = {key: value for key, value in config.get("client", {}).items() if key != "language"}
    server = {key: value for key, value in config.get("server", {}).items() if key != "clear_device_visibility"}
    live_defaults = {key: value for key, value in config.get("live", {}).items()
                     if key not in ("language", "source", "overlay", "aligner_url", "aligner_model")}
    if "url" in config:
        client["url"] = config["url"]
    ctx.default_map = {
        "serve": server,
        "service": {"install": server},
        "transcribe": client,
        "live": {**client, **live_defaults},
    }


@main.command("serve")
@server_options
@command_errors
def serve_command(**options):
    """Start the configured audio server."""
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
    config = load_config()
    if config.get("backend") == "audio.cpp":
        client.model = config.get("server", {}).get("asr_model", "qwen3-asr")
    try:
        click.echo(client.transcribe(pcm))
    finally:
        client.close()


@main.command()
@client_options
@click.option("--source", default="@DEFAULT_MONITOR@", show_default=True)
@click.option("--plain", is_flag=True, help="Disable the Rich live display.")
@click.option("--overlay/--no-overlay", default=False, help="Show Wayland desktop captions with asslayer (Python 3.14+).")
@click.option("--overlay-font-size", type=click.IntRange(12, 120), default=42, show_default=True)
@click.option("--overlay-duration", type=click.IntRange(1, 60), default=8, show_default=True, help="Seconds before an unchanged desktop caption expires.")
@click.option("--partial-interval", type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option("--max-segment", type=click.FloatRange(1, 25), default=15.0, show_default=True)
@command_errors
def live(**options):
    """Live Linux system audio or microphone captions."""
    from .live import run

    config = load_config()
    settings = config.get("server", {})
    if config.get("backend") == "audio.cpp":
        options["asr_model"] = settings.get("asr_model", "qwen3-asr")
        if config.get("live", {}).get("fa", True):
            options["aligner_url"] = options["url"]
            options["aligner_model"] = settings.get("aligner_model", "qwen3-align")
    run(SimpleNamespace(**options))


def confirm(question, *, default=False, abort=False):
    if sys.stdin.isatty():
        import beaupy

        answer = beaupy.confirm(question, default_is_yes=default)
        if answer is None or (abort and not answer):
            raise click.Abort()
        return answer
    return click.confirm(question, default=default, abort=abort)


def select_option(title, options, default):
    if sys.stdin.isatty():
        import beaupy

        answer = beaupy.select(options, title=title, cursor_index=options.index(default))
        if answer is None:
            raise click.Abort()
        return answer
    return click.prompt(title, type=click.Choice(options), default=default)


def select_audio_model(title, saved, preset):
    local = "Local file"
    choice = select_option(title, [preset, local], local if saved and Path(saved).is_absolute() else preset)
    if choice != local:
        return choice
    while True:
        value = click.prompt("Model file", default=saved if saved and Path(saved).is_absolute() else "")
        path = Path(value).expanduser()
        if path.is_absolute() and path.is_file():
            return str(path)
        click.echo("Enter an absolute path to an existing model file.")


def available_devices(server_bin, backend):
    try:
        result = subprocess.run([server_bin, "--list-devices"], capture_output=True, text=True, check=True, timeout=10)
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Timed out querying {server_bin} --list-devices") from error
    devices = {}
    for line in (result.stdout + result.stderr).splitlines():
        if backend == "audio.cpp":
            match = re.match(r'\s*(CPU|CUDA|HIP|ROCM|VULKAN|METAL):(\d+)\s+(.*)', line, re.I)
            if match:
                kind, index, description = match.groups()
                kind = "hip" if kind.lower() == "rocm" else kind.lower()
                devices[f"{kind}:{index} — {description}"] = (kind, int(index))
        else:
            match = re.match(r'\s*(\w+\d+):\s+(.+)', line)
            if match:
                name, description = match.groups()
                devices[f"{name} — {description}"] = name
    if backend == "llama.cpp":
        devices["CPU"] = "none"
    if not devices:
        raise ValueError(f"No devices found. Check '{server_bin} --list-devices'.")
    return devices


@main.group("config")
def config_group():
    """Initialize or inspect persistent user defaults."""


@config_group.command("init")
@click.pass_context
@command_errors
def init_config(ctx):
    """Interactively configure server and client defaults without deploying."""
    previous = load_config()
    path = config_path()
    if path.exists():
        confirm("Update configuration?", default=True, abort=True)
    custom = "Custom URL"
    choices = ["llama.cpp", "audio.cpp", custom]
    default = previous.get("backend", custom if previous.get("url") else "llama.cpp")
    backend = select_option("Service", choices, default)
    config = {}
    if backend == custom:
        config["url"] = click.prompt("Service URL", default=previous.get("url", "http://127.0.0.1:8087"))
    else:
        saved = previous.get("server", {}) if previous.get("backend", "llama.cpp") == backend else {}
        with serve_command.make_context("serve", [], parent=ctx.find_root(), default_map=saved) as server_ctx:
            options = server_ctx.params.copy()
        options["host"] = "127.0.0.1"
        binary = "audiocpp_server" if backend == "audio.cpp" else "llama-server"
        options["server_bin"] = click.prompt("Server executable", default=saved.get("server_bin", binary))
        settings = {}
        if backend == "llama.cpp":
            options["model_dir"] = Path(click.prompt("Model directory", default=str(options["model_dir"])))
        else:
            settings["asr_path"] = select_audio_model("ASR model", saved.get("asr_path"), AUDIO_ASR_MODEL)
            fa = confirm("Enable forced alignment?", default=previous.get("live", {}).get("fa", True))
            config["live"] = {"fa": fa}
            if fa:
                settings["aligner_path"] = select_audio_model("Alignment model", saved.get("aligner_path"), AUDIO_ALIGNER_MODEL)
        devices = available_devices(options["server_bin"], backend)
        selected = saved.get("device") if backend == "llama.cpp" else (saved.get("audio_backend"), saved.get("audio_device", 0))
        default_device = next((label for label, value in devices.items() if value == selected), next(iter(devices)))
        device = devices[select_option("Device", list(devices), default_device)]
        if backend == "llama.cpp":
            options["device"] = device
        else:
            settings["audio_backend"], settings["audio_device"] = device
        options["port"] = click.prompt("Port", default=options["port"], type=click.IntRange(1, 65535))
        config.update(server_config(SimpleNamespace(**options), shutil.which(options["server_bin"])))
        config["backend"] = backend
        config["server"].update(settings)
        config["server"]["idle_seconds"] = saved.get("idle_seconds", 180)
    click.echo(f"Service: {config['url']}")
    click.echo(f"Save to {path}")
    if not confirm("Save configuration?", default=True):
        raise click.Abort()
    save_config(config)
    if backend == custom:
        click.echo("Saved. Ready to transcribe.")
    else:
        click.echo("Saved. Run 'wayasr service install' to start the service.")


@config_group.command("show")
@command_errors
def show_config():
    """Show the configuration file path and saved values."""
    import json

    click.echo(str(config_path()))
    click.echo(json.dumps(load_config(), ensure_ascii=False, indent=2))


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
