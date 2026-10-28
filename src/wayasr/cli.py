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
        click.option("--clear-device-visibility/--keep-device-visibility", default=False, help="Clear or preserve GPU visibility filters only for the server."),
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
        for name in required:
            if not settings.get(name):
                raise ValueError(f"Configure server.{name} with 'wayasr config init' before starting a local audio.cpp server")
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
                          "idle_unload_ms": settings.get("idle_seconds", 0) * 1000, "models": models}, indent=2) + "\n")
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
    idle = settings.get("idle_seconds", 0)
    if idle:
        command.extend(["--sleep-idle-seconds", str(idle)])
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
@click.pass_context
@command_errors
def main(ctx):
    """WayASR: pause-aware local transcription."""
    config = load_config()
    client = config.get("client", {}).copy()
    if "url" in config:
        client["url"] = config["url"]
    ctx.default_map = {
        "serve": config.get("server", {}),
        "service": {"install": config.get("server", {})},
        "transcribe": client,
        "live": {**client, **config.get("live", {})},
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


@main.group("config")
def config_group():
    """Initialize or inspect persistent user defaults."""


@config_group.command("init")
@click.pass_context
@command_errors
def init_config(ctx):
    """Interactively configure server and client defaults without deploying."""
    config = load_config()
    path = config_path()
    if path.exists():
        click.confirm(f"Update configuration at {path}?", default=True, abort=True)
    backend = click.prompt("Backend", type=click.Choice(["llama.cpp", "audio.cpp"]), default=config.get("backend", "llama.cpp"))
    previous_backend = config.get("backend", "llama.cpp")
    config["backend"] = backend
    config.setdefault("live", {}).pop("aligner_url", None)
    config["live"].pop("aligner_model", None)
    config["live"]["fa"] = backend == "audio.cpp" and click.confirm("Enable forced alignment?", default=config["live"].get("fa", True) if previous_backend == backend else True)
    local_server = click.confirm("Configure a local server?", default="server" in config or "url" not in config)
    url_default = config.get("url", "http://127.0.0.1:8087")
    if local_server:
        with serve_command.make_context("serve", [], parent=ctx.find_root()) as server_ctx:
            options = server_ctx.params.copy()
        previous_url = server_config(SimpleNamespace(**options))["url"]
        saved_url = config.get("url")
        options["server_bin"] = click.prompt("Server executable", default=config.get("server", {}).get("server_bin", "audiocpp_server" if backend == "audio.cpp" else "llama-server") if previous_backend == backend else ("audiocpp_server" if backend == "audio.cpp" else "llama-server"))
        if backend == "llama.cpp":
            options["model_dir"] = Path(click.prompt("Model directory", default=str(options["model_dir"])))
            device = click.prompt("Device (auto lets llama.cpp choose)", default=options["device"] or "auto")
            options["device"] = None if device == "auto" else device
        options["clear_device_visibility"] = click.confirm("Clear GPU visibility filters for the server?", default=options["clear_device_visibility"])
        options["host"] = click.prompt("Listen address", default=options["host"])
        options["port"] = click.prompt("Port", default=options["port"], type=click.IntRange(1, 65535))
        updated = server_config(SimpleNamespace(**options), shutil.which(options["server_bin"]))
        updated["server"] = {**config.get("server", {}), **updated["server"]}
        config.update(updated)
        url_default = saved_url if saved_url and saved_url != previous_url else config["url"]
    settings = config.setdefault("server", {})
    settings["backend"] = backend
    if backend == "audio.cpp":
        settings["asr_model"] = click.prompt("ASR model ID", default=settings.get("asr_model", "qwen3-asr"))
        if config["live"]["fa"]:
            settings["aligner_model"] = click.prompt("FA model ID", default=settings.get("aligner_model", "qwen3-align"))
        if local_server:
            settings["audio_backend"] = click.prompt("audio.cpp compute backend", default=settings.get("audio_backend", "cpu"))
            settings["asr_path"] = click.prompt("audio.cpp ASR model path", default=settings.get("asr_path", ""))
            if config["live"]["fa"]:
                settings["aligner_path"] = click.prompt("audio.cpp FA model path", default=settings.get("aligner_path", ""))
    if local_server:
        settings["idle_seconds"] = click.prompt("Idle unload seconds (0 disables)", default=settings.get("idle_seconds", 0), type=click.IntRange(min=0))
    config["url"] = click.prompt("Client service URL", default=url_default)
    language = click.prompt("Language (auto detects it)", default=config.get("client", {}).get("language") or "auto")
    while config["live"]["fa"] and language == "auto":
        click.echo("Forced alignment requires an explicit language.")
        language = click.prompt("Language")
    config.setdefault("client", {})["language"] = None if language == "auto" else language
    config.setdefault("live", {})["source"] = click.prompt("Live audio source", default=config.get("live", {}).get("source", "@DEFAULT_MONITOR@"))
    config["live"]["overlay"] = click.confirm("Enable Wayland desktop captions (Linux, Python 3.14+)?", default=config["live"].get("overlay", False))
    click.echo(f"Save to {path}")
    click.echo(f"Service URL: {config['url']}")
    if not click.confirm("Save configuration?", default=True):
        raise click.Abort()
    save_config(config)
    if local_server:
        click.echo("Saved. Run 'wayasr service install' to deploy the configured server.")
    else:
        click.echo("Saved. Transcription commands will use the configured API.")


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
