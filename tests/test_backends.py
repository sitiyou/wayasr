import json
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner

from wayasr import cli
from wayasr.client import LlamaASR
from wayasr.config import config_path, load_config, save_config, save_server_config


def test_audio_client_sends_model():
    client = LlamaASR(model="qwen3-asr")
    client.session = Mock()
    client.session.post.return_value.json.return_value = {"text": "hello"}
    assert client.transcribe(b"aa") == "hello"
    assert client.session.post.call_args.kwargs["data"]["model"] == "qwen3-asr"


def test_audio_server_and_idle_unload(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    settings = {"asr_path": str(model), "aligner_path": str(model), "idle_seconds": 60}
    save_config({"backend": "audio.cpp", "server": settings, "live": {"fa": True}})
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/audiocpp_server")
    args = SimpleNamespace(server_bin="audiocpp_server", host="127.0.0.1", port=8087)
    command = cli.server_command(args)
    assert command == ["/usr/bin/audiocpp_server", "--config", str(config_path().with_name("audio-server.json"))]
    generated = json.loads(config_path().with_name("audio-server.json").read_text())
    assert generated["idle_unload_ms"] == 60000
    assert [model["task"] for model in generated["models"]] == ["asr", "align"]
    save_config({"backend": "audio.cpp", "server": settings, "live": {"fa": False}})
    cli.server_command(args)
    generated = json.loads(config_path().with_name("audio-server.json").read_text())
    assert len(generated["models"]) == 1


def test_llama_idle_unload(monkeypatch, tmp_path):
    (tmp_path / cli.MODEL_NAME).touch()
    (tmp_path / cli.MMPROJ_NAME).touch()
    save_config({"server": {"idle_seconds": 45}})
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/llama-server")
    args = SimpleNamespace(server_bin="llama-server", model=None, mmproj=None,
                           model_dir=tmp_path, device=None, gpu_layers="99",
                           flash_attn="auto", host="localhost", port=8087)
    assert cli.server_command(args)[-2:] == ["--sleep-idle-seconds", "45"]


def test_local_audio_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "available_devices", lambda *args: {"CPU": ("cpu", 0)})
    monkeypatch.setattr(cli.shutil, "which", lambda binary: None)
    model = tmp_path / "model.gguf"
    model.touch()
    answers = ["audio.cpp", "", "Local file", str(model), "y", "Local file", str(model), "CPU", "", "y"]
    result = CliRunner().invoke(cli.main, ["config", "init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    settings = load_config()["server"]
    assert settings["server_bin"] == "audiocpp_server"
    assert settings["idle_seconds"] == 180
    assert settings["asr_path"] == str(model)
    assert "Device (auto lets llama.cpp choose)" not in result.output


def test_external_audio_initialization():
    answers = ["Custom URL", "http://audio:8080", "y"]
    result = CliRunner().invoke(cli.main, ["config", "init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    config = load_config()
    assert config == {"url": "http://audio:8080"}


def test_live_audio_uses_shared_url(monkeypatch):
    import wayasr.live

    run = Mock()
    monkeypatch.setattr(wayasr.live, "run", run)
    save_config({"backend": "audio.cpp", "url": "http://audio", "live": {"fa": True}})
    result = CliRunner().invoke(cli.main, ["live", "--url", "http://override", "--language", "Chinese"])
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.aligner_url == args.url == "http://override"
    assert args.asr_model == "qwen3-asr"


def test_service_save_preserves_audio_settings(tmp_path):
    save_config({"backend": "audio.cpp", "server": {"asr_path": "model.gguf", "idle_seconds": 30}})
    args = SimpleNamespace(host="localhost", port=8087, model_dir=tmp_path, model=None, mmproj=None)
    save_server_config(args, "/usr/bin/audiocpp_server")
    assert load_config()["server"]["asr_path"] == "model.gguf"
    assert load_config()["server"]["idle_seconds"] == 30
