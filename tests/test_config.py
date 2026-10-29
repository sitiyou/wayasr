import json
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from wayasr import cli, config, live


def test_platformdirs_paths_respect_xdg(tmp_path):
    assert config.config_path() == tmp_path / "user-config/wayasr/config.json"
    assert config.default_model_dir() == tmp_path / "user-data/wayasr/models"


def test_client_default_precedence(monkeypatch):
    config.save_config({"url": "http://localhost:9000", "client": {"language": "Chinese"}})
    captured = []
    monkeypatch.setattr(live, "run", lambda args: captured.append(args))
    runner = CliRunner()
    assert runner.invoke(cli.main, ["live"]).exit_code == 0
    assert captured[-1].url == "http://localhost:9000"
    assert captured[-1].language is None
    monkeypatch.setenv("WAYASR_URL", "http://localhost:9001")
    assert runner.invoke(cli.main, ["live"]).exit_code == 0
    assert captured[-1].url == "http://localhost:9001"
    assert runner.invoke(cli.main, ["live", "--url", "http://localhost:9002"]).exit_code == 0
    assert captured[-1].url == "http://localhost:9002"
    assert config.load_config()["url"] == "http://localhost:9000"
    client = Mock()
    client.transcribe.return_value = "text"
    factory = Mock(return_value=client)
    monkeypatch.setattr(cli, "LlamaASR", factory)
    import wave

    audio = config.config_path().parent / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 512)
    monkeypatch.delenv("WAYASR_URL")
    result = runner.invoke(cli.main, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output
    factory.assert_called_once_with("http://localhost:9000", None, 120)


def test_saved_server_defaults_and_explicit_override(monkeypatch):
    config.save_config({"server": {"port": 9000, "device": "ROCm0", "clear_device_visibility": True}})
    captured = []
    monkeypatch.setattr(cli, "serve", lambda args: captured.append(args) or 0)
    runner = CliRunner()
    assert runner.invoke(cli.main, ["serve"]).exit_code == 0
    assert captured[-1].port == 9000
    assert not hasattr(captured[-1], "clear_device_visibility")
    result = runner.invoke(cli.main, ["serve", "--port", "9001"])
    assert result.exit_code == 0, result.output
    assert captured[-1].port == 9001
    assert not hasattr(captured[-1], "clear_device_visibility")


def test_interactive_initialization_and_show(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.shutil, "which", lambda binary: None)
    models = tmp_path / "model directory"
    monkeypatch.setattr(cli, "available_devices", lambda *args: {"CPU": "none"})
    answers = ["llama.cpp", "llama-server", str(models), "CPU", "9000", "y"]
    runner = CliRunner()
    result = runner.invoke(cli.main, ["config", "init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    saved = config.load_config()
    assert saved["url"] == "http://127.0.0.1:9000"
    assert saved["server"]["port"] == 9000
    assert saved["server"]["model_dir"] == str(models)
    assert saved["server"]["server_bin"] == "llama-server"
    assert "client" not in saved
    assert "live" not in saved
    assert saved["server"]["idle_seconds"] == 180
    assert not (config.config_home() / "systemd/user/wayasr.service").exists()
    shown = runner.invoke(cli.main, ["config", "show"])
    assert shown.exit_code == 0
    assert str(config.config_path()) in shown.output
    assert '"port": 9000' in shown.output


@pytest.mark.parametrize("existing", [{}, {"server": {"port": 9000}, "url": "http://localhost:9000"}])
def test_init_existing_api_skips_server_setup(monkeypatch, existing):
    if existing:
        config.save_config(existing)
    monkeypatch.setattr(cli.shutil, "which", Mock(side_effect=AssertionError("Must not resolve a local server")))
    answers = (["y"] if existing else []) + ["Custom URL", "https://asr.example.test", "y"]
    result = CliRunner().invoke(cli.main, ["config", "init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    saved = config.load_config()
    assert saved["url"] == "https://asr.example.test"
    assert saved == {"url": "https://asr.example.test"}
    assert "Model directory" not in result.output
    assert "service install" not in result.output
    assert "Ready to transcribe" in result.output


def test_init_abort_keeps_previous_config():
    config.save_config({"url": "http://localhost:9000"})
    before = config.config_path().read_bytes()
    result = CliRunner().invoke(cli.main, ["config", "init"], input="n\n")
    assert result.exit_code != 0
    assert config.config_path().read_bytes() == before


def test_reinitialization_preserves_remote_url_and_can_cancel():
    config.save_config({"url": "https://remote.example.test:9000"})
    runner = CliRunner()
    result = runner.invoke(cli.main, ["config", "init"], input="y\nCustom URL\n\ny\n")
    assert result.exit_code == 0, result.output
    assert config.load_config()["url"] == "https://remote.example.test:9000"
    before = config.config_path().read_bytes()
    result = runner.invoke(cli.main, ["config", "init"], input="y\nCustom URL\n\nn\n")
    assert result.exit_code != 0
    assert config.config_path().read_bytes() == before


def test_init_cancel_does_not_save():
    result = CliRunner().invoke(cli.main, ["config", "init"], input="Custom URL\nhttp://remote\nn\n")
    assert result.exit_code != 0
    assert not config.config_path().exists()


@pytest.mark.parametrize("content", ["{invalid", "[]", '{"server": []}', '{"url": 123}', '{"client": []}'])
def test_invalid_config_is_actionable(content):
    path = config.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(content)
    result = CliRunner().invoke(cli.main, ["live"])
    assert result.exit_code == 1
    assert str(path) in result.output
    assert "Traceback" not in result.output


def test_atomic_config_write_leaves_no_temporary_files():
    config.save_config({"url": "http://localhost:9000"})
    config.save_config({"url": "http://localhost:9001"})
    assert json.loads(config.config_path().read_text())["url"] == "http://localhost:9001"
    assert list(config.config_path().parent.iterdir()) == [config.config_path()]
