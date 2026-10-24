import os
from pathlib import Path
import subprocess
import sys
import wave
from unittest.mock import Mock

from click.testing import CliRunner

from pauscribe import cli


def server_args(tmp_path, monkeypatch, *options):
    captured = []
    monkeypatch.setattr(cli, "serve", lambda args: captured.append(args) or 0)
    result = CliRunner().invoke(cli.main, ["serve", "--model-dir", str(tmp_path), *options])
    assert result.exit_code == 0, result.output
    return captured[0]


def test_defaults_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PAUSCRIBE_URL", "http://example.test:9000")
    monkeypatch.setenv("PAUSCRIBE_MODEL_DIR", "/tmp/asr-models")
    captured = []
    monkeypatch.setattr(cli, "serve", lambda args: captured.append(args) or 0)
    assert CliRunner().invoke(cli.main, ["serve"]).exit_code == 0
    assert captured[0].model_dir == Path("/tmp/asr-models")
    from pauscribe import live

    monkeypatch.setattr(live, "run", lambda args: captured.append(args))
    assert CliRunner().invoke(cli.main, ["live"]).exit_code == 0
    assert captured[-1].url == "http://example.test:9000"
    assert CliRunner().invoke(cli.main, ["live", "--url", "http://localhost:8000"]).exit_code == 0
    assert captured[-1].url == "http://localhost:8000"


def test_server_cpu_and_visibility_are_scoped_to_child(tmp_path, monkeypatch):
    serve = cli.serve
    args = server_args(tmp_path, monkeypatch, "--device", "none")
    for name in (cli.MODEL_NAME, cli.MMPROJ_NAME):
        (tmp_path / name).touch()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(cli.shutil, "which", lambda binary: "/usr/bin/llama-server")
    process = Mock()
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert serve(args) == 0
    command = popen.call_args.args[0]
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert command[command.index("--mmproj-device") + 1] == "none"
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    args.clear_device_visibility = True
    serve(args)
    assert "CUDA_VISIBLE_DEVICES" not in popen.call_args.kwargs["env"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_server_missing_model_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda binary: "/usr/bin/llama-server")
    result = CliRunner().invoke(cli.main, ["serve", "--model-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "model file not found" in result.output


def test_wav_transcription_passes_options(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    pcm = b"\x00\x00" * 512
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    client = Mock()
    client.transcribe.return_value = "你好"
    factory = Mock(return_value=client)
    monkeypatch.setattr(cli, "LlamaASR", factory)
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--language", "Chinese", "--timeout", "30"])
    assert result.exit_code == 0, result.output
    factory.assert_called_once_with("http://127.0.0.1:8087", "Chinese", 30)
    client.transcribe.assert_called_once_with(pcm)
    client.close.assert_called_once()
    assert result.output == "你好\n"


def test_cli_help_does_not_import_live_dependencies():
    code = """
import sys
from click.testing import CliRunner
from pauscribe.cli import main
result = CliRunner().invoke(main, ['live', '--help'])
assert result.exit_code == 0
assert '--plain' in result.output
assert not any(name in sys.modules for name in ('torch', 'numpy', 'silero_vad', 'rich'))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_invalid_option_is_a_click_usage_error():
    runner = CliRunner()
    for arguments in (["live", "--max-segment", "30"], ["serve", "--port", "0"],
                      ["live", "--timeout", "0"], ["live", "--partial-interval", "-1"]):
        result = runner.invoke(cli.main, arguments)
        assert result.exit_code == 2
        assert "Invalid value" in result.output
