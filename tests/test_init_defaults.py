from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from wayasr import cli, config, live
from wayasr.client import AudioCppASR
from wayasr.live import TranscriptionWorker


@pytest.mark.parametrize("fa", [True, False])
def test_audio_init_only_asks_for_selected_models(monkeypatch, fa):
    monkeypatch.setattr(cli, "available_devices", lambda *args: {"GPU": ("cuda", 1)})
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    answers = ["Local service", "audio.cpp", "", "", "y" if fa else "n"]
    if fa:
        answers.append("")
    answers += ["GPU", "", "y"]
    result = CliRunner().invoke(cli.main, ["config", "init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    saved = config.load_config()
    assert saved["url"] == "http://127.0.0.1:8087"
    assert saved["server"]["audio_device"] == 1
    assert saved["server"]["audio_backend"] == "cuda"
    assert saved["server"]["idle_seconds"] == 180
    assert ("aligner_path" in saved["server"]) is fa
    assert saved["live"] == {"fa": fa}
    for prompt in ("model ID", "Language", "Live audio source", "Listen address", "Client service URL", "Service URL", "Idle unload", "Wayland"):
        assert prompt not in result.output


def test_custom_route_removes_old_defaults():
    config.save_config({"backend": "audio.cpp", "url": "http://old", "server": {"clear_device_visibility": True},
                        "client": {"language": "Chinese"}, "live": {"fa": True, "source": "old", "overlay": True}})
    result = CliRunner().invoke(cli.main, ["config", "init"], input="y\nRemote address\nllama.cpp\nhttp://new\ny\n")
    assert result.exit_code == 0, result.output
    assert config.load_config() == {"connection": "remote", "backend": "llama.cpp", "url": "http://new"}


def test_runtime_ignores_old_defaults_and_accepts_overrides(monkeypatch):
    config.save_config({"client": {"language": "Chinese"}, "live": {"source": "old", "overlay": True}})
    run = Mock()
    monkeypatch.setattr(live, "run", run)
    runner = CliRunner()
    assert runner.invoke(cli.main, ["live"]).exit_code == 0
    args = run.call_args.args[0]
    assert args.language is None
    assert args.source == "@DEFAULT_MONITOR@"
    assert args.overlay is False
    assert runner.invoke(cli.main, ["live", "--language", "English", "--source", "mic", "--overlay"]).exit_code == 0
    args = run.call_args.args[0]
    assert (args.language, args.source, args.overlay) == ("English", "mic", True)


def test_asr_returns_detected_language_for_alignment():
    client = AudioCppASR("http://audio", model="qwen3-asr")
    client.detect_language = True
    client.session = Mock()
    client.session.post.return_value.json.return_value = {"text": "hello", "language": "English"}
    assert client.transcribe(b"\0\0" * 512) == "hello"
    assert client.detected_language == "English"
    assert client.session.post.call_args.args[0] == "http://audio/v1/audio/transcriptions/details"
    assert "language" not in client.session.post.call_args.kwargs["data"]


def test_worker_passes_detected_language_to_aligner():
    client = SimpleNamespace(language=None, detected_language="Chinese", close=Mock())
    aligner = Mock()
    aligner.cut.return_value = 60
    worker = TranscriptionWorker(client, aligner=aligner)
    try:
        worker.request_cut(b"aa" * 512 * 170, "第一句。第二句继续", 0, (0, "第一句。第二句"))
        assert aligner.language == "Chinese"
        aligner.cut.assert_called_once()
    finally:
        worker.close()


def test_llama_devices_from_stderr(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", Mock(return_value=SimpleNamespace(
        stdout="", stderr="Available devices:\n  ROCm0: GPU one\n  ROCm1: GPU two\n")))
    assert list(cli.available_devices("llama-server", "llama.cpp").values()) == ["ROCm0", "ROCm1", "none"]
