from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from wayasr import cli
from wayasr.client import AudioCppASR, LlamaASR, create_client
from wayasr.config import config_path, load_config, save_config


@pytest.mark.parametrize("backend,kind", [("llama.cpp", LlamaASR), ("audio.cpp", AudioCppASR)])
def test_client_factory(backend, kind):
    client = create_client(backend, "http://server")
    try:
        assert isinstance(client, kind)
        client.detect_language = True
        client.session = Mock()
        client.session.post.return_value.json.return_value = {"text": "hello"}
        assert client.transcribe(b"aa") == "hello"
        url = client.session.post.call_args.args[0]
        data = client.session.post.call_args.kwargs["data"]
        if backend == "audio.cpp":
            assert url.endswith("/transcriptions/details")
            assert data == {"model": "qwen3-asr"}
        else:
            assert url.endswith("/transcriptions")
            assert data == {"response_format": "json", "temperature": "0", "max_tokens": "512"}
    finally:
        client.close()


@pytest.mark.parametrize("command", [["serve"], ["service", "install"]])
def test_remote_configuration_rejects_deployment(command):
    save_config({"connection": "remote", "backend": "audio.cpp", "url": "http://server"})
    result = CliRunner().invoke(cli.main, command)
    assert result.exit_code == 1
    assert "Remote connection configured" in result.output


def test_legacy_configuration_is_not_rewritten():
    save_config({"server": {"asr_model": "legacy"}, "backend": "audio.cpp"})
    before = config_path().read_bytes()
    config = load_config()
    assert config["connection"] == "local"
    assert config["client"]["asr_model"] == "legacy"
    assert config_path().read_bytes() == before


def test_remote_alignment_and_runtime_overrides(monkeypatch):
    import wayasr.live

    result = CliRunner().invoke(cli.main, ["config", "init"], input=(
        "Remote address\naudio.cpp\nhttp://server\nrecognizer\ny\naligner\ny\n"))
    assert result.exit_code == 0, result.output
    run = Mock()
    monkeypatch.setattr(wayasr.live, "run", run)
    result = CliRunner().invoke(cli.main, ["live", "--url", "http://other", "--asr-model", "override"])
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.asr_model == "override"
    assert args.aligner_model == "aligner"
    assert args.aligner_url == args.url == "http://other"
    result = CliRunner().invoke(cli.main, ["live", "--backend", "llama.cpp"])
    assert result.exit_code == 0, result.output
    assert not hasattr(run.call_args.args[0], "aligner_url")
