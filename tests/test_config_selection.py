from unittest.mock import Mock
from types import SimpleNamespace

import beaupy
from click.testing import CliRunner
import pytest

from wayasr import cli, config


@pytest.mark.parametrize("backend", ["llama.cpp", "audio.cpp"])
def test_terminal_backend_selection(monkeypatch, backend):
    config.save_config({"backend": backend, "url": "http://localhost:9000"})
    select = Mock(side_effect=lambda options, **kwargs: backend if kwargs["title"] == "Service" else options[0])
    monkeypatch.setattr(cli, "available_devices", lambda *args: {"CPU": "none" if backend == "llama.cpp" else ("cpu", 0)})
    monkeypatch.setattr(beaupy, "select", select)
    monkeypatch.setattr(beaupy, "confirm", Mock(side_effect=[True] + ([False] if backend == "audio.cpp" else []) + [True]))
    runner = CliRunner()
    with runner.isolation(input="\n\n\n"):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        cli.main.main(args=["config", "init"], standalone_mode=False)
    assert select.call_args_list[0].args == (["llama.cpp", "audio.cpp", "Custom URL"],)
    assert select.call_args_list[0].kwargs == {"title": "Service", "cursor_index": ["llama.cpp", "audio.cpp"].index(backend)}
    assert config.load_config()["backend"] == backend


@pytest.mark.parametrize("answer", [True, False, None])
def test_terminal_confirmation(monkeypatch, answer):
    prompt = Mock(return_value=answer)
    monkeypatch.setattr(beaupy, "confirm", prompt)
    with CliRunner().isolation():
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        if answer is None:
            with pytest.raises(cli.click.Abort):
                cli.confirm("Save?", default=True)
        else:
            assert cli.confirm("Save?", default=True) is answer
    prompt.assert_called_once_with("Save?", default_is_yes=True)


def test_audio_backend_discovery(monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout='available_devices=3\nCPU:0 "CPU" [CPU]\nCUDA:0 "GPU" [GPU]\nCUDA:1 "GPU" [GPU]\n', stderr=""))
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert list(cli.available_devices("/custom/audiocpp_server", "audio.cpp").values()) == [("cpu", 0), ("cuda", 0), ("cuda", 1)]
    run.assert_called_once_with(
        ["/custom/audiocpp_server", "--list-devices"],
        capture_output=True, text=True, check=True, timeout=10,
    )


def test_audio_backend_discovery_empty(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="", stderr="")))
    with pytest.raises(ValueError, match="No devices found"):
        cli.available_devices("audiocpp_server", "audio.cpp")


def test_terminal_backend_cancel_does_not_save(monkeypatch):
    monkeypatch.setattr(beaupy, "select", Mock(return_value=None))
    runner = CliRunner()
    with runner.isolation():
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        with pytest.raises(cli.click.Abort):
            cli.main.main(args=["config", "init"], standalone_mode=False)
    assert not config.config_path().exists()
