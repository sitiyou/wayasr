from pathlib import Path
import subprocess
from unittest.mock import Mock

from click.testing import CliRunner

from wayasr import cli, config, service


def setup_service(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(service.sys, "platform", "linux")
    monkeypatch.setattr(service.shutil, "which", lambda binary: "/usr/bin/" + binary)
    run = Mock()
    monkeypatch.setattr(service.subprocess, "run", run)
    models = tmp_path / "models with spaces % and $"
    models.mkdir()
    for name in (cli.MODEL_NAME, cli.MMPROJ_NAME):
        (models / name).touch()
    unit = tmp_path / "config/systemd/user/wayasr.service"
    return run, models, unit


def test_install_generates_absolute_paths_and_restarts(tmp_path, monkeypatch):
    run, models, unit = setup_service(tmp_path, monkeypatch)
    result = CliRunner().invoke(cli.main, ["service", "install", "--model-dir", str(models),
                                          "--device", "ROCm0"])
    assert result.exit_code == 0, result.output
    content = unit.read_text()
    assert 'ExecStart="/usr/bin/llama-server"' in content
    assert str(models).replace("%", "%%").replace("$", "$$") in content
    assert '"--device" "ROCm0"' in content
    assert "UnsetEnvironment" not in content
    assert "WorkingDirectory" not in content
    calls = [call.args[0] for call in run.call_args_list]
    assert calls == [["systemctl", "--user", "show-environment"],
                     ["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "--user", "enable", "wayasr.service"],
                     ["systemctl", "--user", "restart", "wayasr.service"]]
    assert not list(unit.parent.glob("tmp*"))
    saved = config.load_config()
    assert saved["url"] == "http://127.0.0.1:8087"
    assert saved["server"]["model_dir"] == str(models)
    assert saved["server"]["device"] == "ROCm0"


def test_install_replaces_symlink_without_modifying_target(tmp_path, monkeypatch):
    _, models, unit = setup_service(tmp_path, monkeypatch)
    unit.parent.mkdir(parents=True)
    original = tmp_path / "original.service"
    original.write_text("original")
    unit.symlink_to(original)
    result = CliRunner().invoke(cli.main, ["service", "install", "--model-dir", str(models)])
    assert result.exit_code == 0, result.output
    assert original.read_text() == "original"
    assert not unit.is_symlink()
    assert "UnsetEnvironment" not in unit.read_text()


def test_uninstall_keeps_models_and_dropins(tmp_path, monkeypatch):
    run, models, unit = setup_service(tmp_path, monkeypatch)
    unit.parent.mkdir(parents=True)
    unit.write_text("unit")
    dropin = unit.with_name(unit.name + ".d")
    dropin.mkdir()
    (dropin / "custom.conf").write_text("custom")
    result = CliRunner().invoke(cli.main, ["service", "uninstall"])
    assert result.exit_code == 0, result.output
    assert not unit.exists()
    assert (dropin / "custom.conf").read_text() == "custom"
    assert (models / cli.MODEL_NAME).exists()
    assert run.call_args_list[1].args[0] == ["systemctl", "--user", "disable", "--now", "wayasr.service"]


def test_unavailable_user_manager_does_not_write_unit(tmp_path, monkeypatch):
    run, models, unit = setup_service(tmp_path, monkeypatch)
    run.side_effect = subprocess.CalledProcessError(1, ["systemctl", "--user"])
    result = CliRunner().invoke(cli.main, ["service", "install", "--model-dir", str(models)])
    assert result.exit_code == 1
    assert not unit.exists()


def test_service_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "darwin")
    result = CliRunner().invoke(cli.main, ["service", "install"])
    assert result.exit_code == 1
    assert "requires Linux" in result.output


def test_failed_restart_does_not_replace_saved_defaults(tmp_path, monkeypatch):
    run, models, _ = setup_service(tmp_path, monkeypatch)
    config.save_config({"url": "http://localhost:8087", "server": {"port": 8087}})
    before = config.config_path().read_bytes()

    def systemctl(command, **kwargs):
        if "restart" in command:
            raise subprocess.CalledProcessError(1, command)

    run.side_effect = systemctl
    result = CliRunner().invoke(cli.main, ["service", "install", "--model-dir", str(models), "--port", "9000"])
    assert result.exit_code == 1
    assert config.config_path().read_bytes() == before


def test_install_reuses_port_and_normalizes_ipv6(tmp_path, monkeypatch):
    run, models, _ = setup_service(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["service", "install", "--model-dir", str(models), "--host", "::", "--port", "9000"])
    assert result.exit_code == 0, result.output
    assert config.load_config()["url"] == "http://[::1]:9000"
    run.reset_mock()
    result = runner.invoke(cli.main, ["service", "install"])
    assert result.exit_code == 0, result.output
    assert config.load_config()["server"]["port"] == 9000


def test_unit_argument_escaping():
    assert service.quote_argument('a "b"\\c%$') == '"a \\"b\\"\\\\c%%$$"'
    for character in ("\n", "\r", "\x00"):
        try:
            service.quote_argument("injected" + character)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe argument accepted")
