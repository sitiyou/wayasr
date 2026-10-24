import sys
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from wayasr import cli, config, live, overlay


@pytest.fixture
def controller(monkeypatch):
    controller = Mock()
    factory = Mock(return_value=controller)
    monkeypatch.setitem(sys.modules, "asslayer", SimpleNamespace(ApplicationController=factory))
    monkeypatch.setattr(overlay, "sys", SimpleNamespace(platform="linux", version_info=(3, 14)))
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    monkeypatch.setattr(overlay, "time", SimpleNamespace(monotonic=lambda: 12.345))
    return controller, factory


def test_overlay_controller_lifecycle(controller):
    ctrl, factory = controller
    with overlay.WaylandOverlay(font_size=48, duration=60):
        factory.assert_called_once_with(namespace="wayasr")
        ctrl.start.assert_called_once()
        assert "Style: Default,sans-serif,48," in ctrl.set_ass_input.call_args.args[0]
        ctrl.set_status.assert_called_once_with(paused=False, hidden=False, locked=True,
                                               target_fps=30, start_time_ms=12345)
    ctrl.stop.assert_called_once()
    ctrl.join.assert_called_once()


def test_overlay_wraps_within_half_the_reference_width():
    header = overlay.ASS_HEADER.format(font_size=42)
    assert "PlayResX: 1920" in header
    assert "WrapStyle: 2" in header
    assert ",2,480,480,64,1\n" in header


@pytest.mark.parametrize("character", ["中", "W"])
def test_ass_text_inserts_hard_breaks_without_whitespace(character):
    assert overlay.ass_text(character * 45) == character * 22 + r"\N" + character * 22 + r"\N" + character


def test_ass_text_counts_stable_and_pending_as_one_line():
    text = overlay.ass_text("中" * 20, "文" * 5)
    assert text == "中" * 20 + r"{\c&HAAAAAA&}" + "文" * 2 + r"\N" + "文" * 3


def test_ass_text_keeps_combining_marks_and_existing_line_breaks():
    assert overlay.ass_text("e\u0301" * 23) == "e\u0301" * 22 + r"\N" + "e\u0301"
    assert overlay.ass_text("中" * 22 + "\r\n" + "文" * 22) == "中" * 22 + r"\N" + "文" * 22


def test_font_size_controls_hard_breaks_in_all_caption_paths(controller):
    ctrl, _ = controller
    view = overlay.WaylandOverlay(font_size=48)
    view.partial("中" * 21)
    assert "中" * 20 + r"\N中" in ctrl.set_ass_input.call_args.args[0]
    view.sentence("文" * 21)
    assert "文" * 20 + r"\N文" in ctrl.set_ass_input.call_args.args[0]
    view.partial("")
    assert "文" * 20 + r"\N文" in ctrl.set_ass_input.call_args.args[0]


def test_overlay_stable_pending_final_and_expiry(controller):
    ctrl, _ = controller
    view = overlay.WaylandOverlay(duration=8)
    view.partial("稳定", "待确认")
    content = ctrl.set_ass_input.call_args.args[0]
    assert r"稳定{\c&HAAAAAA&}待确认" in content
    assert "0:00:08.00" in content
    ctrl.set_status.assert_called_with(start_time_ms=12345)
    view.sentence("最终结果。")
    view.partial("")
    content = ctrl.set_ass_input.call_args.args[0]
    assert "最终结果。" in content
    assert "待确认" not in content
    assert r"{\c&HAAAAAA&}" not in content


def test_overlay_text_cannot_inject_ass_events_or_tags(controller):
    ctrl, _ = controller
    view = overlay.WaylandOverlay()
    view.sentence("{\\pos(0,0)}\\N\r\nDialogue: fake\x00")
    content = ctrl.set_ass_input.call_args.args[0]
    assert "｛＼pos(0,0)｝＼N" in content
    assert r"\NDialogue: fake" in content
    assert "\x00" not in content
    assert len([line for line in content.splitlines() if line.startswith("Dialogue:")]) == 1


def test_overlay_start_failure_closes_controller(controller):
    ctrl, _ = controller
    ctrl.start.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        with overlay.WaylandOverlay():
            pytest.fail("Controller must not start")
    ctrl.stop.assert_called_once()
    ctrl.join.assert_called_once()


@pytest.mark.parametrize("platform,version", [("linux", (3, 12)), ("darwin", (3, 14))])
def test_overlay_rejects_unsupported_runtime(controller, monkeypatch, platform, version):
    _, factory = controller
    monkeypatch.setattr(overlay, "sys", SimpleNamespace(platform=platform, version_info=version))
    with pytest.raises(ValueError, match="Linux and Python 3.14"):
        overlay.WaylandOverlay()
    factory.assert_not_called()


def test_overlay_requires_wayland_session(controller, monkeypatch):
    _, factory = controller
    monkeypatch.delenv("WAYLAND_DISPLAY")
    with pytest.raises(ValueError, match="WAYLAND_DISPLAY"):
        overlay.WaylandOverlay()
    factory.assert_not_called()


def test_missing_asslayer_is_actionable(controller, monkeypatch):
    monkeypatch.setitem(sys.modules, "asslayer", None)
    with pytest.raises(ValueError, match="live,wayland"):
        overlay.WaylandOverlay()


def test_subtitles_are_sent_to_both_views():
    tui, desktop = Mock(), Mock()
    output = live.SubtitleOutput(tui, desktop)
    output.update("你好世界")
    output.update("你好朋友")
    output.update("最终结果。", final=True)
    assert tui.mock_calls == desktop.mock_calls
    desktop.partial.assert_any_call("你好", "朋友")
    desktop.sentence.assert_called_once_with("最终结果。")
    desktop.partial.assert_called_with("")


def test_worker_sends_final_to_overlay_and_keeps_terminal_output(capsys):
    client, desktop = Mock(), Mock()
    client.transcribe.return_value = "最终结果。"
    worker = live.TranscriptionWorker(client, overlay=desktop)
    worker.submit(b"\x00\x00" * 512, 0, final=True)
    worker.close()
    desktop.sentence.assert_called_once_with("最终结果。")
    assert "最终结果。" in capsys.readouterr().out
    client.close.assert_called_once()


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_capture_failure_closes_overlay(controller, monkeypatch, error):
    ctrl, _ = controller
    monkeypatch.setattr(live.shutil, "which", lambda name: "/usr/bin/parec")
    capture = Mock(side_effect=error("capture failed"))
    monkeypatch.setattr(live, "capture", capture)
    args = SimpleNamespace(partial_interval=1, max_segment=15, overlay=True,
                           overlay_font_size=42, overlay_duration=8)
    with pytest.raises(error, match="capture failed"):
        live.run(args)
    capture.assert_called_once()
    ctrl.stop.assert_called_once()
    ctrl.join.assert_called_once()


def test_terminal_capture_does_not_construct_overlay(monkeypatch):
    monkeypatch.setattr(live.shutil, "which", lambda name: "/usr/bin/parec")
    factory = Mock(side_effect=AssertionError("Overlay must remain optional"))
    monkeypatch.setattr(overlay, "WaylandOverlay", factory)
    capture = Mock()
    monkeypatch.setattr(live, "capture", capture)
    args = SimpleNamespace(partial_interval=1, max_segment=15, overlay=False)
    live.run(args)
    capture.assert_called_once_with(args)
    factory.assert_not_called()


def test_overlay_cli_defaults_and_overrides(monkeypatch):
    config.save_config({"live": {"overlay": True, "overlay_font_size": 48, "overlay_duration": 10}})
    run = Mock()
    monkeypatch.setattr(live, "run", run)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["live"])
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert (args.overlay, args.overlay_font_size, args.overlay_duration) == (True, 48, 10)
    result = runner.invoke(cli.main, ["live", "--no-overlay"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].overlay is False
    result = runner.invoke(cli.main, ["live", "--overlay-font-size", "0"])
    assert result.exit_code == 2
