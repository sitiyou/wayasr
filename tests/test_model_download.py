import json
from types import SimpleNamespace
from unittest.mock import Mock, call

from click.testing import CliRunner

import huggingface_hub
import pytest

from wayasr import cli
from wayasr.config import config_path, load_config, save_config


@pytest.mark.parametrize("fa", [False, True])
def test_default_audio_models_use_hub_cache(monkeypatch, tmp_path, fa):
    model = tmp_path / "cached.gguf"
    model.touch()
    download = Mock(return_value=str(model))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/audiocpp_server")
    save_config({"backend": "audio.cpp", "live": {"fa": fa}})
    args = SimpleNamespace(server_bin="audiocpp_server", host="127.0.0.1", port=8087)
    cli.server_command(args)
    filenames = ["Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf"]
    if fa:
        filenames.append("Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf")
    assert download.call_args_list == [
        call(repo_id="audio-cpp/audio.cpp-gguf", filename=filename)
        for filename in filenames
    ]
    generated = json.loads(config_path().with_name("audio-server.json").read_text())
    assert [entry["path"] for entry in generated["models"]] == [str(model)] * len(filenames)
    assert "server" not in load_config()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        cli.resolve_audio_model("owner/repo")


@pytest.mark.parametrize("aligner", [False, True])
def test_explicit_preset(monkeypatch, aligner):
    download = Mock(return_value="/cache/model.gguf")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    preset = cli.AUDIO_ALIGNER_MODEL if aligner else cli.AUDIO_ASR_MODEL
    assert cli.resolve_audio_model(preset, aligner=aligner) == "/cache/model.gguf"
    assert download.call_count == 1


def test_local_model_selection(monkeypatch, tmp_path):
    model = tmp_path / "custom.gguf"
    model.touch()
    select = Mock(return_value="Local file")
    monkeypatch.setattr(cli, "select_option", select)
    with CliRunner().isolation(input=f"relative.gguf\n{model}\n"):
        assert cli.select_audio_model("ASR model", str(model), cli.AUDIO_ASR_MODEL) == str(model)
    select.assert_called_once_with("ASR model", [cli.AUDIO_ASR_MODEL, "Local file"], "Local file")


def test_preset_model_selection(monkeypatch):
    monkeypatch.setattr(cli, "select_option", Mock(return_value=cli.AUDIO_ASR_MODEL))
    assert cli.select_audio_model("ASR model", None, cli.AUDIO_ASR_MODEL) == cli.AUDIO_ASR_MODEL


def test_custom_audio_model_skips_download(monkeypatch, tmp_path):
    model = tmp_path / "custom.gguf"
    model.touch()
    download = Mock(side_effect=AssertionError("Must use the local model"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/audiocpp_server")
    save_config({"backend": "audio.cpp", "server": {"asr_path": str(model)}, "live": {"fa": False}})
    cli.server_command(SimpleNamespace(server_bin="audiocpp_server", host="127.0.0.1", port=8087))
    download.assert_not_called()
