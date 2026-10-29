# WayASR

Local speech-to-text with live Linux captions, file transcription, and optional Wayland desktop subtitles. Inference runs in a separate **llama.cpp** or **audio.cpp** server, locally or on another machine.

## Install

Requires Python 3.10+. Python 3.12 is a practical choice for live captions.

```sh
uv tool install 'wayasr[live]'
uv tool update-shell
```

For file transcription and server management only, install `wayasr` without extras. Inference servers and their models are installed separately.

Live capture requires Linux with PulseAudio or PipeWire-Pulse and `parec`:

```sh
# Debian / Ubuntu
sudo apt install pulseaudio-utils

# Arch Linux
sudo pacman -S libpulse
```

## Connect a server

```sh
wayasr config init
```

Choose **Local service** or **Remote address**, then a backend:

| Backend | Local requirements | Model setup |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | A recent `llama-server` with Qwen3-ASR support | Download the ASR model and matching projector yourself |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `audiocpp_server` with Qwen3-ASR and, optionally, Qwen3-ForcedAligner support | Select presets for automatic download at server startup, or choose existing local files |

For llama.cpp, download `Qwen3-ASR-1.7B-Q8_0.gguf` and `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` from [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) into the model directory selected during setup. For audio.cpp, use its native model format; do not assume llama.cpp models are interchangeable.

Initialization saves settings but does not download models or start a server. For a local connection, start it afterward:

```sh
wayasr serve
```

Keep that terminal running. Alternatively, install a Linux systemd user service:

```sh
wayasr service install
systemctl --user status wayasr.service
journalctl --user -u wayasr.service -f
```

`service install` enables and starts the service; running it again updates and restarts it. `wayasr service uninstall` removes the service but keeps models and application settings.

For a remote connection, enter the server's **root URL**, such as `http://localhost:8087`, not an endpoint path. audio.cpp also requires the server's ASR model ID and, if enabled, alignment model ID. There is no API-key option; use a trusted network, SSH tunnel, or authenticated proxy rather than exposing an unprotected server publicly.

## Live captions

Capture system output by default, or select a microphone:

```sh
wayasr live
wayasr live --language Chinese
wayasr live --source @DEFAULT_SOURCE@
pactl list short sources
```

Language is detected automatically unless specified. The terminal shows completed sentences and a revisable live transcript. Normal-colored live text is based on agreement between successive results; gray text is new or changed, **not a confidence score**. Text retained across an aligned cut stays visible in normal color and is compared with the next result.

Useful options:

| Option | Purpose |
| --- | --- |
| `--plain` | Disable the Rich interface; no inference timing lines |
| `--partial-interval 2` | Request intermediate recognition every 2 seconds; default: 1 |
| `--partial-interval 0` | Recognize completed audio segments only; incompatible with forced alignment |
| `--max-segment 8` | Limit the audio buffer to about 8 seconds; default: 10, allowed range: 1–25 |
| `--url URL` | Override the configured server address, not the backend |

Press **Ctrl+C** to stop capture and finish pending work. Shutdown can take time while an inference or alignment request completes. Redirecting output disables the Rich interface, but plain mode still uses terminal line-update sequences; it is not an SRT or JSON export.

### Wayland desktop subtitles

Requires Linux, Python 3.14+, Vulkan, and a compositor supporting `wlr-layer-shell`:

```sh
uv tool install --force --python 3.14 'wayasr[live,wayland]'
wayasr live --overlay
wayasr live --overlay --plain --overlay-font-size 48 --overlay-duration 10
```

The optional `asslayer` integration displays bottom-centered live captions. With `--overlay`, the terminal prints only completed subtitle lines: no Rich interface or dynamic text updates. Desktop subtitles are off by default; unchanged captions expire after 8 seconds. Run inside an active Wayland session.

## Transcribe a file

```sh
wayasr transcribe recording.wav
wayasr transcribe recording.mp3 --language English --timeout 180
```

Audio is converted to mono 16 kHz PCM without changing the original file. Format support depends on libsndfile; AAC/M4A and video containers are not guaranteed. File mode sends the entire recording in one request, so split long recordings to fit your server's limits.

## Settings and diagnostics

```sh
wayasr config show
wayasr live --help
wayasr serve --help
```

Precedence: **command-line options → supported environment variables → saved settings → built-in defaults**. `WAYASR_URL` overrides the server address; `WAYASR_MODEL_DIR` overrides the local model directory. Language, capture source, and desktop overlay are selected at runtime rather than saved by initialization.

Default Linux paths, honoring the corresponding XDG environment variables:

| Content | Location |
| --- | --- |
| Configuration | `~/.config/wayasr/config.json` |
| Models | `~/.local/share/wayasr/models/` |
| Alignment diagnostics | `~/.local/state/wayasr/alignment.log` |

Alignment failures are written only to rotating log files, not the terminal. Logs may contain recognized words. Local servers default to unloading idle models after 180 seconds; the next request may need to reload them.

## Limitations

- Live capture is Linux-only. File transcription does not require Linux audio capture tools.
- Captions are revisable recognition results, not guaranteed verbatim transcripts. Short audio overlaps reduce cut-off words but can still produce repeated text.
- Optional forced alignment is available with audio.cpp and can be selected during setup. It locates recognized text in audio; it does not correct recognition errors. Slow or failed alignment does not block subsequent ASR work; silence and maximum-length cuts remain available.
- If recognition cannot keep up, use a faster model or device, increase the partial interval, or disable intermediate recognition (and forced alignment).

For all commands and options, run `wayasr --help` or a subcommand's `--help`.
