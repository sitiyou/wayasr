# WayASR

使用 llama.cpp 在本地转写音频，提供优先按停顿分段的实时字幕，以及简洁的 Rich Live 终端界面。默认模型为 Qwen3-ASR-1.7B Q8_0。

- 文件转写：连接本地或远程 `/v1/audio/transcriptions` 服务。
- 实时字幕：采集 Linux 系统声音或麦克风，保留中间识别结果的可修正末尾。
- 本地服务：启动独立安装的 `llama-server`，可选择 CPU、CUDA、ROCm、Vulkan 等设备；实际可用后端取决于 llama.cpp 的构建。

## 1. 使用 uv 安装

需要 Python 3.10+；新环境建议 Python 3.12，并确认 PyTorch/ONNX Runtime 有对应平台的安装包。先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。PyPI 上的发行版可直接安装：

```sh
uv tool install --torch-backend cpu 'wayasr[live]'
uv tool update-shell
wayasr --help
```

`uv tool install` 将命令安装到独立环境，不需要手动创建或激活虚拟环境。如果 shell 尚未包含 uv 的工具目录，请按照 `uv tool update-shell` 提示重新打开终端。需要指定 Python 时可加 `--python 3.12`。

如果只需要文件转写、启动服务或管理 systemd，安装基础版本即可，不需要 PyTorch、VAD 或 Rich：

```sh
uv tool install wayasr
```

也可以从源码安装：

```sh
git clone https://github.com/sitiyou/wayasr.git
cd wayasr
uv tool install --torch-backend cpu '.[live]'
```

从源码安装后修改了代码，可在项目目录重新运行安装命令并加 `--force` 更新；卸载使用 `uv tool uninstall wayasr`。工具卸载不会自动卸载 systemd 服务，需要先执行 `wayasr service uninstall`。

实时采集使用 CPU PyTorch 运行 VAD，而 ASR 推理由独立的 llama.cpp 服务处理，无需为 VAD 安装 CUDA/ROCm 版 PyTorch。开发环境的 uv 配置同样优先使用 CPU PyTorch。

## 2. 安装 llama.cpp 与下载模型

已有兼容的转写 API 时可以跳过本节和本地服务启动步骤，通过 `wayasr config init` 接入，或直接指定 `--url`。

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 获取符合平台和 GPU 后端的版本，或按[官方构建说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)编译。需要较新、支持 Qwen3-ASR 音频转写及 `--mmproj-device` 的版本。

确认命令可用：

```sh
llama-server --help
llama-server --list-devices
```

从 [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/tree/main) 下载以下两个文件到项目的 `models/` 目录：

- `Qwen3-ASR-1.7B-Q8_0.gguf`
- `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`

也可使用 Hugging Face CLI：

```sh
uvx --from huggingface-hub hf download ggml-org/Qwen3-ASR-1.7B-GGUF \
  Qwen3-ASR-1.7B-Q8_0.gguf mmproj-Qwen3-ASR-1.7B-Q8_0.gguf \
  --local-dir ./models
```

模型不随 Python 包分发。其他量化版本可通过 `--model`、`--mmproj` 指定，模型和 projector 必须匹配。

## 3. 启动转写服务

```sh
wayasr serve --model-dir ./models
```

默认由 llama.cpp 选择设备，监听 `127.0.0.1:8087`。在另一个终端检查：

```sh
curl --fail http://127.0.0.1:8087/health
```

按需选择设备，名称以 `llama-server --list-devices` 为准：

```sh
wayasr serve --model-dir ./models --device ROCm0
wayasr serve --model-dir ./models --device CUDA0
wayasr serve --model-dir ./models --device Vulkan0
wayasr serve --model-dir ./models --device none --flash-attn off
```

默认尊重当前 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES` 等限制。如果明确需要忽略这些限制，添加 `--clear-device-visibility`；仅对服务子进程生效，不修改当前 shell。清除限制后设备编号可能改变，请重新确认。

其他常用选项：

```sh
wayasr serve --server-bin /path/to/llama-server \
  --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 --port 8087 --gpu-layers 99 --flash-attn auto
```

不要把没有认证保护的服务直接暴露到公网。跨机器访问建议使用 SSH 隧道或经过认证的反向代理。

## 4. 文件转写

输入须为 **16 kHz、单声道、16-bit PCM WAV**。可以先用 FFmpeg 转换：

```sh
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
wayasr transcribe audio.wav --language Chinese
```

省略 `--language` 自动识别语言。连接其他地址或增加超时：

```sh
wayasr transcribe audio.wav --url http://127.0.0.1:8087 --timeout 120
```

当前文件模式将整个 WAV 一次提交；长文件应先分段，支持的最大时长取决于服务和模型。

## 5. 实时字幕（Linux）

需要 PulseAudio 或 PipeWire-Pulse，以及 `parec`。例如：

```sh
# Debian / Ubuntu
sudo apt install pulseaudio-utils

# Arch Linux
sudo pacman -S libpulse
```

默认采集系统输出声音：

```sh
wayasr live --language Chinese
```

采集麦克风或指定设备：

```sh
wayasr live --source @DEFAULT_SOURCE@ --language Chinese
pactl list short sources
wayasr live --source YOUR_SOURCE_NAME
```

界面显示最近的已完成字幕、可修正的实时字幕、语音/静音状态、缓冲时长，以及最近一次最终转写的耗时和 RTF。历史最多保留 100 条，窗口显示其中最近若干条，不提供滚动操作。按 **Ctrl+C** 后等待已排队的最终转写完成，退出后保留界面。

```sh
wayasr live --plain
wayasr live --partial-interval 0
wayasr live --max-segment 10
```

`--plain` 禁用 TUI，`--partial-interval 0` 只识别最终结果，`--max-segment` 设置最大音频缓冲时长（范围 1–25 秒）。

输出重定向时自动禁用 TUI；文本模式保留原有终端行更新控制序列，并非 SRT/JSON 导出。macOS、Windows 可以使用文件转写或连接远程服务，但尚无内置实时采集适配。

### Wayland 桌面字幕

需要 Linux、Python 3.14+、支持 `wlr-layer-shell` 的 Wayland 合成器，以及可用的 Vulkan 驱动。此功能通过 PyPI 的 `asslayer` 提供，不依赖本地源码路径；仅安装基础 CLI 或 `[live]` 时不会引入它。

从项目目录安装桌面显示依赖：

```sh
uv tool install --force --python 3.14 --torch-backend cpu '.[live,wayland]'
```

在已登录的 Wayland 图形会话中运行：

```sh
wayasr live --overlay --language Chinese
wayasr live --overlay --plain --source @DEFAULT_SOURCE@ --language Chinese
wayasr live --overlay --overlay-font-size 48 --overlay-duration 10
```

字幕显示在屏幕底部中央，文本区域约为屏幕宽度的 50%，生成 ASS 时主动插入强制换行，不依赖引擎自动断行。默认字号下每行最多 22 个非组合字符，字号增大时字符上限相应减少；稳定文本和待确认文本共用同一行的计数。稳定文本为白色，待确认文本为灰色；下一条识别结果替换当前字幕。最后一句默认保留 8 秒，无更新后消失。`--overlay-font-size` 设置 ASS 字号（12–120），`--overlay-duration` 设置无更新时的保留秒数（1–60）。识别文本中的反斜杠和花括号显示为全角字符，避免被解释为 ASS 控制指令。

桌面字幕与终端输出同时保留；`--plain` 只禁用 Rich TUI，不禁用桌面字幕。Ctrl+C 结束采集、处理待完成转写，并关闭叠加层。该基础集成不提供托盘、菜单或拖拽操作，默认鼠标穿透；合成器规则可按 layer-shell namespace `wayasr` 匹配，与 layrics 的显示层独立。

`config init` 可以保存是否启用桌面字幕；`--no-overlay` 临时覆盖已保存的启用状态。也可以在配置文件的 `live` 对象中设置 `overlay`、`overlay_font_size` 和 `overlay_duration`。桌面显示并未扩展采集平台范围，音频仍通过 PulseAudio/PipeWire-Pulse 的 `parec` 获取。

如果没有显示，请确认服务已就绪、音源有声音、`WAYLAND_DISPLAY` 有效、合成器支持 layer-shell，以及 Vulkan 驱动可用；查看终端中的 asslayer 日志。

### 分段与识别限制

- 连续约 480 ms 静音结束一段；默认约 15 秒为硬上限。累计约 5 秒后，遇到约 320 ms 的停顿可以提前提交，减少后续重复转写的音频长度。
- 累计约 5 秒后，约 128 ms 的短停顿会额外触发一次中间识别；同一段连续两次识别文本完全一致，且以句末标点结束时，允许在该次识别覆盖的停顿处提前切分。异步识别返回后，新采集的音频仍保留在下一段；不按文字比例猜测音频位置。`--partial-interval 0` 禁用中间识别及标点辅助切分，VAD 提前切分仍生效。
- 到达硬上限时，优先在最近约 3 秒寻找至少约 128 ms 的停顿，切点之后的音频继续保留。
- 没有停顿才硬切，并保留约 512 ms 重叠；较小的最大段长会相应限制重叠长度。只对带重叠的相邻块尝试短文本精确去重，不能保证消除全部重复或区分全部真实重复。
- 连续两次结果一致的前缀正常显示，未稳定末尾暗色显示，仍允许后续修正。未完句跨块接续，按常见中英文句末标点换行。
- 没有词时间戳或强制对齐，无法保证切点严格对应完整句子；英文缩写等复杂断句也可能不准确。
- 最终结果积压超过 16 段会明确报错，不静默丢弃。可关闭中间识别、使用更快的模型或 GPU 来减轻积压。

## 初始化与持久化配置

```sh
wayasr config init
wayasr config show
```

`config init` 首先询问是否配置本地 llama.cpp 服务。选择否即可接入已有 API，只需设置客户端 URL、识别语言、采集源和是否启用 Wayland 桌面字幕，无需安装 llama.cpp 或下载模型；已有本地服务配置会保留。选择是时，还会询问 `llama-server` 路径、模型目录、设备、GPU 可见性限制、监听地址和端口，最后确认保存。已有配置作为初始值；取消不会覆盖文件。初始化仅保存配置，不下载模型、不安装或重启服务。可以将模型目录设置为前面下载文件使用的 `./models`，保存时转换为绝对路径。

已有 API 的 URL 应填写服务根地址（例如 `https://asr.example.com`），不要包含 `/v1/audio/transcriptions`；客户端会自动追加该路径。服务须兼容当前客户端的 WAV multipart 上传及 JSON `text` 响应；目前不提供 API Key 或自定义模型参数配置。

目录由 `platformdirs` 按当前平台管理。Linux 默认使用：

- 配置：`~/.config/wayasr/config.json`，尊重 `XDG_CONFIG_HOME`。
- 未配置时的模型目录：`~/.local/share/wayasr/models`，尊重 `XDG_DATA_HOME`。

macOS、Windows 使用平台标准用户目录；`config show` 显示实际配置文件位置和已保存值。

默认值优先级为：**命令行参数 → 对应环境变量 → 配置文件 → 内置默认值**。临时命令行或环境变量覆盖不会修改文件。

| 环境变量 | 用途 | 未配置时的默认值 |
| --- | --- | --- |
| `WAYASR_URL` | 文件及实时转写的服务地址 | `http://127.0.0.1:8087` |
| `WAYASR_MODEL_DIR` | 服务模型目录 | 平台用户数据目录下的 `wayasr/models` |

`service install` 成功启动服务后，保存实际安装参数和客户端 URL；后续安装复用这些参数，`transcribe` 和 `live` 自动使用保存的端口。监听 `0.0.0.0` 时，客户端连接 `127.0.0.1`；监听 `::` 时连接 `[::1]`。如需保留 GPU 可见性限制，可用 `--keep-device-visibility` 覆盖此前保存的清除选项。

配置文件损坏时会明确报错，不会静默使用错误的服务地址。已有 systemd drop-in 若修改实际监听地址或端口，仍需同步配置；安装器保存的是自身安装参数，不解析外部覆盖项。

CLI 使用 Click，主命令和每个子命令均支持 `--help`。安装了项目的 Python 环境中也可使用 `python -m wayasr` 调用；开发运行方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Linux 用户服务（可选）

在已登录的 Linux 用户会话中执行，无需 sudo 或手动编辑 unit 文件：

```sh
wayasr config init
wayasr service install
```

也可以跳过初始化，直接用 `--model-dir ./models` 等参数安装；缺失的模型仍需先下载。

该命令校验模型和 `llama-server`，将绝对路径写入 `~/.config/systemd/user/wayasr.service`（尊重 `XDG_CONFIG_HOME`），自动重新加载 systemd、启用登录自启动并启动服务。重复执行会更新配置并重启服务。使用与 `serve` 相同的参数配置设备、端口和模型，例如：

```sh
wayasr service install --model-dir ./models --device ROCm0 --clear-device-visibility
wayasr service install --model-dir ./models --device none --flash-attn off
```

服务直接运行安装时解析到的 `llama-server`，不依赖安装目录或虚拟环境路径，但模型和可执行文件移动后需要重新执行安装命令。服务继承 systemd 用户管理器的环境，而不是安装命令所在 shell 的全部环境；`--clear-device-visibility` 会在 unit 中明确清除 GPU 可见性限制。已有自定义 drop-in 保留且仍可能覆盖安装参数。

```sh
systemctl --user status wayasr.service
journalctl --user -u wayasr.service -f
systemctl --user stop wayasr.service
wayasr service uninstall
```

卸载会停止服务、取消自启动并删除生成的 unit，不删除模型、持久化配置或自定义 drop-in；客户端仍保留已保存的连接地址，便于重新部署。安装完成不代表模型已经加载就绪，可通过 `/health` 检查；模型加载失败请查看日志。

用户服务通常随登录会话启动；是否需要登出后继续运行由系统的 user lingering 配置决定。
