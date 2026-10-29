# WayASR

使用 llama.cpp 在本地转写音频，提供优先按停顿分段的实时字幕，以及简洁的 Rich Live 终端界面。默认模型为 Qwen3-ASR-1.7B Q8_0。

- 文件转写：连接本地或远程 `/v1/audio/transcriptions` 服务。
- 实时字幕：采集 Linux 系统声音或麦克风，保留中间识别结果的可修正末尾。
- 本地服务：启动独立安装的 `llama-server`，可选择 CPU、CUDA、ROCm、Vulkan 等设备；实际可用后端取决于 llama.cpp 的构建。

## 1. 使用 uv 安装

需要 Python 3.10+；新环境建议 Python 3.12，并确认 ONNX Runtime 有对应平台的安装包。先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。PyPI 上的发行版可直接安装：

```sh
uv tool install 'wayasr[live]'
uv tool update-shell
wayasr --help
```

`uv tool install` 将命令安装到独立环境，不需要手动创建或激活虚拟环境。如果 shell 尚未包含 uv 的工具目录，请按照 `uv tool update-shell` 提示重新打开终端。需要指定 Python 时可加 `--python 3.12`。

如果只需要文件转写、启动服务或管理 systemd，安装基础版本即可，不需要 ONNX Runtime、VAD 或 Rich：

```sh
uv tool install wayasr
```

也可以从源码安装：

```sh
git clone https://github.com/sitiyou/wayasr.git
cd wayasr
uv tool install '.[live]'
```

从源码安装后修改了代码，可在项目目录重新运行安装命令并加 `--force` 更新；卸载使用 `uv tool uninstall wayasr`。工具卸载不会自动卸载 systemd 服务，需要先执行 `wayasr service uninstall`。

实时采集使用 NumPy 和 ONNX Runtime 在 CPU 上运行 Silero VAD，不依赖 PyTorch 或 torchaudio。固定版本的 ONNX 模型及 MIT 许可证随包附带，无需首次联网下载。ASR 推理由独立的服务处理。

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

服务尊重当前 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES` 等限制，不修改 GPU 可见性。

其他常用选项：

```sh
wayasr serve --server-bin /path/to/llama-server \
  --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 --port 8087 --gpu-layers 99 --flash-attn auto
```

不要把没有认证保护的服务直接暴露到公网。跨机器访问建议使用 SSH 隧道或经过认证的反向代理。

## 4. 文件转写

可直接输入 WAV、FLAC、Ogg、MP3 等 libsndfile 支持的音频格式。工具使用 librosa 自动转为单声道、16 kHz、16-bit PCM，不修改原文件，也不调用 FFmpeg 等外部程序；已符合要求的 WAV 直接读取。格式支持取决于所安装的 libsndfile，AAC/M4A 或视频容器不保证支持。

```sh
wayasr transcribe input.mp3 --language Chinese
```

省略 `--language` 自动识别语言。连接其他地址或增加超时：

```sh
wayasr transcribe audio.wav --url http://127.0.0.1:8087 --timeout 120
```

当前文件模式将转换后的整段音频一次提交；长文件应先分段，支持的最大时长取决于服务和模型。

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
uv tool install --force --python 3.14 '.[live,wayland]'
```

在已登录的 Wayland 图形会话中运行：

```sh
wayasr live --overlay --language Chinese
wayasr live --overlay --plain --source @DEFAULT_SOURCE@ --language Chinese
wayasr live --overlay --overlay-font-size 48 --overlay-duration 10
```

字幕显示在屏幕底部中央，文本区域约为屏幕宽度的 50%，生成 ASS 时主动插入强制换行，不依赖引擎自动断行。默认字号下每行最多 22 个非组合字符，字号增大时字符上限相应减少；稳定文本和待确认文本共用同一行的计数。稳定文本为白色，待确认文本为灰色；下一条识别结果替换当前字幕。最后一句默认保留 8 秒，无更新后消失。`--overlay-font-size` 设置 ASS 字号（12–120），`--overlay-duration` 设置无更新时的保留秒数（1–60）。识别文本中的反斜杠和花括号显示为全角字符，避免被解释为 ASS 控制指令。

桌面字幕与终端输出同时保留；`--plain` 只禁用 Rich TUI，不禁用桌面字幕。Ctrl+C 结束采集、处理待完成转写，并关闭叠加层。该基础集成不提供托盘、菜单或拖拽操作，默认鼠标穿透；合成器规则可按 layer-shell namespace `wayasr` 匹配，与 layrics 的显示层独立。

桌面字幕默认关闭，使用 `--overlay` 开启；不在初始化中配置。桌面显示并未扩展采集平台范围，音频仍通过 PulseAudio/PipeWire-Pulse 的 `parec` 获取。

如果没有显示，请确认服务已就绪、音源有声音、`WAYLAND_DISPLAY` 有效、合成器支持 layer-shell，以及 Vulkan 驱动可用；查看终端中的 asslayer 日志。

### 可选 GGUF 强制对齐（实验）

`wayasr config init` 可选择 llama.cpp 或 [audio.cpp](https://github.com/0xShug0/audio.cpp)。llama.cpp 使用原有的标点/VAD 分段；audio.cpp 在同一个服务中运行 ASR 和可选的 Qwen3-ForcedAligner。本地初始化默认启用 FA，也可关闭；远程初始化默认关闭 FA。不支持混合两个后端。运行时自动读取服务 URL、模型 ID 和 FA 开关。

安装支持 `qwen3_forced_aligner` 的 `audiocpp_server`。ASR 和 FA 模型通过菜单选择预设模型或本地模型，首次启动服务会从 `audio-cpp/audio.cpp-gguf` 自动下载默认的 Qwen3-ASR-1.7B Q8_0，以及启用 FA 时的 Qwen3-ForcedAligner-0.6B Q8_0，保存到本地模型目录的 `audio.cpp` 子目录；后续启动无需重复下载。可通过 `HF_ENDPOINT` 设置镜像。`config init` 仅保存配置，不下载模型。

也可预先下载 **audio.cpp 原生格式**的 GGUF，并在初始化时填写本地绝对路径；llama.cpp 的 GGUF 不保证兼容。例如手动下载 FA 模型：

```bash
HF_ENDPOINT=https://hf-mirror.com uvx --from huggingface-hub hf download \
  audio-cpp/audio.cpp-gguf \
  Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf
```

初始化本地 audio.cpp 服务时，ASR 默认选择 `Qwen3-ASR-1.7B Q8_0`，FA 默认选择 `Qwen3-ForcedAligner-0.6B Q8_0`，无需填写仓库 ID 或文件名。选择 `Local file` 后填写已有模型文件的绝对路径，即可使用自定义模型，不触发下载。工具会在启动时生成服务配置。连接外部服务时，由用户配置两个模型，例如：

```json
{
  "host": "127.0.0.1",
  "port": 8087,
  "backend": "cpu",
  "threads": 8,
  "lazy_load": true,
  "models": [
    {
      "id": "qwen3-asr",
      "family": "qwen3_asr",
      "path": "/absolute/path/to/audio-cpp-asr-model",
      "task": "asr",
      "mode": "offline"
    },
    {
      "id": "qwen3-align",
      "family": "qwen3_forced_aligner",
      "path": "/absolute/path/to/qwen3-forced-aligner-0.6b-q8_0.gguf",
      "task": "align",
      "mode": "offline"
    }
  ]
}
```

外部服务启动后，初始化选择 `Remote address` 和 `audio.cpp`，填写服务根地址及 ASR 模型 ID；如服务部署了对齐模型，也可启用 FA 并填写对齐模型 ID。本地 audio.cpp 服务可用 `wayasr serve` 或 `wayasr service install` 管理：

```bash
audiocpp_server --config server.json --no-ui
wayasr live --language Chinese --url http://127.0.0.1:8087
```

ASR 和 FA 的内部 ID 分别为 `qwen3-asr` 和 `qwen3-align`，无需填写。`live.fa` 控制是否启用对齐。语言默认由 ASR 检测后传给 FA，也可用 `--language` 覆盖；需要支持 `/v1/audio/transcriptions/details` 的 audio.cpp 版本。启用对齐时 `--partial-interval` 必须大于 0。设备从当前环境可见的列表中选择，WayASR 不修改 GPU 可见性环境变量。

同一段连续两次识别的公共前缀包含完整句子、且音频窗口累计约 5 秒后，才尝试对齐；同一候选句子不会随每次 partial 重复对齐。只提交时间戳确认的前缀及其已稳定文本，保留约 512 ms 音频上下文和边界之后的全部音频。时间戳异常、文本不匹配、超时或服务不可用时，会明确报告错误，不静默回退到标点切分；正常的静音分段和最大段长仍生效。

本地服务默认闲置 180 秒后卸载模型，初始化不再询问。可手动修改 `server.idle_seconds`，`0` 表示禁用。llama.cpp 使用 `--sleep-idle-seconds`，audio.cpp 使用 `idle_unload_ms`；再次请求自动加载，但首个请求会增加加载延迟。外部服务需自行设置。需要支持相应功能的服务版本。

对齐会增加额外推理成本，且不能纠正错误转写。首次请求还包含模型加载时间；是否改善实际延迟，需要用自己的音频测量。

### 分段与识别限制

- 累计约 5 秒后，优先按稳定的句末标点切分，不要求 VAD 静音。同一段连续两次识别文本完全一致，且末尾为句末标点时，允许提交该次识别覆盖的音频，并为下一段保留约 512 ms 音频上下文。异步识别期间新采集的音频也继续保留。
- 未启用强制对齐时，只有文本中间出现句末标点、末尾仍有新句内容时，不据此猜测音频切点；普通转写接口没有词时间戳。
- 没有可用的标点切分结果时，才使用 VAD 兜底：连续约 480 ms 静音结束一段；累计约 5 秒后，约 320 ms 的停顿也允许提前提交。默认约 15 秒为硬上限。`--partial-interval 0` 禁用中间识别及标点切分，VAD 兜底仍生效。
- 到达硬上限时，优先在最近约 3 秒寻找至少约 128 ms 的停顿，切点之后的音频继续保留。
- 没有停顿才硬切，并保留约 512 ms 重叠；较小的最大段长会相应限制重叠长度。只对带重叠的相邻块尝试短文本精确去重，不能保证消除全部重复或区分全部真实重复。
- 连续两次结果一致的前缀正常显示，未稳定末尾暗色显示，仍允许后续修正。未完句跨块接续，按常见中英文句末标点换行。
- 未启用强制对齐时，无法保证切点严格对应完整句子；英文缩写等复杂断句即使启用对齐也可能不准确。
- 最终结果积压超过 16 段会明确报错，不静默丢弃。可关闭中间识别、使用更快的模型或 GPU 来减轻积压。

## 初始化与持久化配置

```sh
wayasr config init
wayasr config show
```

`config init` 首先选择 `Local service` 或 `Remote address`，再选择 `llama.cpp` 或 `audio.cpp` 后端。本地服务只配置程序、模型、设备和端口，固定监听 `127.0.0.1`，不询问地址，自动保存服务 URL；audio.cpp 还可选择是否启用 FA。远程模式填写已有服务的根地址，无需本地程序或模型；audio.cpp 还需填写 ASR 模型 ID，并可选启用 FA（默认关闭）。语言默认自动检测，音源默认使用系统监听源，桌面字幕默认关闭，均在运行时通过 CLI 覆盖，不持久化。重新初始化会清理旧字段及不适用的服务设置；取消不会覆盖文件。初始化不下载模型、不安装或重启服务。模型目录保存为绝对路径。

已有 API 的 URL 应填写服务根地址（例如 `https://asr.example.com`），不要包含 `/v1/audio/transcriptions`；客户端会自动追加该路径。服务须兼容当前客户端的 WAV multipart 上传及 JSON `text` 响应；可通过 `--backend` 和 `--asr-model` 覆盖后端及模型 ID，`--url` 仅覆盖地址，不改变后端。目前不提供 API Key 配置。

目录由 `platformdirs` 按当前平台管理。Linux 默认使用：

- 配置：`~/.config/wayasr/config.json`，尊重 `XDG_CONFIG_HOME`。
- 未配置时的模型目录：`~/.local/share/wayasr/models`，尊重 `XDG_DATA_HOME`。

macOS、Windows 使用平台标准用户目录；`config show` 显示实际配置文件位置和已保存值。

默认值优先级为：**命令行参数 → 对应环境变量 → 配置文件 → 内置默认值**。临时命令行或环境变量覆盖不会修改文件。

| 环境变量 | 用途 | 未配置时的默认值 |
| --- | --- | --- |
| `WAYASR_URL` | 文件及实时转写的服务地址 | `http://127.0.0.1:8087` |
| `WAYASR_MODEL_DIR` | 服务模型目录 | 平台用户数据目录下的 `wayasr/models` |

`service install` 成功启动服务后，保存实际安装参数和服务 URL；后续安装复用这些参数，`transcribe` 和 `live` 自动使用保存的端口。监听 `0.0.0.0` 时，客户端连接 `127.0.0.1`；监听 `::` 时连接 `[::1]`。

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
wayasr service install --model-dir ./models --device ROCm0
wayasr service install --model-dir ./models --device none --flash-attn off
```

服务直接运行安装时解析到的 `llama-server`，不依赖安装目录或虚拟环境路径，但模型和可执行文件移动后需要重新执行安装命令。服务继承 systemd 用户管理器的环境，而不是安装命令所在 shell 的全部环境，不主动清除 GPU 可见性限制。已有自定义 drop-in 保留且仍可能覆盖安装参数。

```sh
systemctl --user status wayasr.service
journalctl --user -u wayasr.service -f
systemctl --user stop wayasr.service
wayasr service uninstall
```

卸载会停止服务、取消自启动并删除生成的 unit，不删除模型、持久化配置或自定义 drop-in；客户端仍保留已保存的连接地址，便于重新部署。安装完成不代表模型已经加载就绪，可通过 `/health` 检查；模型加载失败请查看日志。

用户服务通常随登录会话启动；是否需要登出后继续运行由系统的 user lingering 配置决定。
