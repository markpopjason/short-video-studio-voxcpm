# VoxCPM Studio

本项目用 VoxCPM2 实现本地声音克隆，模型默认从魔塔社区 `OpenBMB/VoxCPM2` 下载，并通过 Gradio 同时提供网页界面和 API。

## 一键启动

Windows 双击：

```bat
start_windows.bat
```

这个入口不要求提前安装 Python 或 Torch。它会自动下载便携 `uv`，用 `uv` 安装托管 Python 3.11，创建项目本地 `.venv`，再把 PyTorch、VoxCPM、Gradio、ModelScope 安装进去。

`start_windows.bat` 会自动判断 PyTorch 后端：检测到 `nvidia-smi` 时安装 CUDA 12.1 版 PyTorch，否则安装 CPU 版 PyTorch。如果自动检测不符合你的机器，可以手动指定：

```bat
set VOXCPM_TORCH_BACKEND=cu121
start_windows.bat
```

`VOXCPM_TORCH_BACKEND` 支持 `cpu`、`cu121`、`cu124`。

macOS/Linux：

```bash
chmod +x start_macos_linux.sh
./start_macos_linux.sh
```

启动后打开 `http://127.0.0.1:8808`。首次生成或点击“下载/检查模型”时会下载模型到 `pretrained_models/VoxCPM2`。

## 命令行

```bash
python -m voxcpm_studio download
python -m voxcpm_studio serve --device auto --port 8808
python -m voxcpm_studio synthesize --text "你好" --reference-audio ref.wav --output outputs/out.wav
```

## Gradio API

启动服务后，接口名是 `/generate_voice`，页面右下角的 API 文档会显示可直接调用的 Python 和 curl 示例。

## 运行要求

- Windows 不需要预装 Python 或 Torch，但首次启动需要联网下载运行时和依赖
- Linux/macOS 仍使用本机 Python 3.10 到 3.12
- 推荐 NVIDIA GPU，VoxCPM2 约需 8GB 显存
- CPU 可以启动界面，但生成速度会明显较慢

如果 Windows 安装停在 `torch`，可先设置 `VOXCPM_TORCH_BACKEND=cpu` 或 `VOXCPM_TORCH_BACKEND=cu121` 后重新运行 `start_windows.bat`。

请只克隆你拥有授权的声音，并对合成音频进行清晰标注。
