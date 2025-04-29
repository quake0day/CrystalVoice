# CrystalVoice: Low-Bandwidth Zero-Distortion Voice Communication Prototype

![CrystalVoice Logo](https://i.v2ex.co/E5v87f3mb.png)

[![English](https://img.shields.io/badge/Language-English-blue)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red)](#chinese)

This project implements a prototype system for real-time voice communication between two machines with ultra-low bandwidth usage (~3-6 kbps) and zero perceptible distortion in audio quality.

## Overview

The system consists of three main components:
- **Sender**: Captures live microphone audio, encodes it into compact semantic tokens, and transmits these tokens over a network.
- **Receiver**: Receives the tokens and uses a pre-trained generative voice model to reconstruct high-fidelity speech for playback.
- **WebUI Monitor**: Real-time dashboard for visualizing audio levels, network statistics, and waveforms.

Key features:
- **Ultra Low Bandwidth**: Compresses voice audio to ~3-6 kbps without quality loss
- **High Fidelity**: Preserves the speaker's timbre and nuances with zero perceptible distortion
- **Low Latency**: End-to-end latency under 300ms for natural conversation flow
- **Privacy-Preserving**: All processing is done locally with pre-trained models
- **Standard Protocol**: Uses gRPC for efficient and reliable streaming
- **Real-time Monitoring**: Interactive dashboard with auto-refreshing metrics

## Installation

### Prerequisites
- Python 3.9+ (Python 3.10 recommended)
- Working microphone and speakers
- Network connectivity between sender and receiver machines

### Setup

1. Clone this repository:
```bash
git clone https://github.com/quake0day/CrystalVoice.git
cd CrystalVoice
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate gRPC code (if not already present):
```bash
python -m grpc_tools.protoc -I proto/ --python_out=proto/ --grpc_python_out=proto/ proto/voice_stream.proto
```

## Usage

### Start the Monitor (Optional but Recommended)
Start the WebUI monitor first to visualize the audio and network metrics:

```bash
python webui/monitor_app.py --host 0.0.0.0 --port 7860 --api-port 8765
```

This will start a web interface accessible at `http://localhost:7860` and an API endpoint at port 8765 for receiving metrics.

### Start the Receiver
On the machine that will play the audio:

```bash
python receiver/receiver.py --host 0.0.0.0 --port 50051
```

If you're using the monitor, the receiver will automatically send metrics to it.

### Start the Sender
On the machine with the microphone:

```bash
python sender/sender.py --target RECEIVER_IP:50051 --mic 0 --bitrate 6
```
Replace `RECEIVER_IP` with the IP address of the receiver machine.

If you're using the monitor on a different machine, you can specify its URL:
```bash
MONITOR_URL=http://MONITOR_IP:8765/update python sender/sender.py --target RECEIVER_IP:50051
```

### Testing on a Single Machine
For testing purposes, you can run all components on the same machine:

```bash
# Terminal 1
python webui/monitor_app.py --host 0.0.0.0 --port 7860 --api-port 8765

# Terminal 2
python receiver/receiver.py --host 127.0.0.1 --port 50051

# Terminal 3
python sender/sender.py --target localhost:50051 --mic 0
```

Use headphones to avoid feedback loops.

## Command Line Options

### WebUI Monitor
- `--host HOST`: Interface to bind (default: 0.0.0.0)
- `--port PORT`: Web interface port (default: 7860)
- `--api-port PORT`: API endpoint port (default: 8765)
- `--input DEVICE_INDEX`: Audio input device for monitoring (optional)
- `--output DEVICE_INDEX`: Audio output device for monitoring (optional)
- `--list-devices`: List available audio devices and exit

### Sender
- `--target HOST:PORT`: Address of the receiver (required)
- `--mic DEVICE_INDEX`: Choose input device (default: system default)
- `--bitrate KBPS`: Compression bitrate (default: 6)
- `--list-devices`: List available audio devices and exit

### Receiver
- `--host HOST`: Interface to bind (default: 0.0.0.0)
- `--port PORT`: Port to listen on (default: 50051)
- `--speaker DEVICE_INDEX`: Audio output device (default: system default)
- `--sample-rate RATE`: Sample rate (default: 24000)
- `--bitrate KBPS`: Target bitrate (default: 6)
- `--buffer NUM_FRAMES`: Buffer size for jitter compensation (default: 3)
- `--verbose`: Enable detailed logging
- `--list-devices`: List available audio devices and exit

## WebUI Dashboard Features

The WebUI monitor dashboard provides real-time visualization of:

- **Audio Levels**: Input and output audio levels with responsive visualizations
- **Network Statistics**: Bitrate, latency, packet loss, and buffer size metrics
- **Audio Waveforms**: Real-time waveform displays for input and output audio
- **Auto-refresh**: Dashboard updates automatically every second without manual refresh

You can also manually test the metrics API with:
```bash
curl -X POST http://localhost:8765/update -H "Content-Type: application/json" -d '{"bitrate": 10.5, "latency": 80, "packet_loss": 0.2, "buffer_size": 3}'
```

## Architecture

The system uses EnCodec, a state-of-the-art neural audio codec by Meta AI, to compress audio while maintaining high fidelity. The encoder on the sender side transforms raw audio into semantic tokens, which are transmitted over gRPC. The receiver then uses a generative decoder to reconstruct the original audio from these tokens.

Both sender and receiver components automatically send performance metrics to the WebUI monitor, which visualizes them in real-time using a Gradio-based dashboard.

## License

[MIT License](LICENSE)

---

<a name="chinese"></a>

# CrystalVoice: 低带宽零失真语音通信原型系统

![CrystalVoice Logo](https://i.v2ex.co/E5v87f3mb.png)

[![English](https://img.shields.io/badge/Language-English-blue)](#) [![中文](https://img.shields.io/badge/语言-中文-red)](README.md#chinese)

本项目实现了一个实时语音通信原型系统，能够在两台机器之间以超低带宽（约3-6 kbps）进行语音传输，且无可感知的音质失真。

## 概述

系统由三个主要组件组成：
- **发送端**：捕获麦克风实时音频，将其编码为紧凑的语义令牌，并通过网络传输这些令牌。
- **接收端**：接收令牌，并使用预训练的生成式语音模型重建高保真语音进行播放。
- **WebUI监控**：实时仪表盘，用于可视化音频电平、网络统计数据和波形。

主要特点：
- **超低带宽**：将语音音频压缩到约3-6 kbps，无质量损失
- **高保真度**：保留说话者的音色和细微差别，无可感知的失真
- **低延迟**：端到端延迟低于300毫秒，确保自然的对话流程
- **保护隐私**：所有处理都在本地使用预训练模型完成
- **标准协议**：使用gRPC进行高效可靠的流式传输
- **实时监控**：具有自动刷新功能的交互式仪表盘

## 安装

### 前提条件
- Python 3.9+（推荐Python 3.10）
- 可用的麦克风和扬声器
- 发送端和接收端机器之间的网络连接

### 设置

1. 克隆此存储库：
```bash
git clone https://github.com/quake0day/CrystalVoice.git
cd CrystalVoice
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Windows上：venv\Scripts\activate
```

3. 安装依赖项：
```bash
pip install -r requirements.txt
```

4. 生成gRPC代码（如果尚未存在）：
```bash
python -m grpc_tools.protoc -I proto/ --python_out=proto/ --grpc_python_out=proto/ proto/voice_stream.proto
```

## 使用方法

### 启动监控（可选但推荐）
首先启动WebUI监控，以可视化音频和网络指标：

```bash
python webui/monitor_app.py --host 0.0.0.0 --port 7860 --api-port 8765
```

这将启动一个可在`http://localhost:7860`访问的Web界面，并在端口8765上提供一个用于接收指标的API端点。

### 启动接收端
在将播放音频的机器上：

```bash
python receiver/receiver.py --host 0.0.0.0 --port 50051
```

如果您使用监控，接收端会自动向其发送指标数据。

### 启动发送端
在带有麦克风的机器上：

```bash
python sender/sender.py --target 接收端IP:50051 --mic 0 --bitrate 6
```
请将`接收端IP`替换为接收端机器的IP地址。

如果监控在不同的机器上，您可以指定其URL：
```bash
MONITOR_URL=http://监控IP:8765/update python sender/sender.py --target 接收端IP:50051
```

### 在单机上测试
出于测试目的，您可以在同一台机器上运行所有组件：

```bash
# 终端1
python webui/monitor_app.py --host 0.0.0.0 --port 7860 --api-port 8765

# 终端2
python receiver/receiver.py --host 127.0.0.1 --port 50051

# 终端3
python sender/sender.py --target localhost:50051 --mic 0
```

使用耳机避免回音。

## 命令行选项

### WebUI监控
- `--host HOST`：绑定的接口（默认：0.0.0.0）
- `--port PORT`：Web界面端口（默认：7860）
- `--api-port PORT`：API端点端口（默认：8765）
- `--input DEVICE_INDEX`：监控用的音频输入设备（可选）
- `--output DEVICE_INDEX`：监控用的音频输出设备（可选）
- `--list-devices`：列出可用的音频设备并退出

### 发送端
- `--target HOST:PORT`：接收端的地址（必需）
- `--mic DEVICE_INDEX`：选择输入设备（默认：系统默认）
- `--bitrate KBPS`：压缩比特率（默认：6）
- `--list-devices`：列出可用的音频设备并退出

### 接收端
- `--host HOST`：绑定的接口（默认：0.0.0.0）
- `--port PORT`：监听的端口（默认：50051）
- `--speaker DEVICE_INDEX`：音频输出设备（默认：系统默认）
- `--sample-rate RATE`：采样率（默认：24000）
- `--bitrate KBPS`：目标比特率（默认：6）
- `--buffer NUM_FRAMES`：用于抖动补偿的缓冲区大小（默认：3）
- `--verbose`：启用详细日志记录
- `--list-devices`：列出可用的音频设备并退出

## WebUI仪表盘功能

WebUI监控仪表盘提供以下实时可视化功能：

- **音频电平**：输入和输出音频电平，带有响应式可视化
- **网络统计**：比特率、延迟、丢包率和缓冲区大小指标
- **音频波形**：输入和输出音频的实时波形显示
- **自动刷新**：仪表盘无需手动刷新，每秒自动更新

您也可以手动测试指标API：
```bash
curl -X POST http://localhost:8765/update -H "Content-Type: application/json" -d '{"bitrate": 10.5, "latency": 80, "packet_loss": 0.2, "buffer_size": 3}'
```

## 架构

系统使用Meta AI的EnCodec（一种最先进的神经音频编解码器）来压缩音频，同时保持高保真度。发送端的编码器将原始音频转换为语义令牌，通过gRPC传输。接收端然后使用生成式解码器从这些令牌重建原始音频。

发送端和接收端组件会自动将性能指标发送到WebUI监控，后者使用基于Gradio的仪表盘实时可视化这些数据。

## 许可证

[MIT许可证](LICENSE) 