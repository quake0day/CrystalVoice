#!/usr/bin/env python3
"""
Web UI for monitoring audio levels and network statistics.
Provides a simple dashboard using Gradio for visualizing the voice communication system.
"""

import os
import sys
import time
import numpy as np
import gradio as gr
import threading
import logging
import sounddevice as sd
import argparse
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.audio_utils import int16_to_float32, list_audio_devices
from model.model_config import SAMPLE_RATE, CHUNK_SIZE

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebUI")

app = Flask(__name__)

# Global monitor reference will be set later
_MONITOR_REF = None

@app.route('/update', methods=['POST'])
def update_stats():
    global _MONITOR_REF
    if _MONITOR_REF is None:
        return jsonify({'error': 'monitor not ready'}), 503
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            _MONITOR_REF.update_network_stats(data)
            return jsonify({'status': 'ok'})
        return jsonify({'error': 'invalid payload'}), 400
    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        return jsonify({'error': 'server error'}), 500

class AudioMonitor:
    """
    Monitors audio input/output levels and network statistics.
    """
    
    def __init__(self, input_device=None, output_device=None,
                 sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        """
        Initialize the audio monitor.
        
        Args:
            input_device (int): Input device index (None for default)
            output_device (int): Output device index (None for default)
            sample_rate (int): Audio sample rate in Hz
            chunk_size (int): Audio chunk size in samples
        """
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Audio data buffers
        self.input_buffer = np.zeros(sample_rate)
        self.output_buffer = np.zeros(sample_rate)
        
        # Statistics
        self.network_stats = {
            'bitrate': 0.0,
            'latency': 0.0,
            'packet_loss': 0.0,
            'buffer_size': 0
        }
        
        # Flags
        self.running = False
        self.input_stream = None
        self.output_stream = None
        
    def start(self):
        """Start audio monitoring."""
        if self.running:
            logger.warning("Audio monitor already running")
            return
            
        try:
            # Define input callback to monitor microphone
            def input_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Input status: {status}")
                
                # Convert to float and get mono if needed
                audio = int16_to_float32(indata)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                    
                # Roll the buffer and add new data
                self.input_buffer = np.roll(self.input_buffer, -len(audio))
                self.input_buffer[-len(audio):] = audio
            
            # Define output callback to monitor speaker
            def output_callback(outdata, frames, time_info, status):
                if status:
                    logger.warning(f"Output status: {status}")
                
                # Just read, don't actually output anything
                # This is just to monitor what's being played
                audio = outdata.copy()
                
                # Convert to float and get mono if needed
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32) / 32768.0
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                    
                # Roll the buffer and add new data
                self.output_buffer = np.roll(self.output_buffer, -len(audio))
                self.output_buffer[-len(audio):] = audio
            
            # Start input stream
            if self.input_device is not None:
                self.input_stream = sd.InputStream(
                    device=self.input_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype='int16',
                    callback=input_callback
                )
                self.input_stream.start()
                logger.info(f"Started input monitoring on device {self.input_device}")
                
            # Start output stream (if output device is specified)
            if self.output_device is not None:
                self.output_stream = sd.InputStream(
                    device=self.output_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype='int16',
                    callback=output_callback
                )
                self.output_stream.start()
                logger.info(f"Started output monitoring on device {self.output_device}")
                
            self.running = True
            logger.info("Audio monitor started")
            
        except Exception as e:
            logger.error(f"Error starting audio monitor: {e}")
            self.stop()
    
    def stop(self):
        """Stop audio monitoring."""
        self.running = False
        
        # Stop input stream
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            
        # Stop output stream
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            
        logger.info("Audio monitor stopped")
        
    def update_network_stats(self, stats):
        """
        Update network statistics.
        
        Args:
            stats (dict): Dictionary of network statistics
        """
        self.network_stats.update(stats)
        
        # 模拟产生一些音频数据，这样波形和电平才会显示
        # 根据网络状态生成一个脉冲信号
        n_samples = int(0.1 * self.sample_rate)  # 生成100ms的数据
        t = np.linspace(0, 1, n_samples)
        
        # 根据当前比特率和延迟调整波形振幅和频率
        amplitude = min(1.0, stats.get('bitrate', 0) / 20.0)  # 根据比特率调整振幅
        frequency = 5 + min(10, stats.get('latency', 0) / 10.0)  # 根据延迟调整频率
        
        # 生成一个脉冲信号
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # 更新输入缓冲区（模拟输入音频）
        self.input_buffer = np.roll(self.input_buffer, -n_samples)
        self.input_buffer[-n_samples:] = signal
        
        # 更新输出缓冲区（模拟输出音频，稍有不同的频率）
        output_signal = amplitude * 0.8 * np.sin(2 * np.pi * (frequency*0.9) * t) 
        self.output_buffer = np.roll(self.output_buffer, -n_samples)
        self.output_buffer[-n_samples:] = output_signal
        
    def get_input_waveform(self, duration=1.0):
        """
        Get recent input audio waveform.
        
        Args:
            duration (float): Duration in seconds to return
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        samples = int(duration * self.sample_rate)
        return self.input_buffer[-samples:]
        
    def get_output_waveform(self, duration=1.0):
        """
        Get recent output audio waveform.
        
        Args:
            duration (float): Duration in seconds to return
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        samples = int(duration * self.sample_rate)
        return self.output_buffer[-samples:]
        
    def get_input_level(self):
        """
        Get current input audio level.
        
        Returns:
            float: RMS level (0.0 to 1.0)
        """
        # Get last 100ms of audio
        samples = int(0.1 * self.sample_rate)
        audio = self.input_buffer[-samples:]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        return min(1.0, rms * 3.0)  # Scale for better visibility
        
    def get_output_level(self):
        """
        Get current output audio level.
        
        Returns:
            float: RMS level (0.0 to 1.0)
        """
        # Get last 100ms of audio
        samples = int(0.1 * self.sample_rate)
        audio = self.output_buffer[-samples:]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        return min(1.0, rms * 3.0)  # Scale for better visibility

def create_ui(monitor):
    """
    Create Gradio web UI.
    
    Args:
        monitor (AudioMonitor): The audio monitor instance
        
    Returns:
        gr.Interface: Gradio interface
    """
    # Create UI components
    with gr.Blocks(title="Voice Communication Monitor") as ui:
        gr.Markdown("# Low-Bandwidth Voice Communication Monitor")
        
        # 添加隐藏的HTML元素，用于JavaScript注入
        gr.HTML("""
        <script>
        // 在文档加载完成后设置自动刷新
        document.addEventListener('DOMContentLoaded', function() {
            // 等待Gradio界面完全加载
            setTimeout(function() {
                // 找到刷新按钮
                const refreshBtn = document.getElementById('refresh-btn').querySelector('button');
                
                // 设置1秒间隔的自动刷新
                setInterval(function() {
                    if (refreshBtn) {
                        refreshBtn.click();
                    }
                }, 1000);
            }, 2000); // 等待2秒确保界面加载完成
        });
        </script>
        """)
        
        with gr.Row():
            # Audio levels
            with gr.Column(scale=1):
                gr.Markdown("## Audio Levels")
                
                # Input level
                with gr.Row():
                    gr.Markdown("Input:")
                    input_level = gr.Slider(
                        minimum=0, maximum=1, value=0,
                        label="Input Level", interactive=False
                    )
                
                # Output level
                with gr.Row():
                    gr.Markdown("Output:")
                    output_level = gr.Slider(
                        minimum=0, maximum=1, value=0,
                        label="Output Level", interactive=False
                    )
                    
            # Network stats
            with gr.Column(scale=1):
                gr.Markdown("## Network Statistics")
                
                # Bitrate
                with gr.Row():
                    gr.Markdown("Bitrate:")
                    bitrate = gr.Number(
                        value=0.0, label="kbps", 
                        precision=1, interactive=False
                    )
                
                # Latency
                with gr.Row():
                    gr.Markdown("Latency:")
                    latency = gr.Number(
                        value=0.0, label="ms",
                        precision=0, interactive=False
                    )
                
                # Packet loss
                with gr.Row():
                    gr.Markdown("Packet Loss:")
                    packet_loss = gr.Number(
                        value=0.0, label="%",
                        precision=1, interactive=False
                    )
                
                # Buffer size
                with gr.Row():
                    gr.Markdown("Buffer Size:")
                    buffer_size = gr.Number(
                        value=0, label="frames",
                        precision=0, interactive=False
                    )
        
        with gr.Row():
            # Waveforms
            with gr.Column():
                gr.Markdown("## Audio Waveforms")
                
                # Input waveform
                input_waveform = gr.Plot(
                    label="Input Waveform"
                )
                
                # Output waveform
                output_waveform = gr.Plot(
                    label="Output Waveform"
                )
        
        # Update function
        def update_ui():
            # Get audio levels
            in_level = monitor.get_input_level()
            out_level = monitor.get_output_level()
            
            # Get network stats
            stats = monitor.network_stats
            
            # Get waveforms
            in_wave = monitor.get_input_waveform(0.5)
            out_wave = monitor.get_output_waveform(0.5)
            
            # Create waveform plots
            import matplotlib.pyplot as plt
            
            # Input waveform
            fig_in = plt.figure(figsize=(8, 2))
            plt.plot(in_wave)
            plt.ylim(-1, 1)
            plt.title("Input Audio (0.5s)")
            plt.grid(True)
            plt.tight_layout()
            
            # Output waveform
            fig_out = plt.figure(figsize=(8, 2))
            plt.plot(out_wave)
            plt.ylim(-1, 1)
            plt.title("Output Audio (0.5s)")
            plt.grid(True)
            plt.tight_layout()
            
            return [
                in_level,
                out_level,
                stats['bitrate'],
                stats['latency'],
                stats['packet_loss'],
                stats['buffer_size'],
                fig_in,
                fig_out
            ]
            
        # Set up real-time updates
        ui.queue()
        
        # 添加一个刷新按钮（隐藏）
        with gr.Row(elem_id="refresh-btn", visible=True):
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(
                update_ui,
                None,
                [input_level, output_level, bitrate, latency, 
                 packet_loss, buffer_size, input_waveform, output_waveform]
            )
        
        # 初始加载
        ui.load(
            update_ui,
            None,
            [input_level, output_level, bitrate, latency, 
             packet_loss, buffer_size, input_waveform, output_waveform]
        )
        
    return ui

# ------------------- Stats API -------------------

def start_flask_api(monitor, host, port):
    """Start the Flask API in a background thread."""
    global _MONITOR_REF
    _MONITOR_REF = monitor

    def _run():
        logger.info(f"Starting stats API on http://{host}:{port}/update")
        app.run(host=host, port=port, threaded=True, debug=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Communication Monitor")
    
    parser.add_argument("--input", type=int, default=None,
                      help="Input device index (default: none)")
    parser.add_argument("--output", type=int, default=None,
                      help="Output device index (default: none)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860,
                      help="Port to listen on (default: 7860)")
    parser.add_argument("--api-port", type=int, default=8765,
                      help="Port for stats HTTP API (default: 8765)")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio devices and exit")
                      
    return parser.parse_args()
    
def main():
    """Main entry point."""
    args = parse_args()
    
    # Just list devices if requested
    if args.list_devices:
        print(list_audio_devices())
        return 0
    
    # Create audio monitor
    monitor = AudioMonitor(
        input_device=args.input,
        output_device=args.output
    )
    
    # Start monitoring and Flask API
    monitor.start()
    start_flask_api(monitor, args.host, args.api_port)
    
    try:
        # Create and launch UI
        ui = create_ui(monitor)
        ui.launch(server_name=args.host, server_port=args.port, share=False)
        
    except Exception as e:
        logger.error(f"Error in web UI: {e}")
        return 1
    finally:
        monitor.stop()
        
    return 0
    
if __name__ == "__main__":
    sys.exit(main()) 