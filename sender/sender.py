#!/usr/bin/env python3
"""
Main sender program for the low-bandwidth voice communication system.
Captures audio from microphone, encodes it with neural codec model,
and streams the encoded tokens to a receiver.
"""

import os
import sys
import time
import logging
import argparse
import grpc
import threading
import numpy as np
import torch
import requests
from dotenv import load_dotenv
import sounddevice as sd

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from proto import voice_stream_pb2
from proto import voice_stream_pb2_grpc
from sender.audio_capture import AudioCapture
from sender.encoder import Encoder
from utils.audio_utils import get_timestamp_ms, list_audio_devices, measure_latency
from model.model_config import SAMPLE_RATE, CHUNK_SIZE, TARGET_BITRATE

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("Sender")

class Sender:
    """
    Main sender class that orchestrates audio capture, encoding and transmission.
    """
    
    def __init__(self, target_address, mic_device=None, sample_rate=SAMPLE_RATE,
                 bitrate=TARGET_BITRATE, verbose=False):
        """
        Initialize the sender.
        
        Args:
            target_address (str): The address of the receiver in format "host:port"
            mic_device (int): Microphone device index (None for default)
            sample_rate (int): Audio sample rate in Hz
            bitrate (float): Target bitrate in kbps
            verbose (bool): Enable verbose logging
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        self.target_address = target_address
        self.mic_device = mic_device
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.running = False
        self.sequence_number = 0
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA device for encoding")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon GPU for encoding")
        else:
            self.device = "cpu"
            logger.info("Using CPU for encoding")
            
        # Initialize components
        self.audio_capture = None
        self.encoder = None
        self.grpc_channel = None
        self.stub = None
        
    def setup(self):
        """Set up audio capture, encoder, and gRPC connection."""
        try:
            # Initialize audio capture
            logger.info("Initializing audio capture...")
            self.audio_capture = AudioCapture(
                device_index=self.mic_device,
                sample_rate=self.sample_rate,
                chunk_size=CHUNK_SIZE
            )
            
            # Initialize encoder
            logger.info("Initializing audio encoder...")
            self.encoder = Encoder(device=self.device)
            self.encoder.set_bitrate(self.bitrate)
            
            # Initialize gRPC
            logger.info(f"Setting up gRPC connection to {self.target_address}...")
            self.grpc_channel = grpc.insecure_channel(self.target_address)
            self.stub = voice_stream_pb2_grpc.VoiceStreamStub(self.grpc_channel)
            
            logger.info("Sender setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            self.cleanup()
            return False
            
    def start(self):
        """Start the sender."""
        try:
            # Start audio capture
            if not self.audio_capture.start():
                logger.error("Failed to start audio capture")
                return False
                
            # Start sending thread
            self.running = True
            self.send_thread = threading.Thread(target=self._stream_audio)
            self.send_thread.start()
            
            logger.info("Sender started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting sender: {e}")
            self.stop()
            return False
            
    def _stream_audio(self):
        """Main streaming loop (runs in a separate thread)."""
        logger.info("Audio streaming thread started")
        
        try:
            # Create a streaming RPC
            stream = self.stub.Transmit(self._generate_requests())
            
            # Wait for the RPC to complete (or error)
            response = stream
            logger.info(f"Stream completed with response: {response}")
            
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.code()}: {e.details()}")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.running = False
            logger.info("Audio streaming thread ended")
            
    def _generate_requests(self):
        """Generator that yields audio chunks for the gRPC stream."""
        sent_count = 0
        error_count = 0
        
        last_stats_time = time.time()
        total_bytes_sent = 0
        last_bytes_sent = 0
        last_sent_count = 0
        
        while self.running:
            try:
                # Capture timestamp for latency measurement
                capture_time = get_timestamp_ms()
                
                # Read audio chunk from queue
                audio_chunk = self.audio_capture.read()
                
                if audio_chunk is None:
                    time.sleep(0.01)  # Avoid busy-waiting
                    continue
                    
                # Encode the audio chunk
                encoded_data = self.encoder.encode(audio_chunk)
                
                # Check if encoding was successful
                if encoded_data == b'':
                    error_count += 1
                    if error_count > 10:
                        logger.error("Too many encoding errors, stopping")
                        break
                    continue
                
                # Measure and log encoding latency
                encode_latency = measure_latency(capture_time, "capture")
                
                # Create protobuf message
                chunk = voice_stream_pb2.AudioChunk(
                    encoded_frame=encoded_data,
                    timestamp_ms=capture_time,
                    sequence_number=self.sequence_number
                )
                
                # Increment sequence number
                self.sequence_number += 1
                sent_count += 1
                total_bytes_sent += len(encoded_data)
                
                # Log statistics periodically
                if sent_count % 100 == 0:
                    # Calculate approximate bitrate
                    avg_bytes = sum(len(c.encoded_frame) for c in [chunk]) / 1
                    estimated_kbps = (avg_bytes * 8 * (1000 / (CHUNK_SIZE / self.sample_rate * 1000))) / 1000
                    logger.info(f"Sent {sent_count} chunks, recent avg: {estimated_kbps:.2f} kbps, "
                               f"encode latency: {encode_latency}ms")
                    
                    # Send stats to monitor every 5 seconds
                    now = time.time()
                    if now - last_stats_time >= 5.0:
                        elapsed = now - last_stats_time
                        bytes_diff = total_bytes_sent - last_bytes_sent
                        chunks_diff = sent_count - last_sent_count
                        
                        # Calculate actual bitrate over the last 5 seconds
                        actual_kbps = (bytes_diff * 8 / 1000) / elapsed if elapsed > 0 else 0
                        
                        try:
                            monitor_url = os.environ.get("MONITOR_URL", "http://localhost:8765/update")
                            stats_payload = {
                                "bitrate": round(actual_kbps, 2),
                                "latency": round(encode_latency),
                                "packet_loss": 0.0,  # Sender doesn't have packet loss info
                                "buffer_size": 0  # Sender doesn't have buffer info
                            }
                            requests.post(monitor_url, json=stats_payload, timeout=0.5)
                            logger.debug(f"Sent stats to monitor: {stats_payload}")
                        except Exception as e:
                            logger.debug(f"Failed to send stats to monitor: {e}")
                            
                        last_stats_time = now
                        last_bytes_sent = total_bytes_sent
                        last_sent_count = sent_count
                
                # Yield the chunk for the gRPC stream
                yield chunk
                
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                error_count += 1
                if error_count > 10:
                    logger.error("Too many errors, stopping")
                    break
                    
    def stop(self):
        """Stop streaming and clean up resources."""
        logger.info("Stopping sender...")
        self.running = False
        
        # Wait for streaming thread to end
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
            
        self.cleanup()
        logger.info("Sender stopped")
        
    def cleanup(self):
        """Clean up resources."""
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_stream()
            
        # Close gRPC channel
        if self.grpc_channel:
            self.grpc_channel.close()
            
        self.audio_capture = None
        self.encoder = None
        self.grpc_channel = None
        self.stub = None
        
def list_audio_devices():
    """List all available audio devices."""
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    print("Index\tName")
    print("-" * 50)
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"{i}\t{device['name']}")
    print("-" * 50)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Communication Sender")
    parser.add_argument("--target", type=str,
                      help="Target receiver address in format host:port")
    parser.add_argument("--mic", type=int, default=None,
                      help="Microphone device index (None for default)")
    parser.add_argument("--bitrate", type=float, default=TARGET_BITRATE,
                      help=f"Target bitrate in kbps (default: {TARGET_BITRATE})")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio devices and exit")
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
        
    if not args.target:
        parser.error("--target is required when not listing devices")
        
    sender = Sender(
        target_address=args.target,
        mic_device=args.mic,
        bitrate=args.bitrate
    )
    
    if not sender.setup():
        logger.error("Failed to set up sender")
        return
        
    if not sender.start():
        logger.error("Failed to start sender")
        return
        
    try:
        logger.info("Sender running. Press Ctrl+C to stop.")
        while sender.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        sender.stop()
        
if __name__ == "__main__":
    sys.exit(main()) 