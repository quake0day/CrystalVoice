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
import signal
import pickle # For deserializing test output if needed later

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from proto import voice_stream_pb2
from proto import voice_stream_pb2_grpc
from sender.audio_capture import AudioCapture
from utils.audio_utils import list_audio_devices
from sender.feature_extractor import AudioFeatureExtractor, TARGET_SAMPLE_RATE, CHUNK_SAMPLES
from utils.audio_utils import get_timestamp_ms, measure_latency
from model.model_config import SAMPLE_RATE, CHUNK_SIZE, TARGET_BITRATE
from utils.monitor_client import MonitorClient

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

# Global flag to signal shutdown
shutdown_flag = threading.Event()

class AudioProcessor:
    """
    Processes audio chunks: extracts features and sends them via gRPC.
    Also sends metrics to the monitor.
    """
    def __init__(self, target, device="cpu", monitor_url=None):
        self.target = target
        self.feature_extractor = AudioFeatureExtractor(device=device)
        self.channel = None
        self.stub = None
        self.connected = False
        self.sequence_id = 0
        self.monitor_client = MonitorClient(monitor_url) if monitor_url else None
        self.last_monitor_update = time.time()

    def connect(self):
        """Establish gRPC connection."""
        try:
            # Use insecure channel for local testing
            self.channel = grpc.insecure_channel(self.target)
            self.stub = voice_stream_pb2_grpc.VoiceStreamerStub(self.channel)
            # Add a simple connectivity check if possible, e.g., a dummy unary call
            # Or rely on the stream failing if connection is bad
            self.connected = True
            logger.info(f"Successfully connected to gRPC server at {self.target}")
            return True
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to gRPC server at {self.target}: {e}")
            self.connected = False
            return False

    def _send_data_iterator(self, audio_capture):
        """Generator function to yield VoiceChunk messages."""
        while not shutdown_flag.is_set():
            audio_frame = audio_capture.read(timeout=0.1) # Read with timeout
            if audio_frame is None:
                # No data available currently, yield control briefly
                # time.sleep(0.005) # Small sleep to prevent busy-waiting if needed
                continue

            start_time = time.time()

            # 1. Extract features
            tokens_bytes, embedding_bytes = self.feature_extractor.extract_features(audio_frame)

            if not tokens_bytes or not embedding_bytes:
                 logger.warning("Feature extraction failed, skipping frame.")
                 continue # Skip this frame if extraction failed

            # 2. Create protobuf message
            timestamp_ms = int(time.time() * 1000)
            chunk_msg = voice_stream_pb2.VoiceChunk(
                sequence_id=self.sequence_id,
                timestamp_ms=timestamp_ms,
                hubert_tokens=tokens_bytes,
                speaker_embedding=embedding_bytes
            )
            self.sequence_id += 1

            # 3. Yield the message for sending
            yield chunk_msg

            # --- Monitoring --- (Optional, can be simplified/removed for initial test)
            end_time = time.time()
            processing_latency = (end_time - start_time) * 1000 # ms
            current_time = time.time()
            if self.monitor_client and (current_time - self.last_monitor_update >= 5):
                # Simulate some metrics for now
                metrics = {
                    "bitrate": 0, # Not applicable now
                    "latency": int(processing_latency), # Use processing time as a proxy
                    "packet_loss": 0, # Not tracked here
                    "buffer_size": audio_capture.get_queue_size()
                }
                self.monitor_client.send_metrics(metrics)
                self.last_monitor_update = current_time
            # --- End Monitoring ---

        logger.info("Sender loop finished.")


    def start_streaming(self, audio_capture):
        """Starts the bidirectional streaming process."""
        if not self.connected:
            logger.error("Cannot start streaming, not connected to server.")
            return

        logger.info("Starting audio streaming...")
        try:
            # Create the iterator for sending data
            send_iterator = self._send_data_iterator(audio_capture)

            # Start the bidirectional stream
            # The response stream is currently unused but required by the proto definition
            response_iterator = self.stub.StreamVoice(send_iterator)

            # Consume responses (or just ignore them for now)
            # In a real scenario, this could handle ACKs or control messages
            for ack in response_iterator:
                # logger.debug(f"Received ACK for seq: {ack.sequence_id}, status: {ack.status}")
                pass # Just consume for now

        except grpc.RpcError as e:
            logger.error(f"gRPC Error during streaming: {e.status()} - {e.details()}")
            self.connected = False # Mark as disconnected on error
        except Exception as e:
            logger.error(f"An unexpected error occurred during streaming: {e}", exc_info=True)
        finally:
            logger.info("Streaming stopped.")
            # Consider trying to reconnect if disconnected due to error

    def close(self):
        """Close the gRPC connection."""
        if self.channel:
            self.channel.close()
            logger.info("gRPC connection closed.")
        self.connected = False

def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info("Shutdown signal received. Stopping sender...")
    shutdown_flag.set()

def main():
    parser = argparse.ArgumentParser(description="Audio Sender using HuBERT and Speaker Embeddings")
    parser.add_argument("--target", required=True, help="Receiver address (e.g., localhost:50051)")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device index (use --list-devices to see options)")
    # parser.add_argument("--bitrate", type=float, default=6.0, help="Target bitrate (Not used in this version)") # Removed bitrate
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for models (cpu, cuda, mps)")
    parser.add_argument("--monitor-url", default=os.environ.get("MONITOR_URL"), help="URL for the WebUI monitor API (e.g., http://localhost:8765/update)")
    parser.add_argument("--offline", action="store_true", help="Use only local cached models")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize Audio Capture (ensure sample rate matches feature extractor)
    try:
        audio_capture = AudioCapture(
            device_index=args.mic,
            sample_rate=TARGET_SAMPLE_RATE, # Use 16kHz
            chunk_size=CHUNK_SAMPLES,       # Use 320 samples
            monitor_url=args.monitor_url
        )
        audio_capture.start()
        logger.info("Audio capture started.")
    except Exception as e:
        logger.error(f"Failed to initialize audio capture: {e}", exc_info=True)
        sys.exit(1)

    # Initialize Audio Processor (which includes Feature Extractor)
    try:
        processor = AudioProcessor(
            target=args.target,
            device=args.device,
            monitor_url=args.monitor_url
        )
    except Exception as e:
        logger.error(f"Failed to initialize Audio Processor / Feature Extractor: {e}", exc_info=True)
        if audio_capture:
             audio_capture.stop()
        sys.exit(1)

    # Connection and streaming loop
    while not shutdown_flag.is_set():
        if not processor.connected:
            logger.info("Attempting to connect to receiver...")
            if not processor.connect():
                logger.warning("Connection failed, retrying in 5 seconds...")
                shutdown_flag.wait(5) # Wait for 5 seconds or until shutdown signal
                continue

        # Start streaming (this call blocks until stream ends/fails)
        processor.start_streaming(audio_capture)

        # If start_streaming returns (e.g., due to error), loop will check connection
        if not shutdown_flag.is_set():
             logger.warning("Streaming ended unexpectedly. Will attempt to reconnect...")
             # Optional: Add a small delay before attempting reconnection
             shutdown_flag.wait(2)

    # Cleanup
    logger.info("Initiating cleanup...")
    audio_capture.stop()
    processor.close()
    logger.info("Sender finished gracefully.")

if __name__ == "__main__":
    main() 