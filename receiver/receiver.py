#!/usr/bin/env python3
"""
Main receiver program for the low-bandwidth voice communication system.
Receives encoded tokens from the sender, decodes them with neural codec model,
and plays reconstructed audio through speakers.
"""

import os
import sys
import time
import logging
import argparse
import grpc
from concurrent import futures
import threading
import numpy as np
import torch
from dotenv import load_dotenv
import requests

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from proto import voice_stream_pb2
from proto import voice_stream_pb2_grpc
from receiver.audio_playback import AudioPlayback
from receiver.decoder import Decoder
from utils.streaming_buffer import StreamingBuffer
from utils.audio_utils import get_timestamp_ms, list_audio_devices, measure_latency
from model.model_config import SAMPLE_RATE, CHUNK_SIZE, TARGET_BITRATE

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("Receiver")

class VoiceStreamServicer(voice_stream_pb2_grpc.VoiceStreamServicer):
    """
    gRPC service implementation for receiving audio streams.
    """
    
    def __init__(self, receiver):
        """
        Initialize the service with a reference to the receiver.
        
        Args:
            receiver: The Receiver instance to handle audio processing
        """
        self.receiver = receiver
        
    def Transmit(self, request_iterator, context):
        """
        Handle incoming audio stream from sender.
        
        Args:
            request_iterator: Iterator for incoming AudioChunk messages
            context: gRPC context
            
        Returns:
            Ack: Acknowledgment message
        """
        client_addr = context.peer()
        logger.info(f"New audio stream from {client_addr}")
        
        try:
            # Process each incoming chunk
            chunk_count = 0
            for chunk in request_iterator:
                if not self.receiver.is_running():
                    logger.warning("Receiver stopped while streaming, closing connection")
                    break
                    
                # Process the audio chunk
                self.receiver.process_chunk(chunk)
                chunk_count += 1
                
                # Log statistics periodically
                if chunk_count % 100 == 0:
                    logger.info(f"Received {chunk_count} chunks from {client_addr}")
                
            logger.info(f"Stream from {client_addr} completed. Processed {chunk_count} chunks.")
            return voice_stream_pb2.Ack(received=True, message="Stream processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing stream from {client_addr}: {e}")
            return voice_stream_pb2.Ack(received=False, message=f"Error: {str(e)}")

class Receiver:
    """
    Main receiver class that orchestrates token reception, decoding, and playback.
    """
    
    def __init__(self, host="0.0.0.0", port=50051, speaker_device=None, 
                 sample_rate=SAMPLE_RATE, bitrate=TARGET_BITRATE,
                 buffer_frames=3, verbose=False):
        """
        Initialize the receiver.
        
        Args:
            host (str): Host to bind the server to
            port (int): Port to listen on
            speaker_device (int): Speaker device index (None for default)
            sample_rate (int): Audio sample rate in Hz
            bitrate (float): Target bitrate in kbps
            buffer_frames (int): Number of frames to buffer before playback
            verbose (bool): Enable verbose logging
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.speaker_device = speaker_device
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.buffer_frames = buffer_frames
        self.running = False
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA device for decoding")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon GPU for decoding")
        else:
            self.device = "cpu"
            logger.info("Using CPU for decoding")
            
        # Initialize components
        self.audio_playback = None
        self.decoder = None
        self.buffer = None
        self.server = None
        
        # Stats
        self.chunks_received = 0
        self.chunks_processed = 0
        self.total_bytes_received = 0
        self.last_stats_time = time.time()
        self.stats_lock = threading.Lock()
        
    def setup(self):
        """Set up audio playback, decoder, and streaming buffer."""
        try:
            # Initialize audio playback
            logger.info("Initializing audio playback...")
            self.audio_playback = AudioPlayback(
                device_index=self.speaker_device,
                sample_rate=self.sample_rate,
                chunk_size=CHUNK_SIZE,
                buffer_size=self.buffer_frames
            )
            
            # Initialize decoder
            logger.info("Initializing audio decoder...")
            self.decoder = Decoder(device=self.device)
            self.decoder.set_bitrate(self.bitrate)
            
            # Initialize buffer
            logger.info(f"Initializing streaming buffer (size={self.buffer_frames})...")
            self.buffer = StreamingBuffer(buffer_size=self.buffer_frames)
            
            logger.info("Receiver setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            self.cleanup()
            return False
            
    def start(self):
        """Start audio playback and gRPC server."""
        if self.running:
            logger.warning("Receiver is already running")
            return False
            
        if not self.audio_playback or not self.decoder or not self.buffer:
            logger.error("Receiver not properly set up. Call setup() first.")
            return False
            
        try:
            # Start audio playback
            self.audio_playback.start_stream()
            
            # Create and start gRPC server
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            voice_stream_pb2_grpc.add_VoiceStreamServicer_to_server(
                VoiceStreamServicer(self), self.server
            )
            self.server.add_insecure_port(self.address)
            self.server.start()
            
            # Start playback thread
            self.running = True
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            # Start stats thread
            self.stats_thread = threading.Thread(target=self._stats_loop)
            self.stats_thread.daemon = True
            self.stats_thread.start()
            
            logger.info(f"Receiver started, listening on {self.address}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting receiver: {e}")
            self.cleanup()
            return False
            
    def process_chunk(self, chunk):
        """
        Process an incoming audio chunk.
        
        Args:
            chunk: AudioChunk protobuf message
        """
        try:
            # Extract data from chunk
            encoded_data = chunk.encoded_frame
            timestamp_ms = chunk.timestamp_ms
            sequence_number = chunk.sequence_number
            
            # Update stats
            with self.stats_lock:
                self.chunks_received += 1
                self.total_bytes_received += len(encoded_data)
            
            # Queue for processing
            self.buffer.put(
                (encoded_data, timestamp_ms), 
                sequence_number
            )
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            
    def _playback_loop(self):
        """Main playback loop (runs in a separate thread)."""
        logger.info("Audio playback thread started")
        
        try:
            while self.running:
                # Get next chunk from buffer
                frame_data, timestamp_ms = self.buffer.get_with_sequence(block=True, timeout=0.1)
                
                if frame_data is None:
                    # No data available yet, wait a bit
                    time.sleep(0.01)
                    continue
                    
                # Unpack the frame data
                encoded_data, orig_timestamp = frame_data
                
                # Measure buffer latency
                buffer_latency = measure_latency(orig_timestamp, "network")
                
                # Decode the audio
                audio_data = self.decoder.decode(encoded_data)
                
                # If we got valid audio data
                if audio_data is not None and len(audio_data) > 0:
                    # Play the audio
                    self.audio_playback.play_chunk(audio_data)
                    
                    # Measure decode latency
                    decode_latency = measure_latency(orig_timestamp, "total")
                    
                    # Update stats
                    with self.stats_lock:
                        self.chunks_processed += 1
                        
                    # Log detailed stats occasionally
                    if self.chunks_processed % 100 == 0:
                        logger.debug(f"Played chunk {self.chunks_processed}, "
                                    f"buffer latency: {buffer_latency}ms, "
                                    f"total latency: {decode_latency}ms")
                        
        except Exception as e:
            logger.error(f"Error in playback loop: {e}")
        finally:
            logger.info("Audio playback thread ended")
            
    def _stats_loop(self):
        """Thread to periodically log statistics."""
        last_chunks = 0
        last_bytes = 0
        
        while self.running:
            try:
                time.sleep(5.0)  # Update every 5 seconds
                
                now = time.time()
                elapsed = now - self.last_stats_time
                
                with self.stats_lock:
                    chunks_diff = self.chunks_received - last_chunks
                    bytes_diff = self.total_bytes_received - last_bytes
                    
                    if chunks_diff > 0:
                        # Calculate bitrate
                        bitrate_kbps = (bytes_diff * 8 / 1000) / elapsed
                        
                        # Get buffer stats
                        buffer_stats = self.buffer.get_stats() if self.buffer else {}
                        buffer_size = buffer_stats.get('buffer_size', 0)
                        packet_loss = buffer_stats.get('packet_loss', 0)
                        
                        # Get latency
                        latency = measure_latency(0, "total")  # Get current latency
                        
                        logger.info(f"Stats: received={self.chunks_received}, "
                                   f"rate={chunks_diff/elapsed:.1f} chunks/sec, "
                                   f"bitrate={bitrate_kbps:.2f} kbps, "
                                   f"buffer={buffer_size}, "
                                   f"packet_loss={packet_loss:.1f}%")
                        
                        # Send stats to monitor if available
                        try:
                            monitor_url = os.environ.get("MONITOR_URL", "http://localhost:8765/update")
                            stats_payload = {
                                "bitrate": round(bitrate_kbps, 2),
                                "latency": round(latency),
                                "packet_loss": round(packet_loss, 1),
                                "buffer_size": buffer_size
                            }
                            requests.post(monitor_url, json=stats_payload, timeout=0.5)
                            logger.debug(f"Sent stats to monitor: {stats_payload}")
                        except Exception as e:
                            logger.debug(f"Failed to send stats to monitor: {e}")
                    
                    last_chunks = self.chunks_received
                    last_bytes = self.total_bytes_received
                    self.last_stats_time = now
                    
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")
                
    def is_running(self):
        """Check if the receiver is running."""
        return self.running
        
    def stop(self):
        """Stop the receiver and clean up resources."""
        logger.info("Stopping receiver...")
        self.running = False
        
        # Stop gRPC server
        if self.server:
            self.server.stop(grace=2.0)
            
        # Wait for threads to end
        if hasattr(self, 'playback_thread') and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
            
        if hasattr(self, 'stats_thread') and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=1.0)
            
        self.cleanup()
        logger.info("Receiver stopped")
        
    def cleanup(self):
        """Clean up resources."""
        # Stop audio playback
        if self.audio_playback:
            self.audio_playback.stop_stream()
            
        # Clear buffer
        if self.buffer:
            self.buffer.clear()
            
        self.audio_playback = None
        self.decoder = None
        self.buffer = None
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Communication Receiver")
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=50051,
                      help="Port to listen on (default: 50051)")
    parser.add_argument("--speaker", type=int, default=None,
                      help="Speaker device index (default: system default)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                      help=f"Audio sample rate in Hz (default: {SAMPLE_RATE})")
    parser.add_argument("--bitrate", type=float, default=TARGET_BITRATE,
                      help=f"Target bitrate in kbps (default: {TARGET_BITRATE})")
    parser.add_argument("--buffer", type=int, default=3,
                      help="Buffer size in frames (default: 3)")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio devices and exit")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
                      
    return parser.parse_args()
    
def main():
    """Main entry point."""
    args = parse_args()
    
    # Just list devices if requested
    if args.list_devices:
        print(list_audio_devices())
        return 0
        
    # Create and start receiver
    receiver = Receiver(
        host=args.host,
        port=args.port,
        speaker_device=args.speaker,
        sample_rate=args.sample_rate,
        bitrate=args.bitrate,
        buffer_frames=args.buffer,
        verbose=args.verbose
    )
    
    if not receiver.setup():
        logger.error("Failed to set up receiver")
        return 1
        
    if not receiver.start():
        logger.error("Failed to start receiver")
        return 1
        
    try:
        # Keep server running until user interrupts
        logger.info("Receiver running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        receiver.stop()
        
    return 0
    
if __name__ == "__main__":
    sys.exit(main()) 