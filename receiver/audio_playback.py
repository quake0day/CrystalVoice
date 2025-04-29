"""
Audio playback module for outputting reconstructed audio to speakers.
Handles device selection, configuration, and streaming.
"""

import numpy as np
import sounddevice as sd
import threading
import logging
import queue
import time
from utils.audio_utils import float32_to_int16
from model.model_config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE

class AudioPlayback:
    """
    Handles audio playback to speakers.
    Provides both callback and direct interfaces.
    """
    
    def __init__(self, device_index=None, sample_rate=SAMPLE_RATE,
                 channels=CHANNELS, chunk_size=CHUNK_SIZE, 
                 buffer_size=3):
        """
        Initialize audio playback with the specified settings.
        
        Args:
            device_index (int): Index of output device to use or None for default
            sample_rate (int): Sample rate in Hz
            channels (int): Number of channels (1=mono, 2=stereo)
            chunk_size (int): Number of samples per frame
            buffer_size (int): Size of internal playback buffer (in frames)
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        # For callback-based playback
        self.audio_queue = queue.Queue(maxsize=buffer_size * 2)
        self.callback_mode = False
        self.stream = None
        self.running = False
        
        # Get device information
        if device_index is not None:
            try:
                self.device_info = sd.query_devices(device_index, 'output')
                device_name = self.device_info['name']
            except:
                logging.warning(f"Device index {device_index} not found, using default output")
                self.device_index = None
                device_name = "default"
        else:
            device_name = "default"
        
        logging.info(f"Audio playback initialized: device={device_name}, "
                     f"rate={sample_rate}Hz, channels={channels}, "
                     f"chunk_size={chunk_size} samples")
    
    def start_stream(self):
        """
        Start streaming audio to speakers using callback.
        Audio frames must be added to the audio_queue.
        """
        if self.running:
            logging.warning("Audio playback already running")
            return
            
        # Buffer for silence in case of underflow
        silence = np.zeros(self.chunk_size, dtype=np.float32)
        
        def audio_callback(outdata, frames, time_info, status):
            """Callback for sounddevice stream."""
            if status:
                logging.warning(f"Audio playback status: {status}")
                
            try:
                # Get audio from queue, with timeout to prevent blocking
                audio = self.audio_queue.get(timeout=0.1)
                
                # Make sure we have enough samples
                if len(audio) < frames:
                    # Pad with zeros if we don't have enough samples
                    audio = np.pad(audio, (0, frames - len(audio)))
                    logging.debug(f"Padded audio frame from {len(audio)} to {frames} samples")
                elif len(audio) > frames:
                    # Truncate if we have too many samples
                    audio = audio[:frames]
                    logging.debug(f"Truncated audio frame from {len(audio)} to {frames} samples")
                
                # Reshape for multi-channel if needed
                if self.channels > 1 and audio.ndim == 1:
                    # Duplicate mono to all channels
                    audio = np.tile(audio.reshape(-1, 1), (1, self.channels))
                
                # Copy to output buffer
                outdata[:] = audio.reshape(outdata.shape)
                
            except queue.Empty:
                # Queue underflow - play silence
                outdata[:] = silence.reshape(outdata.shape) if outdata.ndim == 1 else np.tile(silence.reshape(-1, 1), (1, outdata.shape[1]))
                logging.debug("Audio queue underflow, playing silence")
                
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_index,
                channels=self.channels,
                dtype='float32',
                callback=audio_callback
            )
            
            self.stream.start()
            self.running = True
            self.callback_mode = True
            logging.info("Audio playback stream started")
            
        except Exception as e:
            logging.error(f"Error starting audio playback: {e}")
            raise
            
    def stop_stream(self):
        """Stop the audio stream and clean up resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.running = False
            logging.info("Audio playback stream stopped")
            
    def play_chunk(self, audio_chunk):
        """
        Play a single chunk of audio.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, range [-1, 1])
            
        Returns:
            bool: True if successfully played/queued, False otherwise
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False
            
        if self.callback_mode:
            # Queue for callback mode
            try:
                self.audio_queue.put(audio_chunk, block=False)
                return True
            except queue.Full:
                logging.warning("Audio playback queue full, dropping frame")
                return False
        else:
            # Direct playback mode
            try:
                # Reshape for multi-channel if needed
                if self.channels > 1 and audio_chunk.ndim == 1:
                    # Duplicate mono to all channels
                    audio_chunk = np.tile(audio_chunk.reshape(-1, 1), (1, self.channels))
                
                # Directly play the audio
                sd.play(audio_chunk, self.sample_rate)
                return True
            except Exception as e:
                logging.error(f"Error playing audio: {e}")
                return False
                
    def wait(self):
        """Wait for all queued audio to finish playing."""
        if self.callback_mode:
            # Wait for queue to empty in callback mode
            while not self.audio_queue.empty() and self.running:
                time.sleep(0.1)
        else:
            # Wait for direct playback to finish
            sd.wait()
            
    def is_running(self):
        """Check if the audio playback is currently running."""
        return self.running
        
    def get_queue_size(self):
        """Get the current size of the audio queue."""
        return self.audio_queue.qsize() if self.callback_mode else 0 