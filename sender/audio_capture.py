"""
Audio capture module for recording audio from microphone.
Handles device selection, configuration, and streaming.
"""

import numpy as np
import sounddevice as sd
import threading
import logging
import queue
import time
from utils.audio_utils import int16_to_float32, normalize_audio, SoundDeviceStreamWrapper
from model.model_config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE
import pyaudio

# Setup logging
logger = logging.getLogger(__name__)

# Default audio settings (now 16kHz for feature extractors)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_DURATION_MS = 100 # Process 100ms chunks
DEFAULT_CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * DEFAULT_CHUNK_DURATION_MS / 1000) # 1600 samples
# Use a larger blocksize for sounddevice if needed for stability, but process in smaller chunks
# Let's keep processing chunk size small (DEFAULT_CHUNK_SIZE)
# Sounddevice blocksize might need tuning, start with a multiple maybe
SOUNDDEVICE_BLOCKSIZE = DEFAULT_CHUNK_SIZE * 2 # e.g., 3200 samples (200ms buffer for device)

class AudioCapture:
    """
    Captures audio from a microphone using sounddevice and manages processing.
    Provides audio frames to a consumer via a callback or a blocking read method.
    """
    
    def __init__(self, device_index=None, sample_rate=DEFAULT_SAMPLE_RATE, 
                 channels=DEFAULT_CHANNELS, chunk_size=DEFAULT_CHUNK_SIZE, 
                 normalize=True, callback=None, monitor_url=None):
        """
        Initialize audio capture with the specified settings.
        
        Args:
            device_index (int): Index of input device to use or None for default
            sample_rate (int): Sample rate in Hz
            channels (int): Number of channels (1=mono, 2=stereo)
            chunk_size (int): Number of samples per frame
            normalize (bool): Whether to normalize audio to [-1, 1] range
            callback (callable): Optional function to call with audio frames
            monitor_url (str): URL for monitoring the audio capture
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.callback = callback
        self.monitor_url = monitor_url
        
        # For callback-based capture
        self.audio_queue = queue.Queue()
        self.callback_mode = callback is not None
        self.stream = None
        self.running = False
        
        # Get device information
        if device_index is not None:
            try:
                self.device_info = sd.query_devices(device_index, 'input')
                device_name = self.device_info['name']
            except:
                logging.warning(f"Device index {device_index} not found, using default input")
                self.device_index = None
                device_name = "default"
        else:
            device_name = "default"
        
        logging.info(f"Audio capture initialized: device={device_name}, "
                    f"rate={sample_rate}Hz, channels={channels}, "
                    f"chunk_size={chunk_size} samples")
                    
        self.stream = SoundDeviceStreamWrapper(
            device=device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=SOUNDDEVICE_BLOCKSIZE, # Use potentially larger blocksize for device I/O
            callback=self._audio_callback
        )
        self.process_chunk_size = chunk_size # Store the desired processing chunk size
        self.buffer = np.zeros(0, dtype=np.float32) # Internal buffer to hold audio data
        
    def start(self):
        """
        Start audio capture.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Validate audio device
            if self.device_index is not None:
                try:
                    device_info = sd.query_devices(self.device_index)
                    if device_info is None:
                        logging.warning(f"Device index {self.device_index} not found, using default input")
                        self.device_index = None
                    elif device_info['max_input_channels'] == 0:
                        logging.warning(f"Device {self.device_index} is not an input device, using default input")
                        self.device_index = None
                except Exception as e:
                    logging.warning(f"Error checking device {self.device_index}: {e}, using default input")
                    self.device_index = None
                    
            # Open audio stream
            self.stream.start()
            
            # Start processing thread only if a callback is provided
            self.running = True
            if self.callback is not None:
                self.process_thread = threading.Thread(target=self._process_audio)
                self.process_thread.start()
            
            logging.info("Audio capture started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting audio capture: {e}")
            return False
            
    def stop(self):
        """Stop the audio stream and clean up resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.running = False
            logging.info("Audio capture stream stopped")
            
    def read_chunk(self, block=True, timeout=1.0):
        """
        Read one chunk of audio data from the stream.
        
        Args:
            block (bool): Whether to block until data is available
            timeout (float): Timeout for blocking in seconds
            
        Returns:
            numpy.ndarray: Audio frame data (float32, [-1, 1]) or None if no data
        """
        if self.callback_mode:
            # Get from queue in callback mode
            try:
                return self.audio_queue.get(block=block, timeout=timeout)
            except queue.Empty:
                return None
        else:
            # Direct capture in non-callback mode
            try:
                if not self.running:
                    # Open a temporary stream for this capture
                    audio = sd.rec(
                        self.chunk_size, 
                        samplerate=self.sample_rate,
                        channels=self.channels, 
                        dtype='int16',
                        device=self.device_index,
                        blocking=block
                    )
                    if block:
                        sd.wait()
                    else:
                        # If non-blocking, check if we got any data
                        if audio is None or len(audio) == 0:
                            return None
                            
                    # Convert to float32 [-1, 1]
                    audio = int16_to_float32(audio)
                    
                    # If multi-channel, take only the first channel
                    if audio.ndim > 1 and audio.shape[1] > 1:
                        audio = audio[:, 0]
                    
                    # Normalize if requested
                    if self.normalize:
                        audio = normalize_audio(audio)
                        
                    return audio
                else:
                    logging.warning("Cannot use read_chunk when stream is running in callback mode")
                    return None
            except Exception as e:
                logging.error(f"Error capturing audio: {e}")
                return None
                
    def is_running(self):
        """Check if the audio capture is currently running."""
        return self.running
        
    def get_queue_size(self):
        """Get the current size of the audio queue."""
        return self.audio_queue.qsize() if self.callback_mode else 0 

    def read(self, timeout=1.0):
        """
        Read the next available audio chunk from the queue (blocking).
        Returns a numpy array of shape (chunk_size,) or None on timeout/error.
        Data is float32 in range [-1, 1].
        """
        try:
            # Get chunk from queue
            audio_chunk = self.audio_queue.get(timeout=timeout)

            # --- Validation (moved from _process_audio for direct read use) ---
            if audio_chunk is None:
                # logging.warning("Read None from queue.") # Can be noisy
                return None
            if not isinstance(audio_chunk, np.ndarray):
                 logging.warning(f"Read non-numpy data from queue: {type(audio_chunk)}")
                 return None
            if audio_chunk.size == 0:
                # logging.warning("Read empty audio chunk.") # Can be noisy
                return None
            # Check for excessive zeros (optional, can indicate silence or issues)
            # if np.all(audio_chunk == 0):
            #     logging.warning("Read audio chunk containing only zeros.")
            #     return None # Or return zeros if silence is valid

            # --- Normalization --- (should be done by sounddevice, but clip just in case)
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

            return audio_chunk

        except queue.Empty:
            # logging.debug("Audio queue empty, read timeout.") # Normal timeout
            return None
        except Exception as e:
            logging.error(f"Error reading audio chunk: {e}", exc_info=True)
            return None

    def _audio_callback(self, indata, frames, time, status):
        """
        Raw audio callback from sounddevice. Appends data to internal buffer.
        Input data is numpy array (frames, channels).
        """
        if status:
            logging.warning(f"Audio callback status: {status}")

        # Ensure data is float32 and flatten if mono
        audio_data = indata.astype(np.float32)
        if self.channels == 1:
            audio_data = audio_data.flatten()

        # Append to buffer
        self.buffer = np.concatenate((self.buffer, audio_data))

        # Process in fixed chunks
        while len(self.buffer) >= self.process_chunk_size:
            # Extract a chunk
            chunk = self.buffer[:self.process_chunk_size]
            # Remove chunk from buffer start
            self.buffer = self.buffer[self.process_chunk_size:]

            # Put the chunk into the queue for processing
            try:
                # The queue stores the chunks ready for read() or _process_audio
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                logging.warning("Audio processing queue is full, dropping frame.")

    def _process_audio(self):
        """Process audio data from the queue."""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Debug: Log audio data statistics
                if len(audio_data) > 0:
                    logger.debug(f"Processed audio: min={audio_data.min()}, max={audio_data.max()}, "
                               f"mean={audio_data.mean():.2f}, std={audio_data.std():.2f}")
                
                # Validate audio data
                if audio_data is None or len(audio_data) == 0 or np.all(audio_data == 0):
                    logger.warning("Received invalid audio data")
                    continue
                    
                # Normalize audio
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
                # Call callback if provided
                if self.callback:
                    self.callback(audio_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                continue 