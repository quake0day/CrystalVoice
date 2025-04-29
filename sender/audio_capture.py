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
from utils.audio_utils import int16_to_float32, normalize_audio
from model.model_config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE
import pyaudio

# Setup logging
logger = logging.getLogger(__name__)

class AudioCapture:
    """
    Handles audio capture from microphone.
    Provides both callback and blocking interfaces.
    """
    
    def __init__(self, device_index=None, sample_rate=SAMPLE_RATE, 
                 channels=CHANNELS, chunk_size=CHUNK_SIZE, 
                 normalize=True, callback=None):
        """
        Initialize audio capture with the specified settings.
        
        Args:
            device_index (int): Index of input device to use or None for default
            sample_rate (int): Sample rate in Hz
            channels (int): Number of channels (1=mono, 2=stereo)
            chunk_size (int): Number of samples per frame
            normalize (bool): Whether to normalize audio to [-1, 1] range
            callback (callable): Optional function to call with audio frames
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.callback = callback
        
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
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_index,
                channels=self.channels,
                dtype='float32',
                callback=self._audio_callback
            )
            
            # Start the stream
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
            
    def stop_stream(self):
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

    def read(self):
        """
        Read the next audio chunk from the queue.
        
        Returns:
            numpy array of audio samples (float32, [-1, 1]) or None if no data
        """
        try:
            audio_data = self.audio_queue.get(timeout=0.1)
            
            # Debug: Log audio data statistics
            if len(audio_data) > 0:
                logger.debug(f"Read audio: min={audio_data.min()}, max={audio_data.max()}, "
                           f"mean={audio_data.mean():.2f}, std={audio_data.std():.2f}")
            
            # Validate audio data
            if audio_data is None or len(audio_data) == 0 or np.all(audio_data == 0):
                logger.warning("Received invalid audio data")
                return None
                
            # Normalize audio
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            return audio_data
            
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error reading audio data: {e}")
            return None

    def _audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio capture.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio capture status: {status}")
            
        try:
            # Debug: Log audio data statistics
            if len(indata) > 0:
                logger.debug(f"Audio data: min={indata.min()}, max={indata.max()}, "
                           f"mean={indata.mean():.2f}, std={indata.std():.2f}")
            
            # If multi-channel, take only the first channel
            if indata.ndim > 1 and indata.shape[1] > 1:
                audio = indata[:, 0]
            else:
                audio = indata.flatten()
                
            # Normalize if requested
            if self.normalize:
                audio = normalize_audio(audio)
                
            # Add to queue
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                logger.warning("Audio queue full, dropping frame")
                
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            
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