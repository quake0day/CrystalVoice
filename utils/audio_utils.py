"""
Audio utility functions for the voice communication system.
"""

import numpy as np
import sounddevice as sd
import logging
import time
from scipy import signal
import torch
from model.model_config import SAMPLE_RATE, CHANNELS

# Setup logging
logger = logging.getLogger(__name__)

def normalize_audio(audio, target_level=-20.0):
    """
    Normalize audio volume to a target dB level.
    
    Args:
        audio: numpy array of audio samples
        target_level: target dB level
        
    Returns:
        Normalized audio array
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0: return audio # Avoid division by zero for silence
    
    current_level_db = 20 * np.log10(rms)
    gain_db = target_level - current_level_db
    gain_linear = 10**(gain_db / 20.0)
    
    normalized_audio = audio * gain_linear
    
    # Clip just in case normalization pushes values beyond [-1, 1]
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    return normalized_audio

def int16_to_float32(audio):
    """
    Convert int16 PCM audio to float32 in range [-1, 1].
    
    Args:
        audio: numpy array of int16 audio samples
        
    Returns:
        float32, range [-1, 1] audio array
    """
    if audio.dtype != np.int16:
        raise ValueError("Input array must be of type int16")
    return audio.astype(np.float32) / 32768.0

def float32_to_int16(audio):
    """
    Convert float32 audio to int16 PCM format.
    
    Args:
        audio: numpy array of float32 audio, range [-1, 1]
        
    Returns:
        int16 PCM audio array
    """
    if audio.dtype != np.float32:
        raise ValueError("Input array must be of type float32")
    # Ensure values are clipped
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)

def resample_audio(audio, orig_sr, target_sr=SAMPLE_RATE):
    """
    Resample audio to target sample rate if needed.
    
    Args:
        audio: Audio samples (numpy array)
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    # Calculate number of samples in the target sample rate
    n_samples = int(len(audio) * target_sr / orig_sr)
    
    # Use scipy.signal.resample for high-quality resampling
    return signal.resample(audio, n_samples)

def to_tensor(audio, device="cpu"):
    """
    Convert numpy audio array to PyTorch tensor.
    
    Args:
        audio: numpy array of audio samples (float32, [-1, 1])
        device: torch device to put tensor on
        
    Returns:
        PyTorch tensor of shape (1, channels, samples)
    """
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Reshape: (samples,) -> (1, 1, samples) for mono
    # or (samples, channels) -> (1, channels, samples) for stereo
    if audio.ndim == 1:
        # Mono audio
        audio_tensor = torch.tensor(audio).reshape(1, 1, -1)
    else:
        # Multi-channel audio (transpose to get channels first)
        audio_tensor = torch.tensor(audio.T).reshape(1, -1, audio.shape[0])
    
    return audio_tensor.to(device)

def from_tensor(tensor):
    """
    Convert PyTorch tensor to numpy array for audio playback.
    
    Args:
        tensor: PyTorch tensor of shape (1, channels, samples)
        
    Returns:
        numpy array of audio samples (float32, [-1, 1])
    """
    # Get the audio data from tensor and convert to numpy
    # Remove batch dimension and place channels last (if multi-channel)
    if tensor.shape[1] == 1:
        # Mono audio
        audio = tensor[0, 0, :].cpu().numpy()
    else:
        # Multi-channel audio, transpose to get channels last
        audio = tensor[0].cpu().numpy().T
        
    return audio

def get_timestamp_ms():
    """Get the current timestamp in milliseconds."""
    return int(time.time() * 1000)

def measure_latency(start_time_ms, label="process"):
    """Calculate latency based on start time."""
    latency = get_timestamp_ms() - start_time_ms
    logger.debug(f"{label.capitalize()} latency: {latency} ms")
    return latency

def list_audio_devices():
    """
    List all available audio devices.
    
    Returns:
        A formatted string listing all audio devices
    """
    print("\nAvailable Audio Devices:")
    print("Index\tName\t\t\t\tType")
    print("-"*80)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        dev_type = "Input" if device['max_input_channels'] > 0 else ""
        if device['max_output_channels'] > 0:
            if dev_type:
                dev_type += "/Output"
            else:
                dev_type = "Output"
        print(f"{i}\t{device['name']:<40}\t{dev_type}")
    print("-"*80)

# --- Added SoundDeviceStreamWrapper --- 
default_stream_config = {}
class SoundDeviceStreamWrapper:
    """
    A simple wrapper around sounddevice.InputStream for a consistent interface.
    """
    def __init__(self, device=None, samplerate=None, channels=None, blocksize=None, callback=None):
        self.stream = sd.InputStream(
            device=device,
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            dtype='float32', # Assuming float32 based on audio_capture usage
            callback=callback
        )

    def start(self):
        if self.stream and not self.stream.active:
            self.stream.start()
            logger.info("SoundDevice stream started.")

    def stop(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            logger.info("SoundDevice stream stopped.")

    def close(self):
        if self.stream:
            # Ensure stream is stopped before closing
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            logger.info("SoundDevice stream closed.")
            self.stream = None # Prevent reuse

    # Optional: Add an is_active property if needed
    @property
    def active(self):
        return self.stream.active if self.stream else False
# --- End Added SoundDeviceStreamWrapper --- 