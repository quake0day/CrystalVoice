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

def normalize_audio(audio):
    """
    Normalize audio to range [-1, 1].
    
    Args:
        audio: numpy array of audio samples
        
    Returns:
        Normalized audio array
    """
    # Avoid division by zero
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio

def int16_to_float32(audio):
    """
    Convert int16 PCM audio to float32 in range [-1, 1].
    
    Args:
        audio: numpy array of int16 audio samples
        
    Returns:
        float32, range [-1, 1] audio array
    """
    return audio.astype(np.float32) / 32768.0

def float32_to_int16(audio):
    """
    Convert float32 audio to int16 PCM format.
    
    Args:
        audio: numpy array of float32 audio, range [-1, 1]
        
    Returns:
        int16 PCM audio array
    """
    # Clip to the valid range for int16
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

def list_audio_devices():
    """
    List all available audio devices.
    
    Returns:
        A formatted string listing all audio devices
    """
    devices = sd.query_devices()
    result = "Available audio devices:\n"
    
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
            
        channels = f"(in:{device['max_input_channels']}, out:{device['max_output_channels']})"
        result += f"{i}: {device['name']} {channels} {', '.join(device_type)}\n"
    
    return result

def measure_latency(mark_time, reference="capture"):
    """
    Measure and log latency between different stages.
    
    Args:
        mark_time: timestamp of the event (in ms)
        reference: reference point ("capture", "encode", "decode", "playback")
        
    Returns:
        Latency in ms
    """
    now = get_timestamp_ms()
    latency = now - mark_time
    logging.debug(f"Latency from {reference} to current: {latency}ms")
    return latency 