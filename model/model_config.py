"""
Configuration for neural audio codec model loading.
Defines paths, model parameters and loading utilities.
"""

import os
import logging
from pathlib import Path

# EnCodec model configuration
MODEL_NAME = "facebook/encodec_24khz"  # 24kHz model from Meta AI
TARGET_BITRATE = 6.0  # Target bitrate in kbps, can be 1.5, 3, 6, 12, 24
SAMPLE_RATE = 24000  # Sample rate in Hz
CHANNELS = 1  # Mono audio
FRAME_RATE = 50  # EnCodec processes 20ms frames (50 frames per second)

# Frame size in samples (20ms at 24kHz = 480 samples)
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)

# Derived parameters (don't change)
CHUNK_SIZE = FRAME_SIZE  # Use frame_size as chunk size for audio I/O

# Bandwidth settings
# Number of codebooks to use (higher = more quality but higher bitrate)
# For 6 kbps, 8 codebooks is appropriate
NUM_CODEBOOKS = 8

# Local cached model path (if offline mode needed)
# Use in get_model instead of downloading if needed
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/")

# For debugging and development
VERBOSE = False

def get_model(device="cpu", offline=False):
    """
    Load the EnCodec model with specified configuration.
    
    Args:
        device (str): Device to load the model on ('cpu', 'cuda', 'mps' for Apple Silicon)
        offline (bool): Whether to use local cached model only
        
    Returns:
        EncodecModel: The loaded encoder/decoder model
    """
    try:
        # Import inside function to avoid import errors if not used
        from encodec import EncodecModel
        from encodec.utils import convert_audio
        import torch
        
        # Set logging level for huggingface hub
        if not VERBOSE:
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        # Set cache directory if offline 
        kwargs = {}
        if offline:
            kwargs["cache_dir"] = CACHE_DIR
            kwargs["local_files_only"] = True
            logging.info(f"Loading model from cache: {CACHE_DIR}")
            
        # Load the pre-trained EnCodec model
        model = EncodecModel.encodec_model_24khz(**kwargs)
        model.set_target_bandwidth(TARGET_BITRATE)
        
        # Move model to device if needed
        if device != "cpu":
            model = model.to(device)
        
        logging.info(f"Loaded EnCodec model: {MODEL_NAME}")
        logging.info(f"Target bitrate: {TARGET_BITRATE} kbps")
        logging.info(f"Using {NUM_CODEBOOKS} codebooks")
        
        return model
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise 