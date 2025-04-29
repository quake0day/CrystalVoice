"""
Decoder module that reconstructs audio from neural codec tokens.
Uses the EnCodec model to convert encoded tokens back into high-quality audio.
"""

import torch
import logging
import numpy as np
from model.model_config import get_model, TARGET_BITRATE
from utils.audio_utils import from_tensor

class Decoder:
    """
    Handles decoding of encoded tokens to reconstruct audio frames.
    """
    
    def __init__(self, device="cpu", offline=False):
        """
        Initialize the decoder with the specified model.
        
        Args:
            device (str): Device to run inference on ('cpu', 'cuda', 'mps')
            offline (bool): Whether to use locally cached model only
        """
        self.device = device
        self.model = get_model(device=device, offline=offline)
        self.model.eval()  # Set to evaluation mode
        
        # Log decoder configuration
        logging.info(f"Decoder initialized on device: {self.device}")
        logging.info(f"Target bitrate: {TARGET_BITRATE} kbps")
        
    def decode(self, encoded_bytes):
        """
        Decode tokens back into an audio frame.
        
        Args:
            encoded_bytes: Serialized representation of encoded tokens
            
        Returns:
            numpy.ndarray: Reconstructed audio samples (float32, [-1, 1])
        """
        try:
            # Deserialize the tokens
            codes_list = self._deserialize_codes(encoded_bytes)
            
            if not codes_list:
                # Empty or invalid data
                return np.zeros(0, dtype=np.float32)
            
            # Stack codebooks into single tensor of shape (1, K, T)
            codes_tensors = [torch.tensor(codes, dtype=torch.long, device=self.device) for codes in codes_list]
            codes_tensor = torch.cat(codes_tensors, dim=0).unsqueeze(0)  # (1, K, T)
            # Wrap into frame tuple expected by EncodecModel: (codes, scale)
            encoded_frames = [(codes_tensor, None)]
            # Reconstruct the audio
            with torch.no_grad():
                audio_tensor = self.model.decode(encoded_frames)
            
            # Convert to numpy array
            audio = from_tensor(audio_tensor)
            
            return audio
            
        except Exception as e:
            logging.error(f"Error decoding audio frame: {e}")
            # Return empty array in case of error
            return np.zeros(0, dtype=np.float32)
            
    def _deserialize_codes(self, encoded_bytes):
        """
        Deserialize bytes back into encoded tokens.
        
        Args:
            encoded_bytes: Serialized representation from encoder
            
        Returns:
            list: List of codebook outputs (numpy arrays)
        """
        if not encoded_bytes:
            return []
            
        try:
            # Create a byte pointer to track position in the byte array
            pos = 0
            
            # Get number of codebooks
            num_codebooks = int.from_bytes(encoded_bytes[pos:pos+1], byteorder='big')
            pos += 1
            
            codes_list = []
            
            # For each codebook
            for _ in range(num_codebooks):
                # Get time dimension
                time_len = int.from_bytes(encoded_bytes[pos:pos+2], byteorder='big')
                pos += 2
                
                # Calculate bytes needed for this codebook's codes
                # 2 bytes per code (uint16)
                bytes_needed = time_len * 2
                
                # Get codes and reshape
                codes_bytes = encoded_bytes[pos:pos+bytes_needed]
                pos += bytes_needed
                
                # Convert bytes back to numpy array
                codes = np.frombuffer(codes_bytes, dtype=np.uint16).astype(np.int64)
                
                # Reshape to (1, time) for model input
                # 1 is batch dimension, which is always 1 for streaming
                codes = codes.reshape(1, time_len)
                
                codes_list.append(codes)
                
            return codes_list
            
        except Exception as e:
            logging.error(f"Error deserializing encoded tokens: {e}")
            return []
        
    def set_bitrate(self, bitrate_kbps):
        """
        Change the target bitrate.
        
        Args:
            bitrate_kbps (float): Target bitrate in kbps
        """
        self.model.set_target_bandwidth(bitrate_kbps)
        logging.info(f"Decoder bitrate changed to {bitrate_kbps} kbps") 