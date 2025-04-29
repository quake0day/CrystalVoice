"""
Encoder module that compresses audio into neural codec tokens.
Uses the EnCodec model to encode audio frames into a compact representation.
"""

import torch
import logging
import numpy as np
from utils.audio_utils import to_tensor, from_tensor
from model.model_config import get_model, TARGET_BITRATE, NUM_CODEBOOKS, SAMPLE_RATE, CHANNELS, CHUNK_SIZE

class Encoder:
    """
    Handles encoding of audio frames using the neural codec model.
    """
    
    def __init__(self, device="cpu", offline=False):
        """
        Initialize the encoder with the specified model.
        
        Args:
            device (str): Device to run inference on ('cpu', 'cuda', 'mps')
            offline (bool): Whether to use locally cached model only
        """
        self.device = device
        self.model = get_model(device=device, offline=offline)
        self.model.eval()  # Set to evaluation mode
        
        # Log encoder configuration
        logging.info(f"Encoder initialized on device: {self.device}")
        logging.info(f"Target bitrate: {TARGET_BITRATE} kbps")
        
    def encode(self, audio_frame):
        """
        Encode a single audio frame into tokens.
        
        Args:
            audio_frame: Numpy array of shape (samples,) containing audio samples
                         in the range [-1, 1] with sample rate matching the model
        
        Returns:
            bytes: Serialized representation of encoded tokens, or None if encoding failed
        """
        if audio_frame is None:
            logging.error("Cannot encode None audio frame")
            return b''
            
        try:
            # Convert audio to tensor and ensure shape is correct
            audio_tensor = to_tensor(audio_frame, device=self.device)
            
            # Encode the audio frame
            with torch.no_grad():
                encoded_frames = self.model.encode(audio_tensor)
                
            # Serialize the first frame (streaming)
            encoded_bytes = self._serialize_codes(encoded_frames)
            
            return encoded_bytes
            
        except Exception as e:
            logging.error(f"Error encoding audio frame: {e}")
            return b''
            
    def _serialize_codes(self, encoded_frames):
        """
        Serialize encoded tokens to bytes for transmission.
        
        Args:
            encoded_frames: EncodecModel output (list of tensors)
            
        Returns:
            bytes: Serialized representation
        """
        try:
            # EnCodec returns a list of frames; each frame is `(codes, scale)`
            if len(encoded_frames) == 0:
                return b''
            codes_tensor, _ = encoded_frames[0]  # take first frame
            # codes_tensor: shape (B, K, T)
            codes_tensor = codes_tensor[0]  # remove batch dim -> (K, T)
            # Limit to NUM_CODEBOOKS
            codes_tensor = codes_tensor[:NUM_CODEBOOKS]
            num_codebooks, time_len = codes_tensor.shape
            data_parts = []
            # First byte: number of codebooks (<=255)
            data_parts.append(num_codebooks.to_bytes(1, byteorder='big'))
            for k in range(num_codebooks):
                codes_np = codes_tensor[k].cpu().numpy().astype(np.uint16)
                # add length (2 bytes)
                data_parts.append(time_len.to_bytes(2, byteorder='big'))
                data_parts.append(codes_np.tobytes())
            return b''.join(data_parts)
        except Exception as e:
            logging.error(f"Error serializing encoded tokens: {e}")
            return b''
        
    def set_bitrate(self, bitrate_kbps):
        """
        Change the target bitrate.
        
        Args:
            bitrate_kbps (float): Target bitrate in kbps
        """
        self.model.set_target_bandwidth(bitrate_kbps)
        logging.info(f"Encoder bitrate changed to {bitrate_kbps} kbps") 