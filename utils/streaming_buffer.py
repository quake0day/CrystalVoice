"""
Streaming buffer implementation for jitter handling and smooth audio playback.
"""

import queue
import threading
import time
import logging
import numpy as np
from collections import deque

class StreamingBuffer:
    """
    A thread-safe buffer for real-time audio streaming.
    Handles buffering and provides jitter tolerance.
    """
    
    def __init__(self, buffer_size=3, max_size=20):
        """
        Initialize a streaming buffer.
        
        Args:
            buffer_size: Number of frames to buffer before starting playback.
                         Higher values increase latency but improve stability.
            max_size: Maximum number of frames to buffer before dropping.
        """
        self.buffer = queue.Queue(maxsize=max_size)
        self.buffer_size = buffer_size
        self.max_size = max_size
        self.started = False
        self.sequence_tracker = SequenceTracker()
        self.lock = threading.Lock()
        
    def put(self, frame, sequence_number=None):
        """
        Add a frame to the buffer.
        
        Args:
            frame: Audio frame data
            sequence_number: Optional sequence number for tracking
            
        Returns:
            bool: True if successfully added, False if buffer is full
        """
        try:
            if self.buffer.qsize() >= self.max_size:
                # If buffer is full, discard oldest frame
                try:
                    self.buffer.get_nowait()
                    logging.warning("Buffer full, dropping oldest frame")
                except queue.Empty:
                    pass
            
            # Add the frame with sequence number if provided
            if sequence_number is not None:
                self.sequence_tracker.track(sequence_number)
                self.buffer.put((frame, sequence_number), block=False)
            else:
                self.buffer.put((frame, None), block=False)
                
            if not self.started and self.buffer.qsize() >= self.buffer_size:
                with self.lock:
                    self.started = True
                    logging.info(f"Buffer filled with {self.buffer.qsize()} frames, playback starting")
            
            return True
        except queue.Full:
            logging.warning("Buffer full, frame dropped")
            return False
    
    def get(self, block=True, timeout=None):
        """
        Get a frame from the buffer.
        
        Args:
            block: Whether to block until a frame is available
            timeout: Timeout in seconds for blocking
            
        Returns:
            frame: The audio frame, or None if buffer is empty
        """
        with self.lock:
            if not self.started:
                return None  # Not enough frames buffered yet
        
        try:
            frame, sequence_number = self.buffer.get(block=block, timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def get_with_sequence(self, block=True, timeout=None):
        """
        Get a frame from the buffer along with its sequence number.
        
        Args:
            block: Whether to block until a frame is available
            timeout: Timeout in seconds for blocking
            
        Returns:
            tuple: (frame, sequence_number) or (None, None) if buffer is empty
        """
        with self.lock:
            if not self.started:
                return None, None  # Not enough frames buffered yet
        
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            return None, None
    
    def clear(self):
        """Clear all data from the buffer and reset state."""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
            self.started = False
            self.sequence_tracker.reset()
        
    def size(self):
        """Get the current number of items in the buffer."""
        return self.buffer.qsize()
    
    def is_active(self):
        """Check if the buffer is active (has buffered enough frames)."""
        with self.lock:
            return self.started
    
    def get_stats(self):
        """Get buffer statistics."""
        return {
            'buffer_size': self.buffer.qsize(),
            'buffer_capacity': self.max_size,
            'buffer_active': self.is_active(),
            'packet_loss': self.sequence_tracker.get_loss_rate(),
            'out_of_order': self.sequence_tracker.get_out_of_order_count()
        }


class SequenceTracker:
    """
    Tracks sequence numbers to detect packet loss and out-of-order delivery.
    Used for network diagnostics and monitoring.
    """
    
    def __init__(self, window_size=100):
        """
        Initialize sequence number tracker.
        
        Args:
            window_size: Size of the sliding window for statistics
        """
        self.last_seq = None
        self.missing = 0
        self.total = 0
        self.out_of_order = 0
        self.window_size = window_size
        self.sequence_history = deque(maxlen=window_size)
        
    def track(self, seq_num):
        """
        Track a new sequence number.
        
        Args:
            seq_num: Sequence number to track
        """
        if seq_num is None:
            return
            
        if self.last_seq is not None:
            if seq_num == self.last_seq + 1:
                # Normal case - sequence is as expected
                pass
            elif seq_num <= self.last_seq:
                # Out of order packet
                self.out_of_order += 1
                logging.debug(f"Out of order packet: expected > {self.last_seq}, got {seq_num}")
            else:
                # Missing packets
                gap = seq_num - self.last_seq - 1
                self.missing += gap
                logging.debug(f"Detected {gap} missing packets between {self.last_seq} and {seq_num}")
        
        self.sequence_history.append(seq_num)
        self.last_seq = seq_num
        self.total += 1
    
    def reset(self):
        """Reset all counters."""
        self.last_seq = None
        self.missing = 0
        self.total = 0
        self.out_of_order = 0
        self.sequence_history.clear()
    
    def get_loss_rate(self):
        """Get the packet loss rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.missing / (self.total + self.missing)) * 100
    
    def get_out_of_order_count(self):
        """Get the number of out-of-order packets."""
        return self.out_of_order 