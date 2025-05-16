"""
Audio Processor Model
Responsible for audio acquisition and processing.
Follows Single Responsibility Principle by focusing only on audio processing.
"""

import pyaudio
import numpy as np
import threading
import time
from collections import deque

class IAudioProcessor:
    """Interface for AudioProcessor following Interface Segregation Principle"""
    def start(self):
        """Start audio recording and processing"""
        pass
        
    def stop(self):
        """Stop audio recording and processing"""
        pass
        
    def is_running(self):
        """Check if the processor is running"""
        pass

class AudioProcessor(IAudioProcessor):
    """
    Audio processor implementation for watch tick detection
    Follows Single Responsibility Principle by focusing on audio capture and preprocessing
    """
    def __init__(self, callback=None, buffer_size=5):
        """
        Initialize audio processor for watch tick detection
        
        Args:
            callback: Function to call with processed audio chunks
            buffer_size: Size of the buffer to store recent audio data (in seconds)
        """
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100  # High sample rate for precision
        self.CHUNK = 1024  # Audio chunk size
        
        self.callback = callback
        self.running = False
        self.thread = None
        self.p = None
        self.stream = None
        
        # Buffer to store recent audio for analysis
        self.buffer_size = int(buffer_size * self.RATE / self.CHUNK)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
    
    def start(self):
        """Start audio recording and processing"""
        if self.running:
            return
        
        self.running = True
        
        # Open audio stream
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop audio recording and processing"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def is_running(self):
        """Check if the processor is running"""
        return self.running
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        # Convert byte data to numpy array of float32
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.append(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio data in a separate thread"""
        while self.running:
            if len(self.audio_buffer) > 0:
                # Get current audio data from buffer for processing
                buffer_data = np.concatenate(list(self.audio_buffer))
                
                # Process the audio data and send to callback
                if self.callback:
                    self.callback(buffer_data, self.RATE)
            
            # Sleep to prevent excessive CPU usage
            time.sleep(0.05)
    
    def __del__(self):
        """Cleanup resources"""
        self.stop()
        
        if self.p:
            self.p.terminate()
            self.p = None