"""
Audio configuration for Groq Speech services.
"""

import pyaudio
import wave
import soundfile as sf
from typing import Optional, Union
import numpy as np
from .config import Config


class AudioConfig:
    """
    Configuration class for audio input/output in Groq Speech services.
    Supports microphone input, file input, and custom audio streams.
    """
    
    def __init__(self, 
                 filename: Optional[str] = None,
                 device_id: Optional[int] = None,
                 sample_rate: int = None,
                 channels: int = None,
                 format_type: str = "wav"):
        """
        Initialize AudioConfig with audio settings.
        
        Args:
            filename: Path to audio file for file-based input
            device_id: Microphone device ID for microphone input
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            format_type: Audio format type ('wav', 'mp3', etc.)
        """
        self.filename = filename
        self.device_id = device_id or Config.get_device_index()
        self.sample_rate = sample_rate or Config.DEFAULT_SAMPLE_RATE
        self.channels = channels or Config.DEFAULT_CHANNELS
        self.format_type = format_type
        
        # PyAudio instance for microphone input
        self._pyaudio = None
        self._stream = None
        
        # Audio file properties
        self._audio_data = None
        self._audio_length = 0
        
        if filename:
            self._load_audio_file()
    
    def _load_audio_file(self):
        """Load audio file data."""
        try:
            # Load audio file using soundfile
            data, sample_rate = sf.read(self.filename)
            
            # Convert to mono if stereo
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                # Simple resampling (in production, use librosa or scipy)
                ratio = self.sample_rate / sample_rate
                new_length = int(len(data) * ratio)
                data = np.interp(np.linspace(0, len(data), new_length), 
                               np.arange(len(data)), data)
            
            self._audio_data = data
            self._audio_length = len(data)
            self.sample_rate = self.sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to load audio file {self.filename}: {str(e)}")
    
    def get_audio_devices(self) -> list:
        """
        Get list of available audio input devices.
        
        Returns:
            List of device dictionaries with id, name, and channels
        """
        if not self._pyaudio:
            self._pyaudio = pyaudio.PyAudio()
        
        devices = []
        for i in range(self._pyaudio.get_device_count()):
            device_info = self._pyaudio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Input device
                devices.append({
                    'id': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
        
        return devices
    
    def start_microphone_stream(self):
        """
        Start microphone audio stream.
        
        Returns:
            PyAudio stream object
        """
        if not self._pyaudio:
            self._pyaudio = pyaudio.PyAudio()
        
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_id,
            frames_per_buffer=1024
        )
        
        return self._stream
    
    def stop_microphone_stream(self):
        """Stop microphone audio stream."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    def read_audio_chunk(self, chunk_size: int = 1024) -> bytes:
        """
        Read a chunk of audio data from microphone.
        
        Args:
            chunk_size: Size of audio chunk to read
            
        Returns:
            Audio data as bytes
        """
        if not self._stream:
            raise RuntimeError("Microphone stream not started")
        
        return self._stream.read(chunk_size)
    
    def get_file_audio_data(self) -> np.ndarray:
        """
        Get audio data from file.
        
        Returns:
            Audio data as numpy array
        """
        if self._audio_data is None:
            raise RuntimeError("No audio file loaded")
        
        return self._audio_data
    
    def get_file_audio_length(self) -> int:
        """
        Get length of audio file in samples.
        
        Returns:
            Number of audio samples
        """
        return self._audio_length
    
    def create_audio_chunks(self, chunk_duration: float = 0.5) -> list:
        """
        Create audio chunks from file for streaming.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks as numpy arrays
        """
        if self._audio_data is None:
            raise RuntimeError("No audio file loaded")
        
        chunk_size = int(self.sample_rate * chunk_duration)
        chunks = []
        
        for i in range(0, len(self._audio_data), chunk_size):
            chunk = self._audio_data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def save_audio_file(self, filename: str, audio_data: np.ndarray):
        """
        Save audio data to file.
        
        Args:
            filename: Output filename
            audio_data: Audio data as numpy array
        """
        sf.write(filename, audio_data, self.sample_rate)
    
    def get_audio_format_info(self) -> dict:
        """
        Get audio format information.
        
        Returns:
            Dictionary with audio format details
        """
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'format': self.format_type,
            'filename': self.filename,
            'device_id': self.device_id
        }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.filename:  # Only start microphone if not using file
            self.start_microphone_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_microphone_stream() 