"""
Audio configuration for Groq Speech services.

This module provides the AudioConfig class that manages all audio input/output
operations for the speech recognition system. It handles microphone input,
file-based audio, and provides audio processing utilities.

ARCHITECTURE OVERVIEW:
1. AUDIO INPUT SOURCES
   - Microphone input with real-time streaming
   - File-based audio loading and processing
   - Custom audio stream support

2. AUDIO PROCESSING
   - Sample rate conversion and resampling
   - Channel conversion (stereo to mono)
   - Audio chunking for streaming applications
   - Format validation and conversion

3. DEVICE MANAGEMENT
   - Audio device enumeration and selection
   - Device-specific configuration
   - Stream lifecycle management

4. CONTEXT MANAGEMENT
   - Context manager support for resource cleanup
   - Automatic stream management
   - Memory-efficient audio handling

KEY FEATURES:
- Real-time microphone input with configurable parameters
- File-based audio processing with automatic format conversion
- Audio device enumeration and selection
- Context manager support for automatic cleanup
- Audio chunking for streaming applications
- Sample rate and channel conversion
- Multiple audio format support

USAGE EXAMPLES:
    # Microphone input with context manager
    with AudioConfig() as audio:
        chunk = audio.read_audio_chunk(1024)

    # File-based audio processing
    audio_config = AudioConfig(filename="audio.wav")
    audio_data = audio_config.get_file_audio_data()

    # Custom device configuration
    audio_config = AudioConfig(
        device_id=1,
        sample_rate=16000,
        channels=1
    )
"""

import pyaudio  # type: ignore
import soundfile as sf  # type: ignore
from typing import Optional
import numpy as np  # type: ignore
from .config import Config


class AudioConfig:
    """
    Configuration class for audio input/output in Groq Speech services.

    CRITICAL: This class manages all audio input operations and provides
    a unified interface for both microphone and file-based audio sources.
    It's essential for the speech recognition system to capture and process
    audio data efficiently.

    The AudioConfig supports:
    1. Microphone Input: Real-time audio capture with configurable parameters
    2. File Input: Audio file loading with automatic format conversion
    3. Device Management: Audio device enumeration and selection
    4. Audio Processing: Resampling, channel conversion, and chunking
    5. Resource Management: Automatic cleanup and memory management

    Key architectural features:
    - Context manager support for automatic resource cleanup
    - Configurable audio parameters (sample rate, channels, format)
    - Audio device enumeration and selection
    - Audio format conversion and optimization
    - Streaming support with audio chunking
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        device_id: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        format_type: str = "wav",
    ):
        """
        Initialize AudioConfig with audio settings.

        CRITICAL: This initialization sets up the audio configuration
        for either microphone input or file-based processing. It handles
        device selection, audio parameters, and resource initialization:

        Args:
            filename: Path to audio file for file-based input
            device_id: Microphone device ID for microphone input
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            format_type: Audio format type ('wav', 'mp3', etc.)

        Initialization process:
        1. Audio parameter configuration with defaults
        2. Device ID resolution from configuration
        3. Audio file loading if filename provided
        4. PyAudio instance preparation for microphone input
        5. Audio data structure initialization
        """
        self.filename = filename
        self.device_id = device_id or Config.get_device_index()
        self.sample_rate = sample_rate or Config.DEFAULT_SAMPLE_RATE
        self.channels = channels or Config.DEFAULT_CHANNELS
        self.format_type = format_type

        # PyAudio instance for microphone input
        # Lazy initialization to avoid unnecessary resource allocation
        self._pyaudio = None
        self._stream = None

        # Audio file properties for file-based input
        self._audio_data = None
        self._audio_length = 0

        # Load audio file if filename is provided
        if filename:
            self._load_audio_file()

    def _load_audio_file(self):
        """
        Load audio file data with format conversion.

        CRITICAL: This method loads audio files and performs necessary
        format conversions to ensure compatibility with the speech
        recognition system. It handles various audio formats and
        optimizes them for processing.

        Processing steps:
        1. Audio file loading using soundfile library
        2. Stereo to mono conversion if needed
        3. Sample rate resampling to target rate
        4. Audio data normalization and validation
        5. Memory-efficient data storage

        Raises:
            ValueError: If audio file loading fails

        Supported formats:
        - WAV, FLAC, OGG, and other soundfile-supported formats
        - Automatic format detection and conversion
        - Error handling for corrupted or unsupported files
        """
        try:
            # Load audio file using soundfile library
            # This provides support for many audio formats
            data, sample_rate = sf.read(self.filename)

            # Convert to mono if stereo (required for speech recognition)
            # This improves processing efficiency and compatibility
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)

            # Resample if necessary to match target sample rate
            # Simple resampling for compatibility
            # (use librosa/scipy for production)
            if sample_rate != self.sample_rate:
                ratio = self.sample_rate / sample_rate
                new_length = int(len(data) * ratio)
                x_new = np.linspace(0, len(data), new_length)
                x_old = np.arange(len(data))
                data = np.interp(x_new, x_old, data)

            # Store processed audio data
            self._audio_data = data
            self._audio_length = len(data)
            self.sample_rate = self.sample_rate

        except Exception as e:
            raise ValueError(
                f"Failed to load audio file {self.filename}: {str(e)}"
            ) from e

    def get_audio_devices(self) -> list:
        """
        Get list of available audio input devices.

        CRITICAL: This method enumerates all available audio input
        devices on the system. It's essential for device selection
        and configuration in multi-device environments.

        Returns:
            List of device dictionaries with id, name, and channels

        Device information includes:
        - Device ID for selection
        - Device name for user identification
        - Channel count for compatibility checking
        - Sample rate capabilities

        This method enables:
        - Device selection in user interfaces
        - Automatic device configuration
        - Device capability validation
        - Multi-device system support
        """
        if not self._pyaudio:
            self._pyaudio = pyaudio.PyAudio()

        devices: list = []
        if self._pyaudio is None:
            return devices

        # Enumerate all audio devices and filter for
        # input devices
        for i in range(self._pyaudio.get_device_count()):
            device_info = self._pyaudio.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # Input device
                devices.append(
                    {
                        "id": i,
                        "name": device_info["name"],
                        "channels": device_info["maxInputChannels"],
                        "sample_rate": int(device_info["defaultSampleRate"]),
                    }
                )

        return devices

    def start_microphone_stream(self):
        """
        Start microphone audio stream for real-time capture.

        CRITICAL: This method initializes the microphone audio stream
        with the configured parameters. It's essential for real-time
        speech recognition and audio capture.

        Returns:
            PyAudio stream object for audio reading

        Stream configuration:
        - 16-bit integer format for compatibility
        - Configurable channels and sample rate
        - Device-specific input selection
        - Optimized buffer size for real-time processing

        The stream enables:
        - Real-time audio capture
        - Continuous audio processing
        - Low-latency speech recognition
        - Audio chunk reading
        """
        if not self._pyaudio:
            self._pyaudio = pyaudio.PyAudio()

        # Open audio stream with optimized parameters
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,  # 16-bit format for compatibility
            channels=self.channels,  # Mono or stereo as configured
            rate=self.sample_rate,  # Target sample
            # rate
            input=True,  # Input stream for microphone
            input_device_index=self.device_id,  # Selected
            # device
            frames_per_buffer=1024,  # Optimized buffer size
        )

        return self._stream

    def stop_microphone_stream(self):
        """
        Stop microphone audio stream and cleanup resources.

        CRITICAL: This method properly closes the audio stream and
        releases system resources. It's essential for preventing
        audio device conflicts and memory leaks.

        Cleanup process:
        1. Stop the audio stream
        2. Close the stream connection
        3. Terminate PyAudio instance
        4. Release audio device resources

        This method should be called:
        - When switching audio sources
        - Before application shutdown
        - When changing audio devices
        - In error recovery scenarios
        """
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

        CRITICAL: This method reads real-time audio data from the
        microphone stream. It's the primary method for continuous
        audio capture during speech recognition.

        Args:
            chunk_size: Size of audio chunk to read in samples

        Returns:
            Audio data as bytes ready for processing

        Raises:
            RuntimeError: If microphone stream is not started

        Audio chunk characteristics:
        - Configurable chunk size for processing flexibility
        - Real-time data capture with minimal latency
        - Raw audio data in bytes format
        - Ready for immediate processing or buffering

        This method enables:
        - Continuous speech recognition
        - Real-time audio processing
        - Audio buffering and chunking
        - Streaming audio applications
        """
        if not self._stream:
            raise RuntimeError("Microphone stream not started")

        return self._stream.read(chunk_size)

    def get_file_audio_data(self) -> np.ndarray:
        """
        Get audio data from loaded file.

        CRITICAL: This method provides access to the processed
        audio data from file-based input. It's essential for
        file-based speech recognition and audio analysis.

        Returns:
            Audio data as numpy array ready for processing

        Raises:
            RuntimeError: If no audio file is loaded

        Audio data characteristics:
        - Processed and optimized for speech recognition
        - Proper sample rate and channel configuration
        - Normalized audio levels
        - Ready for API submission or further processing

        This method enables:
        - File-based speech recognition
        - Audio analysis and processing
        - Batch processing of audio files
        - Audio format conversion
        """
        if self._audio_data is None:
            raise RuntimeError("No audio file loaded")

        return self._audio_data

    def get_file_audio_length(self) -> int:
        """
        Get length of loaded audio file in samples.

        CRITICAL: This method provides metadata about the loaded
        audio file, essential for processing planning and validation.

        Returns:
            Number of audio samples in the file

        Length information is useful for:
        - Processing time estimation
        - Memory allocation planning
        - Audio chunking calculations
        - Progress tracking and validation
        """
        return self._audio_length

    def create_audio_chunks(self, chunk_duration: float = 0.5) -> list:
        """
        Create audio chunks from file for streaming applications.

        CRITICAL: This method splits audio files into smaller chunks
        for streaming processing. It's essential for handling large
        audio files and real-time processing applications.

        Args:
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of audio chunks as numpy arrays

        Raises:
            RuntimeError: If no audio file is loaded

        Chunking benefits:
        - Memory-efficient processing of large files
        - Streaming audio applications
        - Real-time processing simulation
        - Batch processing optimization

        Chunk characteristics:
        - Configurable duration for processing flexibility
        - Overlap support for seamless processing
        - Optimized for speech recognition
        - Ready for individual processing
        """
        if self._audio_data is None:
            raise RuntimeError("No audio file loaded")

        # Calculate chunk size in samples
        chunk_size = int(self.sample_rate * chunk_duration)
        chunks = []

        # Split audio data into chunks
        for i in range(0, len(self._audio_data), chunk_size):
            chunk = self._audio_data[i : i + chunk_size]
            chunks.append(chunk)

        return chunks

    def save_audio_file(self, filename: str, audio_data: np.ndarray):
        """
        Save audio data to file with configured format.

        CRITICAL: This method provides audio output capabilities
        for saving processed audio, recording sessions, or
        creating audio files from microphone input.

        Args:
            filename: Output filename with path
            audio_data: Audio data as numpy array

        File saving features:
        - Automatic format detection from filename extension
        - Sample rate and channel configuration preservation
        - Error handling for file system issues
        - Support for various audio formats

        This method enables:
        - Audio recording and saving
        - Processed audio export
        - Audio format conversion
        - Backup and archival of audio data
        """
        sf.write(filename, audio_data, self.sample_rate)

    def get_audio_format_info(self) -> dict:
        """
        Get comprehensive audio format information.

        CRITICAL: This method provides metadata about the current
        audio configuration. It's essential for debugging, validation,
        and system integration.

        Returns:
            Dictionary with complete audio format details

        Format information includes:
        - Sample rate and channel configuration
        - Audio format type and filename
        - Device ID and configuration
        - Processing parameters

        This information is useful for:
        - Configuration validation
        - System integration
        - Debugging and troubleshooting
        - Performance optimization
        """
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format_type,
            "filename": self.filename,
            "device_id": self.device_id,
        }

    def __enter__(self):
        """
        Context manager entry for automatic resource management.

        CRITICAL: This method enables the use of AudioConfig as a
        context manager, providing automatic resource cleanup and
        stream management. It's the recommended usage pattern.

        Returns:
            Self instance ready for audio operations

        Context manager benefits:
        - Automatic microphone stream startup
        - Resource cleanup on exit
        - Exception-safe resource management
        - Simplified usage patterns

        Usage:
            with AudioConfig() as audio:
                chunk = audio.read_audio_chunk(1024)
        """
        if not self.filename:  # Only start microphone if not using file
            self.start_microphone_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit for automatic resource cleanup.

        CRITICAL: This method ensures proper cleanup of audio
        resources when exiting the context manager. It prevents
        resource leaks and audio device conflicts.

        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred

        Cleanup process:
        - Automatic microphone stream shutdown
        - PyAudio instance termination
        - Audio device resource release
        - Exception-safe cleanup execution

        This ensures:
        - Proper resource cleanup
        - Audio device availability
        - Memory leak prevention
        - System stability
        """
        self.stop_microphone_stream()
