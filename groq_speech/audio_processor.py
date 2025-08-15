"""
Optimized audio processing for Groq Speech services.

This module provides advanced audio processing capabilities optimized for
speech recognition. It implements Voice Activity Detection (VAD), efficient
audio buffering, and preprocessing to ensure optimal API performance.

ARCHITECTURE OVERVIEW:
1. VOICE ACTIVITY DETECTION (VAD)
   - Real-time speech detection with configurable thresholds
   - Hysteresis-based state management for stability
   - Energy-based analysis with historical averaging
   - Configurable sensitivity and duration parameters

2. OPTIMIZED AUDIO PROCESSING
   - Efficient audio chunking and buffering
   - Real-time preprocessing pipeline
   - Noise reduction and audio normalization
   - Performance monitoring and optimization

3. AUDIO CHUNKING SYSTEM
   - Sliding window chunking for large files
   - Overlap management for seamless processing
   - Crossfade merging for continuous audio
   - Memory-efficient processing

KEY FEATURES:
- Real-time Voice Activity Detection
- Optimized audio preprocessing pipeline
- Configurable audio quality parameters
- Performance monitoring and statistics
- Memory-efficient buffering system
- Audio format conversion and optimization
- Noise reduction and normalization
- Large file chunking with overlap

USAGE EXAMPLES:
    # Voice Activity Detection
    vad = VoiceActivityDetector(sample_rate=16000)
    is_speech = vad.detect_speech(audio_frame)

    # Audio Processing
    processor = OptimizedAudioProcessor(enable_vad=True)
    processed_chunk = processor.process_audio_chunk(audio_data)

    # Audio Chunking
    chunker = AudioChunker(chunk_duration=30.0, overlap_duration=2.0)
    chunks = chunker.chunk_audio(audio_data)
"""

import numpy as np
import io
import time
from typing import Optional, List
from collections import deque
import soundfile as sf  # type: ignore


class VoiceActivityDetector:
    """
    Voice Activity Detection for real-time audio processing.

    CRITICAL: This class provides real-time speech detection that
    enables efficient audio processing by identifying when speech
    is present. It's essential for optimizing API calls and
    improving recognition accuracy.

    The VAD system uses:
    1. Energy-based analysis for speech detection
    2. Hysteresis-based state management for stability
    3. Historical averaging to reduce false positives
    4. Configurable thresholds for different environments
    5. Duration-based validation for speech segments

    VAD benefits:
    - Reduces unnecessary API calls during silence
    - Improves recognition accuracy by focusing on speech
    - Enables efficient audio buffering and processing
    - Provides real-time speech segment information
    - Optimizes resource usage and performance
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.03,
        silence_threshold: float = 0.005,  # Better sensitivity
        speech_threshold: float = 0.05,  # Better sensitivity
        silence_duration: float = 1.0,  # More tolerance
        speech_duration: float = 0.3,  # Min speech duration
    ):
        """
        Initialize VAD with configurable parameters.

        CRITICAL: This initialization sets up the VAD system with
        optimized parameters for real-time speech detection. The
        parameters are tuned for speech recognition applications.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration: Duration of each analysis frame in seconds
            silence_threshold: Energy threshold for silence detection
            speech_threshold: Energy threshold for speech detection
            silence_duration: Minimum silence duration to end speech
            speech_duration: Minimum speech duration to start speech

        Parameter optimization:
        - Frame duration: 30ms for real-time responsiveness
        - Thresholds: Balanced for sensitivity vs. stability
        - Durations: Prevent rapid state changes
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.silence_threshold = silence_threshold
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.speech_duration = speech_duration
        self.silence_frames = int(silence_duration / frame_duration)
        self.speech_frames = int(speech_duration / frame_duration)

        # State tracking for speech detection
        self.is_speech = False
        self.silence_frame_count = 0
        self.speech_frame_count = 0
        self.energy_history: deque = deque(maxlen=20)  # Better averaging
        self.speech_start_time: Optional[float] = None

    def detect_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech.

        CRITICAL: This method analyzes each audio frame to determine
        if it contains speech. It uses energy analysis and state
        management to provide stable speech detection.

        Args:
            audio_frame: Audio frame as numpy array

        Returns:
            True if speech detected, False otherwise

        Detection process:
        1. Calculate frame energy using RMS
        2. Update energy history for averaging
        3. Apply hysteresis-based state transitions
        4. Validate speech/silence durations
        5. Update speech state and timing
        """
        # Calculate frame energy using RMS for stability
        energy = np.mean(audio_frame**2)
        self.energy_history.append(energy)

        # Calculate average energy over recent frames
        # This reduces noise and provides stability
        avg_energy = np.mean(list(self.energy_history))

        # Update speech state with hysteresis for stability
        if avg_energy > self.speech_threshold:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            # Only start speech if we've had enough speech frames
            # This prevents false positives from brief sounds
            if self.speech_frame_count >= self.speech_frames:
                if not self.is_speech:
                    self.is_speech = True
                    self.speech_start_time = time.time()
        elif avg_energy < self.silence_threshold:
            self.silence_frame_count += 1
            self.speech_frame_count = 0

            # Only end speech if we've had enough silence frames
            # This prevents cutting off speech during brief pauses
            if self.silence_frame_count >= self.silence_frames:
                if self.is_speech:
                    self.is_speech = False
                    self.speech_start_time = None

        return self.is_speech

    def get_speech_duration(self) -> float:
        """
        Get duration of current speech segment.

        CRITICAL: This method provides timing information for
        speech segments, essential for processing decisions and
        user feedback.

        Returns:
            Duration of current speech segment in seconds
        """
        if self.is_speech and self.speech_start_time:
            return time.time() - self.speech_start_time
        return 0.0


class OptimizedAudioProcessor:
    """
    Optimized audio processor with efficient buffering and processing.

    CRITICAL: This class provides the core audio processing pipeline
    that optimizes audio data for speech recognition. It implements
    efficient buffering, preprocessing, and VAD integration.

    The processor optimizes:
    1. Audio buffering with configurable sizes
    2. Real-time preprocessing pipeline
    3. VAD integration for speech detection
    4. Audio quality optimization
    5. Performance monitoring and statistics

    Processing benefits:
    - Optimized audio quality for recognition
    - Efficient memory usage and buffering
    - Real-time processing capabilities
    - Performance monitoring and optimization
    - Configurable processing parameters
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.5,
        buffer_size: int = 8192,
        enable_vad: bool = True,
        enable_compression: bool = True,
        min_speech_duration: float = 0.5,  # Min speech duration
        max_speech_duration: float = 30.0,  # Max speech duration
    ):
        """
        Initialize optimized audio processor.

        CRITICAL: This initialization sets up the complete audio
        processing pipeline with optimized parameters for speech
        recognition performance.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
            buffer_size: Size of audio buffer in bytes
            enable_vad: Enable Voice Activity Detection
            enable_compression: Enable audio compression
            min_speech_duration: Minimum speech duration to process
            max_speech_duration: Maximum speech duration before processing

        System optimization:
        - Buffer sizes tuned for real-time processing
        - VAD integration for speech detection
        - Configurable processing parameters
        - Performance monitoring setup
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer_size = buffer_size
        self.enable_vad = enable_vad
        self.enable_compression = enable_compression
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

        # Initialize VAD if enabled
        self.vad = VoiceActivityDetector(sample_rate) if enable_vad else None

        # Audio buffers with improved sizing
        self.audio_buffer: deque = deque(
            maxlen=int(sample_rate * 15)
        )  # Increased to 15 seconds
        self.speech_buffer: deque = deque()
        self.is_recording = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None

        # Performance tracking for optimization
        self.processing_times: deque = deque(maxlen=100)
        self.chunk_count = 0

    def process_audio_chunk(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Process audio chunk with optimization.

        CRITICAL: This method processes incoming audio data in real-time,
        applying VAD, buffering, and preprocessing. It's the main entry
        point for audio processing.

        Args:
            audio_data: Raw audio data as bytes

        Returns:
            Processed audio chunk or None if no speech detected

        Processing pipeline:
        1. Audio data conversion and normalization
        2. Buffer management and storage
        3. VAD analysis for speech detection
        4. Speech buffer management
        5. Audio preprocessing and optimization
        6. Performance tracking and monitoring
        """
        start_time = time.time()

        # Convert bytes to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize audio data
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        # Add to main audio buffer for continuous processing
        self.audio_buffer.extend(audio_array_float)

        # VAD processing if enabled
        if self.enable_vad and self.vad:
            is_speech = self.vad.detect_speech(audio_array_float)

            if is_speech:
                # Start or continue speech recording
                if not self.is_recording:
                    self.is_recording = True
                    self.speech_start_time = time.time()
                self.last_speech_time = time.time()

                # Add to speech buffer for processing
                self.speech_buffer.extend(audio_array_float)
            else:
                # Check if we should process the speech buffer
                if self.is_recording and self.speech_start_time:
                    speech_duration = time.time() - self.speech_start_time

                    # Process if we have enough speech or if
                    # silence detected
                    if speech_duration >= self.min_speech_duration and (
                        speech_duration >= self.max_speech_duration
                        or (
                            self.last_speech_time
                            and time.time() - self.last_speech_time > 1.0
                        )
                    ):

                        # Extract speech data for processing
                        speech_data = np.array(list(self.speech_buffer))
                        self.speech_buffer.clear()
                        self.is_recording = False
                        self.speech_start_time = None

                        # Preprocess speech data for optimal recognition
                        processed_chunk = self._preprocess_chunk(speech_data)

                        # Update performance statistics
                        self.chunk_count += 1
                        self.processing_times.append(time.time() - start_time)

                        return processed_chunk

        # Track processing time for performance monitoring
        self.processing_times.append(time.time() - start_time)
        return None

    def _preprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess audio chunk for optimal API performance.

        CRITICAL: This method applies audio preprocessing to optimize
        the audio quality for speech recognition. It ensures the
        audio meets API requirements and improves recognition accuracy.

        Args:
            chunk: Audio chunk as numpy array

        Returns:
            Preprocessed audio chunk ready for API submission

        Preprocessing steps:
        1. Channel conversion (stereo to mono)
        2. Noise reduction for cleaner audio
        3. Audio normalization for consistent levels
        4. Quality optimization for recognition
        """
        # Ensure mono audio (required for speech recognition)
        if len(chunk.shape) > 1 and chunk.shape[1] > 1:
            chunk = np.mean(chunk, axis=1)

        # Apply gentle noise reduction for cleaner audio
        chunk = self._apply_noise_reduction(chunk)

        # Normalize audio levels for optimal recognition
        chunk = self._normalize_audio(chunk)

        return chunk

    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction using high-pass filter.

        CRITICAL: This method reduces low-frequency noise that can
        interfere with speech recognition while preserving speech
        quality. It uses a gentle filter to avoid artifacts.

        Args:
            audio: Input audio data

        Returns:
            Noise-reduced audio with preserved speech quality

        Filter characteristics:
        - High-pass filter for low-frequency noise removal
        - Gentle filtering to preserve speech quality
        - Configurable alpha for filter strength
        - Real-time processing capability
        """
        # Gentle high-pass filter to remove low-frequency noise
        if len(audio) > 1:
            # Reduced alpha for gentler filtering
            alpha = 0.98  # Increased from 0.95 for gentler filtering
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i - 1] + audio[i] - audio[i - 1])
            return filtered
        return audio

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels for optimal recognition.

        CRITICAL: This method ensures consistent audio levels that
        are optimal for speech recognition. It prevents audio that
        is too quiet or too loud from affecting recognition accuracy.

        Args:
            audio: Input audio data

        Returns:
            Normalized audio with optimal levels

        Normalization process:
        1. RMS calculation for level measurement
        2. Gain adjustment to target levels
        3. Conservative limiting to prevent distortion
        4. Clipping protection for signal integrity
        """
        if len(audio) == 0:
            return audio

        # Calculate RMS for level measurement
        rms = np.sqrt(np.mean(audio**2))

        if rms > 0:
            # Normalize to target RMS with gentler approach
            target_rms = 0.15  # Increased from 0.1 for
            # better volume
            gain = target_rms / rms
            # Apply gain with more conservative
            # limiting
            gain = min(gain, 5.0)  # Reduced from 10.0 for less
            # aggressive amplification
            audio = audio * gain

        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def create_audio_buffer(self, audio_data: bytes) -> io.BytesIO:
        """
        Create optimized audio buffer for API transmission.

        CRITICAL: This method prepares audio data for API submission
        by converting it to the optimal format and applying preprocessing.
        It ensures the audio meets all API requirements.

        Args:
            audio_data: Raw audio data as bytes

        Returns:
            Audio buffer ready for API transmission in WAV format

        Buffer preparation:
        1. Audio data conversion and preprocessing
        2. Format optimization for API requirements
        3. WAV file creation with proper parameters
        4. Buffer preparation for transmission
        """
        # Convert to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        # Apply preprocessing for optimal quality
        processed_audio = self._preprocess_chunk(audio_array_float)

        # Convert back to 16-bit PCM for API compatibility
        processed_audio = (processed_audio * 32767).astype(np.int16)

        # Create WAV buffer for API transmission
        buffer = io.BytesIO()
        sf.write(buffer, processed_audio, self.sample_rate, format="WAV")
        buffer.seek(0)

        return buffer

    def get_performance_stats(self) -> dict:
        """
        Get comprehensive performance statistics.

        CRITICAL: This method provides detailed performance metrics
        that enable optimization and monitoring of the audio processing
        pipeline. It's essential for performance tuning and debugging.

        Returns:
            Dictionary with comprehensive performance metrics

        Performance metrics include:
        - Processing time statistics (avg, min, max)
        - Buffer utilization and management
        - Chunk processing counts
        - System state information
        - Memory usage and efficiency
        """
        if not self.processing_times:
            return {
                "avg_processing_time": 0.0,
                "total_chunks": 0,
                "buffer_size": len(self.audio_buffer),
            }

        return {
            "avg_processing_time": np.mean(list(self.processing_times)),
            "max_processing_time": np.max(list(self.processing_times)),
            "min_processing_time": np.min(list(self.processing_times)),
            "total_chunks": self.chunk_count,
            "buffer_size": len(self.audio_buffer),
            "speech_buffer_size": len(self.speech_buffer),
            "is_recording": self.is_recording,
        }

    def clear_buffers(self):
        """
        Clear all audio buffers and reset state.

        CRITICAL: This method provides clean state management
        for the audio processor. It's essential for memory
        management and system stability.

        Cleanup process:
        1. Clear audio buffers to free memory
        2. Reset processing state and counters
        3. Clear performance tracking data
        4. Prepare for fresh audio processing
        """
        self.audio_buffer.clear()
        self.speech_buffer.clear()
        self.processing_times.clear()
        self.chunk_count = 0
        self.is_recording = False
        self.speech_start_time = None
        self.last_speech_time = None


class AudioChunker:
    """
    Efficient audio chunking for large files.

    CRITICAL: This class provides efficient audio chunking for
    large audio files that need to be processed in segments.
    It implements sliding window chunking with overlap for
    seamless processing.

    Chunking benefits:
    1. Memory-efficient processing of large files
    2. Overlap management for seamless results
    3. Configurable chunk sizes for optimization
    4. Crossfade merging for continuous audio
    5. Support for various file sizes and formats

    Use cases:
    - Large audio file processing
    - Streaming audio applications
    - Batch processing optimization
    - Memory-constrained environments
    """

    def __init__(
        self,
        chunk_duration: float = 30.0,
        overlap_duration: float = 2.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize audio chunker with configurable parameters.

        CRITICAL: This initialization sets up the chunking system
        with optimized parameters for seamless audio processing.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            sample_rate: Audio sample rate in Hz

        Parameter optimization:
        - Chunk duration: 30s for optimal API processing
        - Overlap duration: 2s for seamless transitions
        - Sample rate: Configurable for different audio sources
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate

        # Calculate chunk sizes in samples
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.step_size = self.chunk_size - self.overlap_size

    def chunk_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        Chunk audio data into overlapping segments.

        CRITICAL: This method splits large audio files into
        manageable chunks with overlap for seamless processing.
        It ensures no audio data is lost during chunking.

        Args:
            audio_data: Complete audio data as numpy array

        Returns:
            List of audio chunks with overlap

        Chunking process:
        1. Sliding window chunking with overlap
        2. Final chunk padding for consistency
        3. Overlap management for seamless processing
        4. Memory-efficient chunk generation
        """
        chunks = []
        start = 0

        # Create overlapping chunks using sliding window
        while start + self.chunk_size <= len(audio_data):
            chunk = audio_data[start : start + self.chunk_size]
            chunks.append(chunk)
            start += self.step_size

        # Add final chunk if there's remaining audio
        if start < len(audio_data):
            final_chunk = audio_data[start:]
            # Pad with zeros if necessary for consistency
            if len(final_chunk) < self.chunk_size:
                padding = np.zeros(self.chunk_size - len(final_chunk))
                final_chunk = np.concatenate([final_chunk, padding])
            chunks.append(final_chunk)

        return chunks

    def merge_chunks(self, chunks: List[np.ndarray]) -> np.ndarray:
        """
        Merge overlapping chunks back into continuous audio.

        CRITICAL: This method reconstructs continuous audio from
        overlapping chunks using crossfade techniques. It ensures
        seamless audio reconstruction without artifacts.

        Args:
            chunks: List of audio chunks with overlap

        Returns:
            Merged audio data as continuous numpy array

        Merging process:
        1. Overlap detection and management
        2. Crossfade application for smooth transitions
        3. Non-overlapping section handling
        4. Continuous audio reconstruction
        """
        if not chunks:
            return np.array([])

        # Calculate total length for merged audio
        total_length = (len(chunks) - 1) * self.step_size + self.chunk_size
        merged = np.zeros(total_length)

        for i, chunk in enumerate(chunks):
            start = i * self.step_size
            end = start + self.chunk_size

            # Crossfade in overlap region for smooth transitions
            if i > 0:
                overlap_start = start
                overlap_end = start + self.overlap_size

                # Linear crossfade for smooth transitions
                fade_in = np.linspace(0, 1, self.overlap_size)
                fade_out = np.linspace(1, 0, self.overlap_size)

                # Apply crossfade in overlap region
                merged[overlap_start:overlap_end] = (
                    merged[overlap_start:overlap_end] * fade_out
                    + chunk[: self.overlap_size] * fade_in
                )

                # Add non-overlapping part of the chunk
                merged[overlap_end:end] = chunk[self.overlap_size :]
            else:
                # First chunk has no overlap
                merged[start:end] = chunk

        return merged
