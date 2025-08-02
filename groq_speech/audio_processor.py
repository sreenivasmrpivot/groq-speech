"""
Optimized audio processing for Groq Speech services.
Implements efficient audio processing with Voice Activity Detection (VAD)
and optimized buffering for real-time performance.
"""

import numpy as np
import io
import time
import threading
from typing import Optional, List, Tuple, Callable
from collections import deque
import soundfile as sf
from .config import Config


class VoiceActivityDetector:
    """Voice Activity Detection for real-time audio processing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.03,
        silence_threshold: float = 0.01,
        speech_threshold: float = 0.1,
        silence_duration: float = 0.5,
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate
            frame_duration: Duration of each frame in seconds
            silence_threshold: Energy threshold for silence detection
            speech_threshold: Energy threshold for speech detection
            silence_duration: Minimum silence duration to trigger end of speech
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.silence_threshold = silence_threshold
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.silence_frames = int(silence_duration / frame_duration)

        # State tracking
        self.is_speech = False
        self.silence_frame_count = 0
        self.energy_history = deque(maxlen=10)

    def detect_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech.

        Args:
            audio_frame: Audio frame as numpy array

        Returns:
            True if speech detected, False otherwise
        """
        # Calculate frame energy
        energy = np.mean(audio_frame**2)
        self.energy_history.append(energy)

        # Calculate average energy over recent frames
        avg_energy = np.mean(list(self.energy_history))

        # Update speech state
        if avg_energy > self.speech_threshold:
            self.is_speech = True
            self.silence_frame_count = 0
        elif avg_energy < self.silence_threshold:
            self.silence_frame_count += 1
            if self.silence_frame_count >= self.silence_frames:
                self.is_speech = False

        return self.is_speech


class OptimizedAudioProcessor:
    """
    Optimized audio processor with efficient buffering and processing.
    Implements O(1) audio chunking and O(n) audio preprocessing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.5,
        buffer_size: int = 8192,
        enable_vad: bool = True,
        enable_compression: bool = True,
    ):
        """
        Initialize optimized audio processor.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_duration: Duration of each audio chunk in seconds
            buffer_size: Size of audio buffer in bytes
            enable_vad: Enable Voice Activity Detection
            enable_compression: Enable audio compression
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer_size = buffer_size
        self.enable_vad = enable_vad
        self.enable_compression = enable_compression

        # Initialize VAD if enabled
        self.vad = VoiceActivityDetector(sample_rate) if enable_vad else None

        # Audio buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * 10))  # 10 seconds buffer
        self.speech_buffer = deque()
        self.is_recording = False

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.chunk_count = 0

    def process_audio_chunk(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Process audio chunk with optimization.

        Args:
            audio_data: Raw audio data as bytes

        Returns:
            Processed audio chunk or None if no speech detected
        """
        start_time = time.time()

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

        # Add to buffer
        self.audio_buffer.extend(audio_array)

        # VAD processing if enabled
        if self.enable_vad and self.vad:
            if not self.vad.detect_speech(audio_array):
                # No speech detected, return None
                self.processing_times.append(time.time() - start_time)
                return None

        # Extract chunk from buffer
        if len(self.audio_buffer) >= self.chunk_size:
            chunk = np.array(list(self.audio_buffer)[: self.chunk_size])
            # Remove processed data from buffer
            for _ in range(self.chunk_size):
                self.audio_buffer.popleft()

            # Preprocess chunk
            processed_chunk = self._preprocess_chunk(chunk)

            self.chunk_count += 1
            self.processing_times.append(time.time() - start_time)

            return processed_chunk

        self.processing_times.append(time.time() - start_time)
        return None

    def _preprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess audio chunk for optimal API performance.

        Args:
            chunk: Audio chunk as numpy array

        Returns:
            Preprocessed audio chunk
        """
        # Ensure mono
        if len(chunk.shape) > 1 and chunk.shape[1] > 1:
            chunk = np.mean(chunk, axis=1)

        # Apply noise reduction (simple high-pass filter)
        chunk = self._apply_noise_reduction(chunk)

        # Normalize audio levels
        chunk = self._normalize_audio(chunk)

        return chunk

    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply simple noise reduction using high-pass filter.

        Args:
            audio: Input audio

        Returns:
            Noise-reduced audio
        """
        # Simple high-pass filter to remove low-frequency noise
        # This is a basic implementation - in production, use scipy.signal
        if len(audio) > 1:
            # Simple first-order high-pass filter
            alpha = 0.95
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i - 1] + audio[i] - audio[i - 1])
            return filtered
        return audio

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels for optimal recognition.

        Args:
            audio: Input audio

        Returns:
            Normalized audio
        """
        if len(audio) == 0:
            return audio

        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))

        if rms > 0:
            # Normalize to target RMS
            target_rms = 0.1
            gain = target_rms / rms
            # Apply gain with limiting
            gain = min(gain, 10.0)  # Prevent excessive amplification
            audio = audio * gain

        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def create_audio_buffer(self, audio_data: bytes) -> io.BytesIO:
        """
        Create optimized audio buffer for API transmission.

        Args:
            audio_data: Raw audio data

        Returns:
            Audio buffer ready for API transmission
        """
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        # Preprocess
        processed_audio = self._preprocess_chunk(audio_array)

        # Convert back to 16-bit PCM
        processed_audio = (processed_audio * 32767).astype(np.int16)

        # Create buffer
        buffer = io.BytesIO()
        sf.write(buffer, processed_audio, self.sample_rate, format="WAV")
        buffer.seek(0)

        return buffer

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {
                "avg_processing_time": 0.0,
                "total_chunks": 0,
                "buffer_size": len(self.audio_buffer),
            }

        return {
            "avg_processing_time": np.mean(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "total_chunks": self.chunk_count,
            "buffer_size": len(self.audio_buffer),
            "speech_buffer_size": len(self.speech_buffer),
        }

    def clear_buffers(self):
        """Clear all audio buffers."""
        self.audio_buffer.clear()
        self.speech_buffer.clear()
        self.processing_times.clear()
        self.chunk_count = 0


class AudioChunker:
    """
    Efficient audio chunking for large files.
    Implements sliding window chunking with overlap for seamless processing.
    """

    def __init__(
        self,
        chunk_duration: float = 30.0,
        overlap_duration: float = 2.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize audio chunker.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            sample_rate: Audio sample rate
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate

        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.step_size = self.chunk_size - self.overlap_size

    def chunk_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        Chunk audio data into overlapping segments.

        Args:
            audio_data: Complete audio data

        Returns:
            List of audio chunks
        """
        chunks = []
        start = 0

        while start + self.chunk_size <= len(audio_data):
            chunk = audio_data[start : start + self.chunk_size]
            chunks.append(chunk)
            start += self.step_size

        # Add final chunk if there's remaining audio
        if start < len(audio_data):
            final_chunk = audio_data[start:]
            # Pad with zeros if necessary
            if len(final_chunk) < self.chunk_size:
                padding = np.zeros(self.chunk_size - len(final_chunk))
                final_chunk = np.concatenate([final_chunk, padding])
            chunks.append(final_chunk)

        return chunks

    def merge_chunks(self, chunks: List[np.ndarray]) -> np.ndarray:
        """
        Merge overlapping chunks back into continuous audio.

        Args:
            chunks: List of audio chunks

        Returns:
            Merged audio data
        """
        if not chunks:
            return np.array([])

        # Calculate total length
        total_length = (len(chunks) - 1) * self.step_size + self.chunk_size
        merged = np.zeros(total_length)

        for i, chunk in enumerate(chunks):
            start = i * self.step_size
            end = start + self.chunk_size

            # Crossfade in overlap region
            if i > 0:
                overlap_start = start
                overlap_end = start + self.overlap_size

                # Linear crossfade
                fade_in = np.linspace(0, 1, self.overlap_size)
                fade_out = np.linspace(1, 0, self.overlap_size)

                merged[overlap_start:overlap_end] = (
                    merged[overlap_start:overlap_end] * fade_out
                    + chunk[: self.overlap_size] * fade_in
                )

                # Add non-overlapping part
                merged[overlap_end:end] = chunk[self.overlap_size :]
            else:
                merged[start:end] = chunk

        return merged
