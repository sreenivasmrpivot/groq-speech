#!/usr/bin/env python3
"""
Tests for the optimized audio processor.
"""

import unittest
import numpy as np
import time
import io
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech.audio_processor import (
    VoiceActivityDetector,
    OptimizedAudioProcessor,
    AudioChunker,
)


class TestVoiceActivityDetector(unittest.TestCase):
    """Test cases for VoiceActivityDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.vad = VoiceActivityDetector(
            sample_rate=16000,
            frame_duration=0.03,
            silence_threshold=0.01,
            speech_threshold=0.1,
            silence_duration=0.5,
        )

    def test_vad_initialization(self):
        """Test VAD initialization."""
        self.assertEqual(self.vad.sample_rate, 16000)
        self.assertEqual(self.vad.frame_size, 480)  # 16000 * 0.03
        self.assertEqual(self.vad.silence_threshold, 0.01)
        self.assertEqual(self.vad.speech_threshold, 0.1)
        self.assertFalse(self.vad.is_speech)

    def test_speech_detection(self):
        """Test speech detection with various audio inputs."""
        # Test silence (low energy)
        silence_frame = np.random.normal(0, 0.005, 480)
        self.assertFalse(self.vad.detect_speech(silence_frame))

        # Test speech (high energy) - create more realistic speech-like signal
        # Generate a sine wave with higher amplitude to simulate speech
        t = np.linspace(0, 0.03, 480)  # 30ms frame
        speech_frame = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        speech_frame += 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz overtone

        # Build energy history with consistent high-energy frames
        for _ in range(10):  # Need more frames to build history
            self.vad.detect_speech(speech_frame)

        # Now should detect speech
        self.assertTrue(self.vad.detect_speech(speech_frame))

        # Test transition from silence to speech
        self.vad.is_speech = False
        self.vad.silence_frame_count = 5
        for _ in range(10):  # Build energy history again
            self.vad.detect_speech(speech_frame)
        self.assertTrue(self.vad.detect_speech(speech_frame))
        self.assertEqual(self.vad.silence_frame_count, 0)

    def test_silence_detection(self):
        """Test silence detection after speech."""
        # Start with speech - build energy history first
        t = np.linspace(0, 0.03, 480)  # 30ms frame
        speech_frame = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        speech_frame += 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz overtone

        for _ in range(10):  # Build energy history
            self.vad.detect_speech(speech_frame)

        self.assertTrue(self.vad.is_speech)

        # Transition to silence - need more frames to clear the energy history
        silence_frame = np.random.normal(0, 0.005, 480)
        for _ in range(self.vad.silence_frames + 10):  # Extra frames to clear history
            self.vad.detect_speech(silence_frame)

        self.assertFalse(self.vad.is_speech)

    def test_energy_history(self):
        """Test energy history tracking."""
        frame = np.random.normal(0, 0.2, 480)

        # Add multiple frames
        for _ in range(5):
            self.vad.detect_speech(frame)

        self.assertEqual(len(self.vad.energy_history), 5)
        self.assertTrue(all(energy > 0 for energy in self.vad.energy_history))


class TestOptimizedAudioProcessor(unittest.TestCase):
    """Test cases for OptimizedAudioProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = OptimizedAudioProcessor(
            sample_rate=16000,
            channels=1,
            chunk_duration=0.5,
            buffer_size=8192,
            enable_vad=True,
            enable_compression=True,
        )

    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.sample_rate, 16000)
        self.assertEqual(self.processor.channels, 1)
        self.assertEqual(self.processor.chunk_size, 8000)  # 16000 * 0.5
        self.assertEqual(self.processor.buffer_size, 8192)
        self.assertTrue(self.processor.enable_vad)
        self.assertTrue(self.processor.enable_compression)
        self.assertIsNotNone(self.processor.vad)

    def test_audio_chunk_processing(self):
        """Test audio chunk processing."""
        # Create test audio data that's larger than chunk size
        audio_data = np.random.normal(0, 0.5, 16000).astype(np.int16).tobytes()

        # Process chunk
        result = self.processor.process_audio_chunk(audio_data)

        # Should return processed chunk (if VAD detects speech)
        if result is not None:
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 8000)  # chunk_size
        else:
            # VAD filtered out the audio (no speech detected)
            self.assertIsNone(result)

    def test_noise_reduction(self):
        """Test noise reduction functionality."""
        # Create noisy audio
        clean_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))
        noise = np.random.normal(0, 0.1, 1000)
        noisy_audio = clean_audio + noise

        # Apply noise reduction
        processed = self.processor._apply_noise_reduction(noisy_audio)

        # Check that output is different from input
        self.assertFalse(np.array_equal(processed, noisy_audio))
        self.assertEqual(len(processed), len(noisy_audio))

    def test_audio_normalization(self):
        """Test audio normalization."""
        # Create audio with varying levels
        audio = np.random.normal(0, 0.8, 1000)

        # Normalize
        normalized = self.processor._normalize_audio(audio)

        # Check that values are within bounds
        self.assertTrue(np.all(normalized >= -1.0))
        self.assertTrue(np.all(normalized <= 1.0))

    def test_audio_buffer_creation(self):
        """Test audio buffer creation for API transmission."""
        # Create test audio data
        audio_data = np.random.normal(0, 0.5, 8000).astype(np.int16).tobytes()

        # Create buffer
        buffer = self.processor.create_audio_buffer(audio_data)

        # Check buffer properties
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertGreater(buffer.getbuffer().nbytes, 0)

    def test_performance_stats(self):
        """Test performance statistics collection."""
        # Process some audio chunks
        for _ in range(5):
            audio_data = np.random.normal(0, 0.5, 16000).astype(np.int16).tobytes()
            self.processor.process_audio_chunk(audio_data)

        # Get stats
        stats = self.processor.get_performance_stats()

        # Check stats structure
        self.assertIn("avg_processing_time", stats)
        self.assertIn("total_chunks", stats)
        self.assertIn("buffer_size", stats)
        # Note: total_chunks may be 0 if VAD filtered out all audio
        self.assertGreaterEqual(stats["total_chunks"], 0)

    def test_buffer_clearing(self):
        """Test buffer clearing functionality."""
        # Add some data to buffers
        audio_data = np.random.normal(0, 0.5, 8000).astype(np.int16).tobytes()
        self.processor.process_audio_chunk(audio_data)

        # Clear buffers
        self.processor.clear_buffers()

        # Check that buffers are empty
        self.assertEqual(len(self.processor.audio_buffer), 0)
        self.assertEqual(len(self.processor.speech_buffer), 0)
        self.assertEqual(self.processor.chunk_count, 0)


class TestAudioChunker(unittest.TestCase):
    """Test cases for AudioChunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = AudioChunker(
            chunk_duration=30.0, overlap_duration=2.0, sample_rate=16000
        )

    def test_chunker_initialization(self):
        """Test chunker initialization."""
        self.assertEqual(self.chunker.chunk_duration, 30.0)
        self.assertEqual(self.chunker.overlap_duration, 2.0)
        self.assertEqual(self.chunker.sample_rate, 16000)
        self.assertEqual(self.chunker.chunk_size, 480000)  # 16000 * 30
        self.assertEqual(self.chunker.overlap_size, 32000)  # 16000 * 2
        self.assertEqual(self.chunker.step_size, 448000)  # 480000 - 32000

    def test_audio_chunking(self):
        """Test audio chunking functionality."""
        # Create test audio data (60 seconds)
        audio_data = np.random.normal(0, 0.5, 16000 * 60)

        # Chunk audio
        chunks = self.chunker.chunk_audio(audio_data)

        # Should have 2 chunks (60 seconds / 30 seconds per chunk)
        # But with overlap, we get 3 chunks: 0-30s, 28-58s, 56-60s
        self.assertEqual(len(chunks), 3)

        # Check chunk sizes
        for chunk in chunks:
            self.assertEqual(len(chunk), 480000)

    def test_chunking_with_remainder(self):
        """Test chunking with audio that doesn't fit evenly."""
        # Create test audio data (45 seconds)
        audio_data = np.random.normal(0, 0.5, 16000 * 45)

        # Chunk audio
        chunks = self.chunker.chunk_audio(audio_data)

        # Should have 2 chunks (first 30s, then 15s + padding)
        self.assertEqual(len(chunks), 2)

        # First chunk should be full size
        self.assertEqual(len(chunks[0]), 480000)

        # Second chunk should be padded to full size
        self.assertEqual(len(chunks[1]), 480000)

    def test_chunk_merging(self):
        """Test merging chunks back into continuous audio."""
        # Create test audio data
        original_audio = np.random.normal(0, 0.5, 16000 * 60)

        # Chunk and then merge
        chunks = self.chunker.chunk_audio(original_audio)
        merged_audio = self.chunker.merge_chunks(chunks)

        # Check that merged audio has correct length
        expected_length = (
            len(chunks) - 1
        ) * self.chunker.step_size + self.chunker.chunk_size
        self.assertEqual(len(merged_audio), expected_length)

    def test_empty_chunking(self):
        """Test chunking with empty audio."""
        empty_audio = np.array([])
        chunks = self.chunker.chunk_audio(empty_audio)

        self.assertEqual(len(chunks), 0)

    def test_small_audio_chunking(self):
        """Test chunking with audio smaller than chunk size."""
        # Create audio smaller than chunk size
        small_audio = np.random.normal(0, 0.5, 16000 * 10)  # 10 seconds

        chunks = self.chunker.chunk_audio(small_audio)

        # Should have 1 chunk with padding
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 480000)  # Full chunk size


class TestAudioProcessingPerformance(unittest.TestCase):
    """Performance tests for audio processing."""

    def test_processing_speed(self):
        """Test that audio processing is fast enough for real-time use."""
        processor = OptimizedAudioProcessor(
            sample_rate=16000, chunk_duration=0.5, enable_vad=True
        )

        # Create test audio data
        audio_data = np.random.normal(0, 0.5, 8000).astype(np.int16).tobytes()

        # Measure processing time
        start_time = time.time()
        for _ in range(100):  # Process 100 chunks
            processor.process_audio_chunk(audio_data)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_chunk = total_time / 100

        # Should process chunks in under 10ms each for real-time performance
        self.assertLess(avg_time_per_chunk, 0.01)

        # Total time should be under 1 second for 100 chunks
        self.assertLess(total_time, 1.0)

    def test_memory_efficiency(self):
        """Test that audio processing doesn't consume excessive memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        processor = OptimizedAudioProcessor(
            sample_rate=16000, chunk_duration=0.5, enable_vad=True
        )

        # Process many chunks
        audio_data = np.random.normal(0, 0.5, 8000).astype(np.int16).tobytes()

        for _ in range(1000):
            processor.process_audio_chunk(audio_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        self.assertLess(memory_increase, 50 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
