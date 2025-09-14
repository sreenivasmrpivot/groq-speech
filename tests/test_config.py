#!/usr/bin/env python3
"""
Test Configuration and Utilities

This module provides shared configuration and utilities for all tests.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_AUDIO_FILES = {
    "test1": "examples/test1.wav",
    "test2": "examples/test2.wav", 
    "test3": "examples/test3.mp3",
    "test4": "examples/test4.mp3"
}

# Test environment variables
TEST_ENV_VARS = {
    'GROQ_API_KEY': 'test_groq_api_key_here',
    'HF_TOKEN': 'test_hf_token_here',
    'GROQ_MODEL_ID': 'whisper-large-v3',
    'GROQ_TEMPERATURE': '0.0',
    'AUDIO_SAMPLE_RATE': '16000',
    'AUDIO_CHANNELS': '1',
    'MAX_AUDIO_FILE_SIZE': '25000000',
    'DIARIZATION_MIN_SEGMENT_DURATION': '2.0',
    'DIARIZATION_SILENCE_THRESHOLD': '0.8',
    'DIARIZATION_MAX_SEGMENTS_PER_CHUNK': '8',
    'DIARIZATION_CHUNK_STRATEGY': 'adaptive',
    'DIARIZATION_MAX_SPEAKERS': '5'
}

# Mock results for testing
MOCK_RECOGNITION_RESULT = MagicMock(
    text="This is a test transcription result",
    reason=MagicMock(),
    confidence=0.95,
    language="en-US",
    timestamps=[]
)

MOCK_DIARIZATION_RESULT = MagicMock(
    text="This is a test transcription with diarization",
    num_speakers=2,
    segments=[
        MagicMock(speaker_id="SPEAKER_00", text="Speaker 0 text"),
        MagicMock(speaker_id="SPEAKER_01", text="Speaker 1 text")
    ]
)

MOCK_TRANSLATION_RESULT = MagicMock(
    text="This is a test translation result",
    reason=MagicMock(),
    confidence=0.90,
    language="en",
    timestamps=[]
)

def create_test_audio_file(file_path, duration=3.0, sample_rate=16000, frequency=440):
    """Create a test audio file for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(file_path, audio_data, sample_rate)
    return file_path

def ensure_test_audio_files():
    """Ensure all test audio files exist."""
    for name, file_path in TEST_AUDIO_FILES.items():
        path = Path(file_path)
        if not path.exists():
            print(f"Creating test audio file: {file_path}")
            create_test_audio_file(path)

def setup_test_environment():
    """Set up test environment variables."""
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

def cleanup_test_environment():
    """Clean up test environment variables."""
    for key in TEST_ENV_VARS.keys():
        if key in os.environ:
            del os.environ[key]

def mock_speech_recognizer():
    """Create a mock SpeechRecognizer for testing."""
    mock_recognizer = MagicMock()
    mock_recognizer.recognize_file.return_value = MOCK_RECOGNITION_RESULT
    mock_recognizer.translate_file.return_value = MOCK_TRANSLATION_RESULT
    mock_recognizer.recognize_audio_data.return_value = MOCK_RECOGNITION_RESULT
    mock_recognizer.translate_audio_data.return_value = MOCK_TRANSLATION_RESULT
    return mock_recognizer

def mock_diarization_result():
    """Create a mock diarization result for testing."""
    return MOCK_DIARIZATION_RESULT

def get_test_audio_path(name="test1"):
    """Get the path to a test audio file."""
    return Path(TEST_AUDIO_FILES.get(name, TEST_AUDIO_FILES["test1"]))

# Test decorators
def requires_api_key(test_func):
    """Decorator to mark tests that require API key."""
    def wrapper(*args, **kwargs):
        if not os.getenv('GROQ_API_KEY'):
            return unittest.skip("Requires GROQ_API_KEY")(test_func)(*args, **kwargs)
        return test_func(*args, **kwargs)
    return wrapper

def requires_hf_token(test_func):
    """Decorator to mark tests that require HF token."""
    def wrapper(*args, **kwargs):
        if not os.getenv('HF_TOKEN'):
            return unittest.skip("Requires HF_TOKEN")(test_func)(*args, **kwargs)
        return test_func(*args, **kwargs)
    return wrapper

def mock_microphone_input(test_func):
    """Decorator to mock microphone input for testing."""
    def wrapper(*args, **kwargs):
        with patch('examples.speech_demo.process_microphone_single') as mock_single, \
             patch('examples.speech_demo.process_microphone_continuous') as mock_continuous:
            mock_single.return_value = MOCK_RECOGNITION_RESULT
            mock_continuous.return_value = None
            return test_func(*args, **kwargs)
    return wrapper
