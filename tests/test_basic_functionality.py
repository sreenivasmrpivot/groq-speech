#!/usr/bin/env python3
"""
Basic tests for Groq Speech SDK functionality.
"""

import unittest
import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Add the parent directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import (
    SpeechConfig, 
    AudioConfig, 
    SpeechRecognizer, 
    ResultReason, 
    CancellationReason,
    PropertyId
)


class TestSpeechConfig(unittest.TestCase):
    """Test SpeechConfig functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key = os.getenv('GROQ_API_KEY', 'test-key')
    
    def test_speech_config_creation(self):
        """Test creating SpeechConfig with API key."""
        config = SpeechConfig(api_key=self.api_key)
        self.assertEqual(config.api_key, self.api_key)
        self.assertEqual(config.region, 'us-east-1')  # default region
        self.assertEqual(config.speech_recognition_language, 'en-US')  # default language
    
    def test_speech_config_with_region(self):
        """Test creating SpeechConfig with custom region."""
        config = SpeechConfig(api_key=self.api_key, region='us-west-1')
        self.assertEqual(config.region, 'us-west-1')
    
    def test_speech_config_with_language(self):
        """Test setting recognition language."""
        config = SpeechConfig(api_key=self.api_key)
        config.speech_recognition_language = 'de-DE'
        self.assertEqual(config.speech_recognition_language, 'de-DE')
    
    def test_speech_config_properties(self):
        """Test setting and getting properties."""
        config = SpeechConfig(api_key=self.api_key)
        config.set_property(PropertyId.Speech_SegmentationStrategy, "Semantic")
        self.assertEqual(config.get_property(PropertyId.Speech_SegmentationStrategy), "Semantic")
    
    def test_speech_config_validation(self):
        """Test configuration validation."""
        # Test with valid config
        config = SpeechConfig(api_key=self.api_key)
        config.validate()  # Should not raise an exception
        
        # Test with invalid config (no API key)
        config_no_key = SpeechConfig(api_key=None)
        with self.assertRaises(ValueError):
            config_no_key.validate()


class TestAudioConfig(unittest.TestCase):
    """Test AudioConfig functionality."""
    
    def test_audio_config_creation(self):
        """Test creating AudioConfig."""
        config = AudioConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.channels, 1)
        self.assertEqual(config.format_type, 'wav')
        self.assertIsNone(config.filename)
        self.assertIsNone(config.device_id)
    
    def test_audio_config_with_filename(self):
        """Test creating AudioConfig with filename."""
        config = AudioConfig(filename='test.wav')
        self.assertEqual(config.filename, 'test.wav')
    
    def test_audio_config_with_device_id(self):
        """Test creating AudioConfig with device ID."""
        config = AudioConfig(device_id=0)
        self.assertEqual(config.device_id, 0)
    
    def test_audio_config_with_custom_settings(self):
        """Test creating AudioConfig with custom settings."""
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            format_type='mp3'
        )
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.channels, 2)
        self.assertEqual(config.format_type, 'mp3')
    
    def test_audio_config_get_audio_devices(self):
        """Test getting audio devices."""
        config = AudioConfig()
        devices = config.get_audio_devices()
        self.assertIsInstance(devices, list)
        # Should have at least one input device
        self.assertGreater(len(devices), 0)
        
        # Check device structure
        if devices:
            device = devices[0]
            self.assertIn('id', device)
            self.assertIn('name', device)
            self.assertIn('channels', device)
            self.assertIn('sample_rate', device)


class TestSpeechRecognizer(unittest.TestCase):
    """Test SpeechRecognizer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key = os.getenv('GROQ_API_KEY', 'test-key')
        self.speech_config = SpeechConfig(api_key=self.api_key)
    
    def test_speech_recognizer_creation(self):
        """Test creating SpeechRecognizer."""
        recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.assertEqual(recognizer.speech_config, self.speech_config)
        self.assertIsNone(recognizer.audio_config)
    
    def test_speech_recognizer_with_audio_config(self):
        """Test creating SpeechRecognizer with AudioConfig."""
        audio_config = AudioConfig()
        recognizer = SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        self.assertEqual(recognizer.audio_config, audio_config)
    
    def test_speech_recognizer_event_handlers(self):
        """Test event handler connection."""
        recognizer = SpeechRecognizer(speech_config=self.speech_config)
        
        # Test connecting event handlers
        def test_handler(evt):
            pass
        
        recognizer.connect('recognized', test_handler)
        recognizer.connect('session_started', test_handler)
        recognizer.connect('session_stopped', test_handler)
        recognizer.connect('canceled', test_handler)
        
        # Check that handlers were added
        self.assertIn(test_handler, recognizer.recognized_handlers)
        self.assertIn(test_handler, recognizer.session_started_handlers)
        self.assertIn(test_handler, recognizer.session_stopped_handlers)
        self.assertIn(test_handler, recognizer.canceled_handlers)
    
    def test_speech_recognizer_invalid_event(self):
        """Test connecting invalid event handler."""
        recognizer = SpeechRecognizer(speech_config=self.speech_config)
        
        def test_handler(evt):
            pass
        
        with self.assertRaises(ValueError):
            recognizer.connect('invalid_event', test_handler)
    
    def test_speech_recognizer_is_recognizing(self):
        """Test is_recognizing method."""
        recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.assertFalse(recognizer.is_recognizing())


class TestResultClasses(unittest.TestCase):
    """Test result classes functionality."""
    
    def test_speech_recognition_result(self):
        """Test SpeechRecognitionResult creation."""
        from groq_speech.speech_recognizer import SpeechRecognitionResult
        
        result = SpeechRecognitionResult(
            text="Hello world",
            reason=ResultReason.RecognizedSpeech,
            confidence=0.95,
            language="en-US"
        )
        
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.reason, ResultReason.RecognizedSpeech)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.language, "en-US")
    
    def test_cancellation_details(self):
        """Test CancellationDetails creation."""
        from groq_speech.speech_recognizer import CancellationDetails
        
        details = CancellationDetails(
            reason=CancellationReason.Error,
            error_details="Test error"
        )
        
        self.assertEqual(details.reason, CancellationReason.Error)
        self.assertEqual(details.error_details, "Test error")
    
    def test_no_match_details(self):
        """Test NoMatchDetails creation."""
        from groq_speech.speech_recognizer import NoMatchDetails
        
        details = NoMatchDetails(
            reason="NoMatch",
            error_details="No speech detected"
        )
        
        self.assertEqual(details.reason, "NoMatch")
        self.assertEqual(details.error_details, "No speech detected")


class TestEnums(unittest.TestCase):
    """Test enum classes."""
    
    def test_result_reason_enum(self):
        """Test ResultReason enum values."""
        self.assertEqual(ResultReason.RecognizedSpeech.value, "RecognizedSpeech")
        self.assertEqual(ResultReason.NoMatch.value, "NoMatch")
        self.assertEqual(ResultReason.Canceled.value, "Canceled")
    
    def test_cancellation_reason_enum(self):
        """Test CancellationReason enum values."""
        self.assertEqual(CancellationReason.Error.value, "Error")
        self.assertEqual(CancellationReason.EndOfStream.value, "EndOfStream")
        self.assertEqual(CancellationReason.Timeout.value, "Timeout")
    
    def test_property_id_enum(self):
        """Test PropertyId enum values."""
        self.assertEqual(PropertyId.Speech_SegmentationStrategy.value, "Speech-SegmentationStrategy")
        self.assertEqual(PropertyId.Speech_LogFilename.value, "SpeechServiceConnection_LogFilename")


if __name__ == '__main__':
    # Check if GROQ_API_KEY is set for integration tests
    if not os.getenv('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY not set. Some tests may be skipped.")
        print("Set GROQ_API_KEY environment variable to run full integration tests.")
    
    unittest.main() 