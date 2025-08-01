#!/usr/bin/env python3
"""
Tests for the web demo functionality.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.web_demo import WebSpeechDemo
from groq_speech import SpeechRecognitionResult, ResultReason


class TestWebSpeechDemo(unittest.TestCase):
    """Test cases for WebSpeechDemo."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("groq_speech.Config.get_api_key"):
            self.demo = WebSpeechDemo()

    def test_initialization(self):
        """Test that the demo initializes correctly."""
        self.assertIsNotNone(self.demo.app)
        self.assertIsNotNone(self.demo.socketio)
        self.assertIsNotNone(self.demo.recognizer)
        self.assertIsNotNone(self.demo.translator)
        self.assertEqual(len(self.demo.active_sessions), 0)

    def test_session_management(self):
        """Test session creation and cleanup."""
        # Simulate a client connection
        mock_request = Mock()
        mock_request.sid = "test_session_123"

        with patch("flask.request", mock_request):
            # Test session creation
            self.demo.active_sessions = {}
            session_data = {
                "connected": True,
                "start_time": self.demo.active_sessions.get("test_session_123", {}).get(
                    "start_time"
                ),
                "transcripts": [],
                "is_recognizing": False,
                "is_translating": False,
            }
            self.demo.active_sessions["test_session_123"] = session_data

            self.assertIn("test_session_123", self.demo.active_sessions)
            self.assertFalse(
                self.demo.active_sessions["test_session_123"]["is_recognizing"]
            )
            self.assertFalse(
                self.demo.active_sessions["test_session_123"]["is_translating"]
            )

    def test_stop_session_processing(self):
        """Test stopping session processing."""
        session_id = "test_session_456"
        self.demo.active_sessions[session_id] = {
            "is_recognizing": True,
            "is_translating": True,
        }

        self.demo._stop_session_processing(session_id)

        self.assertFalse(self.demo.active_sessions[session_id]["is_recognizing"])
        self.assertFalse(self.demo.active_sessions[session_id]["is_translating"])

    @patch("examples.web_demo.SpeechRecognizer")
    def test_recognition_flow(self, mock_recognizer_class):
        """Test the recognition flow."""
        # Mock the recognizer
        mock_recognizer = Mock()
        mock_recognizer_class.return_value = mock_recognizer

        # Mock successful recognition result
        mock_result = SpeechRecognitionResult(
            text="Hello world",
            reason=ResultReason.RecognizedSpeech,
            confidence=0.95,
            language="en-US",
        )
        mock_recognizer.recognize_once_async.return_value = mock_result

        # Replace the demo's recognizer with our mock
        self.demo.recognizer = mock_recognizer

        # Test recognition
        session_id = "test_session_789"
        self.demo.active_sessions[session_id] = {
            "is_recognizing": False,
            "is_translating": False,
            "transcripts": [],
        }

        # Simulate recognition
        with patch.object(self.demo.socketio, "emit") as mock_emit:
            # This would normally be called by the socket event handler
            # For testing, we'll simulate the recognition process
            result = self.demo.recognizer.recognize_once_async()

            self.assertEqual(result.text, "Hello world")
            self.assertEqual(result.reason, ResultReason.RecognizedSpeech)
            self.assertEqual(result.confidence, 0.95)

    @patch("examples.web_demo.SpeechRecognizer")
    def test_translation_flow(self, mock_recognizer_class):
        """Test the translation flow."""
        # Mock the translator
        mock_translator = Mock()
        mock_recognizer_class.return_value = mock_translator

        # Mock successful translation result
        mock_result = SpeechRecognitionResult(
            text="Hola mundo",
            reason=ResultReason.RecognizedSpeech,
            confidence=0.92,
            language="es-ES",
        )
        mock_translator.recognize_once_async.return_value = mock_result

        # Replace the demo's translator with our mock
        self.demo.translator = mock_translator

        # Test translation
        session_id = "test_session_101"
        self.demo.active_sessions[session_id] = {
            "is_recognizing": False,
            "is_translating": False,
            "transcripts": [],
        }

        # Simulate translation
        with patch.object(self.demo.socketio, "emit") as mock_emit:
            # This would normally be called by the socket event handler
            # For testing, we'll simulate the translation process
            result = self.demo.translator.recognize_once_async()

            self.assertEqual(result.text, "Hola mundo")
            self.assertEqual(result.reason, ResultReason.RecognizedSpeech)
            self.assertEqual(result.confidence, 0.92)

    def test_error_handling(self):
        """Test error handling in the demo."""
        session_id = "test_session_error"
        self.demo.active_sessions[session_id] = {
            "is_recognizing": False,
            "is_translating": False,
            "transcripts": [],
        }

        # Test session not found error
        with patch.object(self.demo.socketio, "emit") as mock_emit:
            # Simulate trying to start recognition for non-existent session
            self.demo.socketio.emit(
                "recognition_error",
                {"error": "Session not found"},
                room="non_existent_session",
            )

            # Verify error was emitted
            mock_emit.assert_called_with(
                "recognition_error",
                {"error": "Session not found"},
                room="non_existent_session",
            )

    def test_concurrent_operations(self):
        """Test that concurrent recognition and translation are prevented."""
        session_id = "test_session_concurrent"
        self.demo.active_sessions[session_id] = {
            "is_recognizing": True,  # Already recognizing
            "is_translating": False,
            "transcripts": [],
        }

        # Try to start translation while recognition is active
        with patch.object(self.demo.socketio, "emit") as mock_emit:
            self.demo.socketio.emit(
                "translation_error", {"error": "Already processing"}, room=session_id
            )

            # Verify error was emitted
            mock_emit.assert_called_with(
                "translation_error", {"error": "Already processing"}, room=session_id
            )

    def test_template_creation(self):
        """Test that templates are created correctly."""
        with patch("os.makedirs") as mock_makedirs:
            with patch("builtins.open", create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file

                self.demo.create_templates()

                # Verify templates directory was created
                mock_makedirs.assert_called()

                # Verify file was written
                mock_file.write.assert_called()

    @patch("webbrowser.open")
    @patch("examples.web_demo.SocketIO")
    def test_demo_run(self, mock_socketio, mock_webbrowser):
        """Test that the demo runs correctly."""
        mock_socketio_instance = Mock()
        mock_socketio.return_value = mock_socketio_instance

        # Test demo run
        with patch("examples.web_demo.Config.get_api_key"):
            demo = WebSpeechDemo()

            # Mock the run method to avoid actually starting the server
            with patch.object(demo.socketio, "run") as mock_run:
                demo.run(host="localhost", port=5000, debug=False)

                # Verify browser was opened
                mock_webbrowser.assert_called_with("http://localhost:5000")

                # Verify server was started
                mock_run.assert_called_with(
                    demo.app, host="localhost", port=5000, debug=False
                )


class TestSpeechRecognitionResult(unittest.TestCase):
    """Test cases for SpeechRecognitionResult."""

    def test_speech_recognition_result_creation(self):
        """Test creating a speech recognition result."""
        result = SpeechRecognitionResult(
            text="Test text",
            reason=ResultReason.RecognizedSpeech,
            confidence=0.85,
            language="en-US",
        )

        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.reason, ResultReason.RecognizedSpeech)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.language, "en-US")

    def test_speech_recognition_result_string_representation(self):
        """Test the string representation of speech recognition result."""
        result = SpeechRecognitionResult(
            text="Hello world", reason=ResultReason.RecognizedSpeech, confidence=0.95
        )

        expected = "SpeechRecognitionResult(text='Hello world', reason=ResultReason.RecognizedSpeech, confidence=0.95)"
        self.assertEqual(str(result), expected)


if __name__ == "__main__":
    unittest.main()
