"""
Custom exceptions for Groq Speech SDK.

This module provides custom exception classes for specific error conditions
that can occur during speech recognition and diarization operations.
"""


class GroqSpeechError(Exception):
    """Base exception for all Groq Speech SDK errors."""

    pass


class DiarizationError(GroqSpeechError):
    """Exception raised when speaker diarization fails."""

    pass


class AudioProcessingError(GroqSpeechError):
    """Exception raised when audio processing fails."""

    pass


class ConfigurationError(GroqSpeechError):
    """Exception raised when configuration is invalid."""

    pass


class APIError(GroqSpeechError):
    """Exception raised when Groq API calls fail."""

    pass
