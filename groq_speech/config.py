"""
Configuration management for Groq Speech SDK.
Reads settings from .env file with sensible defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Configuration class for Groq Speech SDK."""

    # Groq API Settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_API_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_CUSTOM_ENDPOINT = os.getenv("GROQ_CUSTOM_ENDPOINT", None)

    # Model Configuration
    GROQ_MODEL_ID = os.getenv(
        "GROQ_MODEL_ID", "whisper-large-v3"
    )  # whisper-large-v3 (supports translation) or whisper-large-v3-turbo (transcription only)
    GROQ_RESPONSE_FORMAT = os.getenv(
        "GROQ_RESPONSE_FORMAT", "verbose_json"
    )  # json, verbose_json, text
    GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
    GROQ_ENABLE_WORD_TIMESTAMPS = (
        os.getenv("GROQ_ENABLE_WORD_TIMESTAMPS", "true").lower() == "true"
    )
    GROQ_ENABLE_SEGMENT_TIMESTAMPS = (
        os.getenv("GROQ_ENABLE_SEGMENT_TIMESTAMPS", "true").lower() == "true"
    )

    # Speech Recognition Settings
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-US")
    DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
    DEFAULT_CHANNELS = int(os.getenv("DEFAULT_CHANNELS", "1"))
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1024"))

    # Audio Device Settings
    DEFAULT_DEVICE_INDEX = os.getenv("DEFAULT_DEVICE_INDEX", "None")
    DEFAULT_FRAMES_PER_BUFFER = int(os.getenv("DEFAULT_FRAMES_PER_BUFFER", "1024"))

    # Recognition Timeouts
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    DEFAULT_PHRASE_TIMEOUT = int(
        os.getenv("DEFAULT_PHRASE_TIMEOUT", "5")
    )  # Increased from 3
    DEFAULT_SILENCE_TIMEOUT = int(
        os.getenv("DEFAULT_SILENCE_TIMEOUT", "2")
    )  # Increased from 1

    # Audio Processing Optimization
    AUDIO_CHUNK_DURATION = float(
        os.getenv("AUDIO_CHUNK_DURATION", "1.0")
    )  # Increased from 0.5
    AUDIO_BUFFER_SIZE = int(
        os.getenv("AUDIO_BUFFER_SIZE", "16384")
    )  # Increased from 8192
    AUDIO_SILENCE_THRESHOLD = float(
        os.getenv("AUDIO_SILENCE_THRESHOLD", "0.005")
    )  # Reduced from 0.01
    AUDIO_VAD_ENABLED = os.getenv("AUDIO_VAD_ENABLED", "true").lower() == "true"

    # Performance Settings
    ENABLE_AUDIO_COMPRESSION = (
        os.getenv("ENABLE_AUDIO_COMPRESSION", "true").lower() == "true"
    )
    ENABLE_AUDIO_CACHING = os.getenv("ENABLE_AUDIO_CACHING", "true").lower() == "true"
    MAX_AUDIO_FILE_SIZE = int(
        os.getenv("MAX_AUDIO_FILE_SIZE", "25")
    )  # MB for free tier

    # Advanced Features
    ENABLE_SEMANTIC_SEGMENTATION = (
        os.getenv("ENABLE_SEMANTIC_SEGMENTATION", "true").lower() == "true"
    )
    ENABLE_LANGUAGE_IDENTIFICATION = (
        os.getenv("ENABLE_LANGUAGE_IDENTIFICATION", "true").lower() == "true"
    )

    @classmethod
    def get_device_index(cls) -> Optional[int]:
        """Get the device index, handling 'None' string."""
        if cls.DEFAULT_DEVICE_INDEX == "None":
            return None
        try:
            return int(cls.DEFAULT_DEVICE_INDEX)
        except (ValueError, TypeError):
            return None

    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate that API key is set."""
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

    @classmethod
    def get_api_key(cls) -> str:
        """Get the API key with validation."""
        if not cls.validate_api_key():
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your .env file or environment variables."
            )
        return cls.GROQ_API_KEY

    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration."""
        return {
            "model_id": cls.GROQ_MODEL_ID,
            "response_format": cls.GROQ_RESPONSE_FORMAT,
            "temperature": cls.GROQ_TEMPERATURE,
            "enable_word_timestamps": cls.GROQ_ENABLE_WORD_TIMESTAMPS,
            "enable_segment_timestamps": cls.GROQ_ENABLE_SEGMENT_TIMESTAMPS,
        }

    @classmethod
    def get_audio_config(cls) -> dict:
        """Get audio processing configuration."""
        return {
            "sample_rate": cls.DEFAULT_SAMPLE_RATE,
            "channels": cls.DEFAULT_CHANNELS,
            "chunk_size": cls.DEFAULT_CHUNK_SIZE,
            "chunk_duration": cls.AUDIO_CHUNK_DURATION,
            "buffer_size": cls.AUDIO_BUFFER_SIZE,
            "silence_threshold": cls.AUDIO_SILENCE_THRESHOLD,
            "vad_enabled": cls.AUDIO_VAD_ENABLED,
            "enable_compression": cls.ENABLE_AUDIO_COMPRESSION,
            "enable_caching": cls.ENABLE_AUDIO_CACHING,
            "max_file_size": cls.MAX_AUDIO_FILE_SIZE,
        }


# Convenience function to get config
def get_config() -> Config:
    """Get the configuration instance."""
    return Config
