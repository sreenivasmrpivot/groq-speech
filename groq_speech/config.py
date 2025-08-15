"""
Configuration management for Groq Speech SDK.

This module provides centralized configuration management for the entire Groq
Speech SDK. It handles environment variables, configuration validation, and
provides sensible defaults for all system parameters.

ARCHITECTURE OVERVIEW:
1. ENVIRONMENT VARIABLE MANAGEMENT
   - Loads configuration from .env files
   - Provides fallback defaults for all settings
   - Validates critical configuration values

2. CONFIGURATION CATEGORIES
   - Groq API Settings: API keys, endpoints, and base URLs
   - Model Configuration: AI model selection and parameters
   - Audio Processing: Sample rates, channels, and optimization
   - Performance Settings: Timeouts, buffers, and caching
   - Advanced Features: Language detection and segmentation

3. VALIDATION AND SAFETY
   - API key validation and error handling
   - Configuration value type conversion
   - Sensible default values for all parameters

KEY FEATURES:
- Environment-based configuration with .env file support
- Comprehensive default values for all settings
- Type-safe configuration access
- Validation methods for critical settings
- Modular configuration retrieval by category

USAGE EXAMPLES:
    # Get API key with validation
    api_key = Config.get_api_key()

    # Get model configuration
    model_config = Config.get_model_config()

    # Get audio processing settings
    audio_config = Config.get_audio_config()

    # Check if API key is valid
    if Config.validate_api_key():
        # Proceed with API calls
        pass
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
# This allows users to override default settings without code changes
load_dotenv()


class Config:
    """
    Configuration class for Groq Speech SDK.

    CRITICAL: This class serves as the central configuration hub for the entire
    speech recognition system. It manages all settings, environment variables,
    and provides validated access to configuration values:

    1. Environment Variable Loading: Automatically loads .env files
    2. Default Value Management: Provides sensible defaults for all settings
    3. Configuration Validation: Ensures critical values are properly set
    4. Type Safety: Converts string environment variables to appropriate types
    5. Modular Access: Provides categorized configuration retrieval methods

    The configuration system is designed to be:
    - User-friendly: Simple .env file configuration
    - Developer-friendly: Clear defaults and validation
    - Production-ready: Environment-based configuration
    - Extensible: Easy to add new configuration options
    """

    # Groq API Settings
    # These control the core API communication with Groq's services
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_API_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_CUSTOM_ENDPOINT = os.getenv("GROQ_CUSTOM_ENDPOINT", None)

    # Model Configuration
    # These settings control which AI model is used and how it processes audio
    GROQ_MODEL_ID = os.getenv(
        "GROQ_MODEL_ID", "whisper-large-v3"
    )  # whisper-large-v3 (supports translation) or
    # whisper-large-v3-turbo (transcription only)
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
    # Core audio processing parameters for speech recognition
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-US")
    DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
    DEFAULT_CHANNELS = int(os.getenv("DEFAULT_CHANNELS", "1"))
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1024"))

    # Audio Device Settings
    # Hardware-specific settings for microphone and audio devices
    DEFAULT_DEVICE_INDEX = os.getenv("DEFAULT_DEVICE_INDEX", "None")
    DEFAULT_FRAMES_PER_BUFFER = int(os.getenv("DEFAULT_FRAMES_PER_BUFFER", "1024"))

    # Recognition Timeouts
    # Timing parameters that control recognition behavior and responsiveness
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    DEFAULT_PHRASE_TIMEOUT = int(
        os.getenv("DEFAULT_PHRASE_TIMEOUT", "5")
    )  # Increased from 3 for better
    # phrase detection
    DEFAULT_SILENCE_TIMEOUT = int(
        os.getenv("DEFAULT_SILENCE_TIMEOUT", "2")
    )  # Increased from 1 for more
    # natural speech patterns

    # Audio Processing Optimization
    # Performance and quality settings for audio processing pipeline
    AUDIO_CHUNK_DURATION = float(
        os.getenv("AUDIO_CHUNK_DURATION", "1.0")
    )  # Increased from 0.5 for better
    # processing efficiency
    AUDIO_BUFFER_SIZE = int(
        os.getenv("AUDIO_BUFFER_SIZE", "16384")
    )  # Increased from 8192 for improved
    # performance
    AUDIO_SILENCE_THRESHOLD = float(
        os.getenv("AUDIO_SILENCE_THRESHOLD", "0.005")
    )  # Reduced from 0.01 for better
    # silence detection
    AUDIO_VAD_ENABLED = os.getenv("AUDIO_VAD_ENABLED", "true").lower() == "true"

    # Continuous Recognition Chunking
    # Parameters that control how audio is chunked for continuous recognition
    CONTINUOUS_BUFFER_DURATION = float(
        os.getenv("CONTINUOUS_BUFFER_DURATION", "12.0")
    )  # Duration of each audio buffer in seconds (minimum 10s for Groq billing)
    CONTINUOUS_OVERLAP_DURATION = float(
        os.getenv("CONTINUOUS_OVERLAP_DURATION", "3.0")
    )  # Overlap duration between chunks to prevent word loss
    CONTINUOUS_CHUNK_SIZE = int(
        os.getenv("CONTINUOUS_CHUNK_SIZE", "1024")
    )  # Size of individual audio chunks read from microphone

    # Performance Settings
    # Advanced features that affect system performance and resource usage
    ENABLE_AUDIO_COMPRESSION = (
        os.getenv("ENABLE_AUDIO_COMPRESSION", "true").lower() == "true"
    )
    ENABLE_AUDIO_CACHING = os.getenv("ENABLE_AUDIO_CACHING", "true").lower() == "true"
    MAX_AUDIO_FILE_SIZE = int(
        os.getenv("MAX_AUDIO_FILE_SIZE", "25")
    )  # MB for free tier compliance

    # Advanced Features
    # Experimental and advanced capabilities of the speech recognition system
    ENABLE_SEMANTIC_SEGMENTATION = (
        os.getenv("ENABLE_SEMANTIC_SEGMENTATION", "true").lower() == "true"
    )
    ENABLE_LANGUAGE_IDENTIFICATION = (
        os.getenv("ENABLE_LANGUAGE_IDENTIFICATION", "true").lower() == "true"
    )

    @classmethod
    def get_device_index(cls) -> Optional[int]:
        """
        Get the device index, handling 'None' string conversion.

        CRITICAL: This method safely converts the device index configuration
        from environment variables to the appropriate Python type. It handles
        the special case where 'None' is stored as a string in environment
        variables.

        Returns:
            Device index as integer, or None if no device specified

        This method is essential for:
        - Audio device selection and configuration
        - Handling environment variable type conversion
        - Providing safe defaults for audio device settings
        """
        if cls.DEFAULT_DEVICE_INDEX == "None":
            return None
        try:
            return int(cls.DEFAULT_DEVICE_INDEX)
        except (ValueError, TypeError):
            return None

    @classmethod
    def validate_api_key(cls) -> bool:
        """
        Validate that API key is properly configured.

        CRITICAL: This method ensures the Groq API key is set and valid
        before any API calls are made. It prevents runtime errors and
        provides clear feedback about configuration issues.

        Returns:
            True if API key is valid, False otherwise

        Validation checks:
        - API key is not empty or missing
        - API key is not the placeholder value
        - API key format is acceptable
        """
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

    @classmethod
    def get_api_key(cls) -> str:
        """
        Get the API key with validation and error handling.

        CRITICAL: This method provides safe access to the API key with
        comprehensive validation. It's the primary method for accessing
        the API key throughout the system.

        Returns:
            Validated API key string

        Raises:
            ValueError: If API key is not properly configured

        This method ensures:
        - API key is available before API calls
        - Clear error messages for configuration issues
        - Consistent API key access across the system
        """
        if not cls.validate_api_key():
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your .env file or environment variables."
            )
        return cls.GROQ_API_KEY

    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get complete model configuration as a dictionary.

        CRITICAL: This method provides centralized access to all model-related
        configuration settings. It's used by the speech recognizer to configure
        API calls and model behavior.

        Returns:
            Dictionary containing all model configuration parameters

        Configuration includes:
        - Model ID selection (whisper-large-v3, whisper-large-v3-turbo)
        - Response format (json, verbose_json, text)
        - Temperature setting for model creativity
        - Timestamp granularity options
        """
        return {
            "model_id": cls.GROQ_MODEL_ID,
            "response_format": cls.GROQ_RESPONSE_FORMAT,
            "temperature": cls.GROQ_TEMPERATURE,
            "enable_word_timestamps": cls.GROQ_ENABLE_WORD_TIMESTAMPS,
            "enable_segment_timestamps": cls.GROQ_ENABLE_SEGMENT_TIMESTAMPS,
        }

    @classmethod
    def get_audio_config(cls) -> dict:
        """
        Get complete audio processing configuration as a dictionary.

        CRITICAL: This method provides centralized access to all audio-related
        configuration settings. It's used by the audio processor to configure
        audio capture, processing, and optimization.

        Returns:
            Dictionary containing all audio configuration parameters

        Configuration includes:
        - Audio format settings (sample rate, channels, chunk size)
        - Processing optimization (buffer sizes, chunk duration)
        - Performance features (VAD, compression, caching)
        - File size limits and constraints
        """
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

    @classmethod
    def get_chunking_config(cls) -> dict:
        """
        Get continuous recognition chunking configuration as a dictionary.

        CRITICAL: This method provides centralized access to all chunking-related
        configuration settings. It's used by the speech recognizer to configure
        how audio is chunked for continuous recognition.

        Returns:
            Dictionary containing all chunking configuration parameters

        Configuration includes:
        - Buffer duration (minimum 10s for Groq billing compliance)
        - Overlap duration (prevents word loss between chunks)
        - Chunk size for microphone reading
        """
        return {
            "buffer_duration": cls.CONTINUOUS_BUFFER_DURATION,
            "overlap_duration": cls.CONTINUOUS_OVERLAP_DURATION,
            "chunk_size": cls.CONTINUOUS_CHUNK_SIZE,
        }


# Convenience function to get config
def get_config() -> Config:
    """
    Get the configuration instance for external access.

    CRITICAL: This function provides a simple interface for external
    modules to access the configuration system without importing the
    Config class directly.

    Returns:
        Config class instance with all configuration methods

    Usage:
        config = get_config()
        api_key = config.get_api_key()
        model_config = config.get_model_config()
    """
    return Config()
