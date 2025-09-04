"""
Groq Speech SDK - Real-time speech recognition and synthesis.

This module provides a comprehensive Python SDK for Groq's speech services,
enabling real-time speech-to-text and text-to-speech capabilities with
advanced AI-powered models.

ARCHITECTURE OVERVIEW:
1. CORE COMPONENTS
   - SpeechConfig: Configuration management for speech services
   - SpeechRecognizer: Main speech recognition engine
   - AudioConfig: Audio input/output configuration and management
   - ResultReason: Recognition result status and categorization

2. CONFIGURATION SYSTEM
   - Config: Centralized configuration management
   - PropertyId: Configurable property definitions
   - Environment-based configuration with defaults

3. ERROR HANDLING
   - Comprehensive exception hierarchy
   - Detailed error context and debugging
   - User-friendly error messages

KEY FEATURES:
- Real-time speech recognition with Groq's AI models
- Support for both transcription and translation
- Configurable audio processing and optimization
- Comprehensive error handling and debugging
- Environment-based configuration management
- Audio device enumeration and selection
- File-based and microphone-based audio processing
- Performance monitoring and timing metrics
- Event-driven architecture for real-time applications

USAGE EXAMPLES:
    # Basic speech recognition
    from groq_speech import SpeechConfig, SpeechRecognizer

    config = SpeechConfig()
    recognizer = SpeechRecognizer(config)
    result = recognizer.recognize_once()

    # Translation to English
    config.enable_translation = True
    recognizer = SpeechRecognizer(config)
    result = recognizer.translate_audio_data(audio_data)

    # Continuous recognition
    recognizer.start_continuous_recognition()
    # ... handle events ...
    recognizer.stop_continuous_recognition()

    # Audio configuration
    from groq_speech import AudioConfig

    with AudioConfig() as audio:
        chunk = audio.read_audio_chunk(1024)

    # Configuration management
    from groq_speech import Config

    api_key = Config.get_api_key()
    model_config = Config.get_model_config()
"""

from .speech_config import SpeechConfig
from .speech_recognizer import SpeechRecognizer, SpeechRecognitionResult
from .result_reason import ResultReason, CancellationReason
from .property_id import PropertyId
from .config import Config, get_config
from .speaker_diarization import DiarizationConfig, SpeakerSegment, DiarizationResult, Diarizer

# SDK version information
__version__ = "1.0.0"

# Public API exports
# These are the main classes and functions that users should import
__all__ = [
    "SpeechConfig",  # Speech recognition configuration
    "SpeechRecognizer",  # Main speech recognition engine
    "SpeechRecognitionResult",  # Recognition result objects
    "ResultReason",  # Recognition result status
    "CancellationReason",  # Error categorization
    "PropertyId",  # Configurable properties
    "Config",  # Configuration management
    "get_config",  # Configuration access function
    "DiarizationConfig",  # Speaker diarization config
    "SpeakerSegment",  # Individual speaker segment
    "DiarizationResult",  # Complete diarization result
    "Diarizer",  # Simplified diarizer (replaces SpeakerDiarizer and EnhancedDiarizer)
]
