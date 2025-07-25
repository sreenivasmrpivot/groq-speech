"""
Groq Speech SDK - Real-time speech recognition and synthesis
A Python SDK for Groq's speech services, providing speech-to-text and text-to-speech capabilities.
"""

from .speech_config import SpeechConfig
from .speech_recognizer import SpeechRecognizer
from .audio_config import AudioConfig
from .result_reason import ResultReason, CancellationReason
from .property_id import PropertyId
from .config import Config, get_config

__version__ = "1.0.0"
__all__ = [
    "SpeechConfig",
    "SpeechRecognizer", 
    "AudioConfig",
    "ResultReason",
    "CancellationReason",
    "PropertyId",
    "Config",
    "get_config"
] 