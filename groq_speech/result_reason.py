"""
Result reason enums for Groq Speech services.
"""

from enum import Enum


class ResultReason(Enum):
    """
    Enumeration of speech recognition result reasons.
    """
    # Successful recognition
    RecognizedSpeech = "RecognizedSpeech"
    
    # No speech detected
    NoMatch = "NoMatch"
    
    # Recognition was canceled
    Canceled = "Canceled"
    
    # Recognition is in progress (for continuous recognition)
    Recognizing = "Recognizing"
    
    # Session started
    SessionStarted = "SessionStarted"
    
    # Session stopped
    SessionStopped = "SessionStopped"


class CancellationReason(Enum):
    """
    Enumeration of cancellation reasons.
    """
    # Error occurred during recognition
    Error = "Error"
    
    # Recognition was canceled by user
    EndOfStream = "EndOfStream"
    
    # Recognition was canceled due to timeout
    Timeout = "Timeout"
    
    # Recognition was canceled due to network issues
    NetworkError = "NetworkError"
    
    # Recognition was canceled due to service issues
    ServiceError = "ServiceError"
    
    # Recognition was canceled due to invalid audio
    InvalidAudio = "InvalidAudio"
    
    # Recognition was canceled due to language not supported
    LanguageNotSupported = "LanguageNotSupported" 