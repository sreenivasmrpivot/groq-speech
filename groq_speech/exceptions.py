"""
Custom exceptions for Groq Speech SDK.
Provides comprehensive error handling and debugging capabilities.
"""

from typing import Optional, Dict, Any


class GroqSpeechError(Exception):
    """Base exception for all Groq Speech SDK errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        """String representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(GroqSpeechError):
    """Raised when there's an issue with SDK configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: The configuration key that caused the error
            details: Additional error details
        """
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class AuthenticationError(GroqSpeechError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "AUTH_ERROR", details)


class APIKeyError(AuthenticationError):
    """Raised when API key is invalid or missing."""
    
    def __init__(self, message: str = "Invalid or missing API key", 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize API key error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, details)


class AudioError(GroqSpeechError):
    """Raised when there's an issue with audio processing."""
    
    def __init__(self, message: str, audio_source: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize audio error.
        
        Args:
            message: Error message
            audio_source: The audio source that caused the error
            details: Additional error details
        """
        super().__init__(message, "AUDIO_ERROR", details)
        self.audio_source = audio_source


class AudioDeviceError(AudioError):
    """Raised when there's an issue with audio devices."""
    
    def __init__(self, message: str, device_id: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize audio device error.
        
        Args:
            message: Error message
            device_id: The device ID that caused the error
            details: Additional error details
        """
        super().__init__(message, f"device_{device_id}" if device_id else None, details)
        self.device_id = device_id


class AudioFileError(AudioError):
    """Raised when there's an issue with audio files."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize audio file error.
        
        Args:
            message: Error message
            file_path: The file path that caused the error
            details: Additional error details
        """
        super().__init__(message, file_path, details)
        self.file_path = file_path


class RecognitionError(GroqSpeechError):
    """Raised when speech recognition fails."""
    
    def __init__(self, message: str, recognition_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize recognition error.
        
        Args:
            message: Error message
            recognition_id: The recognition session ID
            details: Additional error details
        """
        super().__init__(message, "RECOGNITION_ERROR", details)
        self.recognition_id = recognition_id


class NetworkError(GroqSpeechError):
    """Raised when there's a network connectivity issue."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize network error.
        
        Args:
            message: Error message
            endpoint: The endpoint that failed
            details: Additional error details
        """
        super().__init__(message, "NETWORK_ERROR", details)
        self.endpoint = endpoint


class RateLimitError(GroqSpeechError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            details: Additional error details
        """
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after


class TimeoutError(GroqSpeechError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            timeout_seconds: The timeout duration in seconds
            details: Additional error details
        """
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds


class ValidationError(GroqSpeechError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: The field that failed validation
            value: The invalid value
            details: Additional error details
        """
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class UnsupportedFeatureError(GroqSpeechError):
    """Raised when a feature is not supported."""
    
    def __init__(self, message: str, feature: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize unsupported feature error.
        
        Args:
            message: Error message
            feature: The unsupported feature
            details: Additional error details
        """
        super().__init__(message, "UNSUPPORTED_FEATURE_ERROR", details)
        self.feature = feature


class ResourceNotFoundError(GroqSpeechError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize resource not found error.
        
        Args:
            message: Error message
            resource_type: The type of resource
            resource_id: The ID of the resource
            details: Additional error details
        """
        super().__init__(message, "RESOURCE_NOT_FOUND_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Convenience functions for error handling
def handle_api_error(response, context: Optional[str] = None) -> None:
    """
    Handle API errors and raise appropriate exceptions.
    
    Args:
        response: The API response object
        context: Additional context for the error
        
    Raises:
        AuthenticationError: If authentication fails
        RateLimitError: If rate limits are exceeded
        NetworkError: If there's a network issue
        RecognitionError: If recognition fails
    """
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key or authentication failed")
    elif response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        raise RateLimitError(retry_after=int(retry_after) if retry_after else None)
    elif response.status_code >= 500:
        raise NetworkError(f"Server error: {response.status_code}")
    elif response.status_code >= 400:
        raise RecognitionError(f"Recognition failed: {response.status_code}")


def validate_api_key(api_key: Optional[str]) -> None:
    """
    Validate API key format and presence.
    
    Args:
        api_key: The API key to validate
        
    Raises:
        APIKeyError: If the API key is invalid or missing
    """
    if not api_key:
        raise APIKeyError("API key is required")
    
    if not isinstance(api_key, str):
        raise APIKeyError("API key must be a string")
    
    if len(api_key.strip()) == 0:
        raise APIKeyError("API key cannot be empty")
    
    # Basic format validation (Groq API keys typically start with 'gsk_')
    if not api_key.startswith('gsk_'):
        raise APIKeyError("Invalid API key format. Groq API keys should start with 'gsk_'") 