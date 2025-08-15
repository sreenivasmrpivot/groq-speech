"""
Custom exceptions for Groq Speech SDK.

This module provides a comprehensive exception hierarchy for the Groq Speech
SDK. It enables detailed error handling, debugging, and user feedback by
categorizing different types of errors and providing context-specific
information for each error scenario.

ARCHITECTURE OVERVIEW:
1. EXCEPTION HIERARCHY
   - Base GroqSpeechError class with common functionality
   - Specialized exception types for different error categories
   - Consistent error structure and information

2. ERROR CATEGORIZATION
   - Configuration and authentication errors
   - Audio processing and device errors
   - Recognition and API errors
   - Network and timeout errors
   - Validation and resource errors

3. ERROR HANDLING UTILITIES
   - API error handling functions
   - Validation helper functions
   - Context-aware error creation
   - Debugging and logging support

KEY FEATURES:
- Comprehensive error categorization and handling
- Detailed error context and debugging information
- Consistent error structure across the system
- Helper functions for common error scenarios
- Integration with logging and monitoring systems

USAGE EXAMPLES:
    # Handle specific error types
    try:
        recognizer.recognize_once()
    except AudioDeviceError as e:
        print(f"Audio device error: {e.message}")
        print(f"Device ID: {e.device_id}")

    # Use error handling utilities
    try:
        validate_api_key(api_key)
    except APIKeyError as e:
        print(f"API key error: {e.message}")

    # Handle API errors
    try:
        response = api_call()
        handle_api_error(response, "recognition")
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after} seconds")
"""

from typing import Optional, Dict, Any


class GroqSpeechError(Exception):
    """
    Base exception for all Groq Speech SDK errors.

    CRITICAL: This is the foundation of the entire exception system.
    It provides a consistent structure for all errors and enables
    comprehensive error handling throughout the SDK.

    The base exception includes:
    1. Human-readable error messages for users
    2. Machine-readable error codes for automation
    3. Detailed error context for debugging
    4. Consistent error structure across all exceptions

    Error handling benefits:
    - Consistent error reporting across the system
    - Detailed debugging information for developers
    - User-friendly error messages for end users
    - Automated error handling and recovery
    - Integration with logging and monitoring systems
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the base exception with error information.

        CRITICAL: This initialization sets up the complete error context
        that enables comprehensive error handling and debugging:

        Args:
            message: Human-readable error message for users
            error_code: Machine-readable error code for automation
            details: Additional error details for debugging

        The error structure enables:
        - User-friendly error reporting
        - Automated error handling
        - Detailed debugging information
        - Error categorization and filtering
        - Integration with monitoring systems
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self):
        """
        String representation of the exception.

        CRITICAL: This method provides a consistent string format
        that includes both the error code and message for easy
        identification and debugging.

        Returns:
            Formatted error string with code and message

        Format examples:
        - With error code: "[AUTH_ERROR] Authentication failed"
        - Without error code: "Authentication failed"
        """
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(GroqSpeechError):
    """
    Raised when there's an issue with SDK configuration.

    CRITICAL: This exception handles all configuration-related errors,
    including missing settings, invalid values, and configuration
    conflicts. It's essential for preventing runtime errors.

    Configuration errors can occur due to:
    - Missing required configuration values
    - Invalid configuration parameter values
    - Configuration file format issues
    - Environment variable problems
    - Configuration validation failures
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration error with context.

        Args:
            message: Descriptive error message
            config_key: The configuration key that caused the error
            details: Additional error details and context
        """
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class AuthenticationError(GroqSpeechError):
    """
    Raised when authentication fails.

    CRITICAL: This exception handles all authentication-related errors,
    including API key issues, token problems, and authorization
    failures. It's essential for security and access control.

    Authentication errors can occur due to:
    - Invalid or expired API keys
    - Missing authentication credentials
    - Incorrect authentication format
    - Authorization permission issues
    - Service authentication problems
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize authentication error.

        Args:
            message: Authentication error description
            details: Additional authentication error details
        """
        super().__init__(message, "AUTH_ERROR", details)


class APIKeyError(AuthenticationError):
    """
    Raised when API key is invalid or missing.

    CRITICAL: This exception specifically handles API key-related
    errors, which are the most common authentication issues.
    It provides detailed information about API key problems.

    API key errors can occur due to:
    - Missing API key configuration
    - Invalid API key format
    - Expired or revoked API keys
    - Incorrect API key storage
    - Environment variable issues
    """

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API key error.

        Args:
            message: API key error description
            details: Additional API key error details
        """
        super().__init__(message, details)


class AudioError(GroqSpeechError):
    """
    Raised when there's an issue with audio processing.

    CRITICAL: This exception handles all audio-related errors,
    including capture, processing, and format issues. It's
    essential for audio system reliability and debugging.

    Audio errors can occur due to:
    - Audio device initialization failures
    - Audio format incompatibility
    - Audio processing pipeline errors
    - Audio quality or corruption issues
    - Audio system resource problems
    """

    def __init__(
        self,
        message: str,
        audio_source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize audio error with source information.

        Args:
            message: Audio error description
            audio_source: The audio source that caused the error
            details: Additional audio error details
        """
        super().__init__(message, "AUDIO_ERROR", details)
        self.audio_source = audio_source


class AudioDeviceError(AudioError):
    """
    Raised when there's an issue with audio devices.

    CRITICAL: This exception handles audio device-specific errors,
    including hardware problems, driver issues, and device
    configuration problems. It's essential for device management.

    Device errors can occur due to:
    - Hardware malfunction or disconnection
    - Driver compatibility issues
    - Device permission problems
    - Device configuration errors
    - Resource allocation failures
    """

    def __init__(
        self,
        message: str,
        device_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize audio device error with device context.

        Args:
            message: Device error description
            device_id: The device ID that caused the error
            details: Additional device error details
        """
        super().__init__(message, f"device_{device_id}" if device_id else None, details)
        self.device_id = device_id


class AudioFileError(AudioError):
    """
    Raised when there's an issue with audio files.

    CRITICAL: This exception handles audio file-related errors,
    including format issues, corruption, and access problems.
    It's essential for file-based
    # audio processing.

    File errors can occur due to:
    - Unsupported audio file formats
    - Corrupted or damaged audio files
    - File access permission issues
    - File system problems
    - Audio file metadata issues
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize audio file error with file context.

        Args:
            message: File error description
            file_path: The file path that caused the error
            details: Additional file error details
        """
        super().__init__(message, file_path, details)
        self.file_path = file_path


class RecognitionError(GroqSpeechError):
    """
    Raised when speech recognition fails.

    CRITICAL: This exception handles all speech recognition failures,
    including API errors, processing failures, and result validation
    issues. It's essential for recognition system reliability.

    Recognition errors can occur due to:
    - API service failures or errors
    - Audio quality or format issues
    - Model processing failures
    - Result validation problems
    - Recognition timeout issues
    """

    def __init__(
        self,
        message: str,
        recognition_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize recognition error with session context.

        Args:
            message: Recognition error description
            recognition_id: The recognition session ID
            details: Additional recognition error details
        """
        super().__init__(message, "RECOGNITION_ERROR", details)
        self.recognition_id = recognition_id


class NetworkError(GroqSpeechError):
    """
    Raised when there's a network connectivity issue.

    CRITICAL: This exception handles all network-related errors,
    including connectivity issues, API communication problems,
    and network configuration issues. It's essential for
    reliable API communication.

    Network errors can occur due to:
    - Internet connectivity problems
    - API endpoint unavailability
    - Network timeout issues
    - Proxy or firewall problems
    - DNS resolution failures
    """

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize network error with endpoint context.

        Args:
            message: Network error description
            endpoint: The endpoint that failed
            details: Additional network error details
        """
        super().__init__(message, "NETWORK_ERROR", details)
        self.endpoint = endpoint


class RateLimitError(GroqSpeechError):
    """
    Raised when API rate limits are exceeded.

    CRITICAL: This exception handles API rate limiting scenarios,
    providing information about retry timing and rate limit
    policies. It's essential for implementing retry logic.

    Rate limit errors occur when:
    - API request frequency exceeds limits
    - Account quota is exceeded
    - Service is under high load
    - Rate limit policies are enforced
    - Burst request patterns are detected
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize rate limit error with retry information.

        Args:
            message: Rate limit error description
            retry_after: Seconds to wait before retrying
            details: Additional rate limit error details
        """
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after


class TimeoutError(GroqSpeechError):
    """
    Raised when an operation times out.

    CRITICAL: This exception handles timeout scenarios across
    all system operations, providing context about what
    operation failed and how long it took. It's essential
    for system responsiveness and reliability.

    Timeout errors can occur due to:
    - Network communication delays
    - Audio processing bottlenecks
    - API response delays
    - System resource constraints
    - Configuration timeout values
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize timeout error with timing context.

        Args:
            message: Timeout error description
            timeout_seconds: The timeout duration in seconds
            details: Additional timeout error details
        """
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds


class ValidationError(GroqSpeechError):
    """
    Raised when input validation fails.

    CRITICAL: This exception handles all input validation failures,
    providing detailed information about what failed validation
    and why. It's essential for data integrity and user feedback.

    Validation errors can occur due to:
    - Invalid parameter values
    - Missing required parameters
    - Parameter type mismatches
    - Value range violations
    - Format validation failures
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize validation error with field context.

        Args:
            message: Validation error description
            field: The field that failed validation
            value: The invalid value that caused the error
            details: Additional validation error details
        """
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class UnsupportedFeatureError(GroqSpeechError):
    """
    Raised when a feature is not supported.

    CRITICAL: This exception handles requests for unsupported
    features, providing clear information about what's not
    available and potential alternatives. It's essential for
    user guidance and system compatibility.

    Unsupported feature errors can occur due to:
    - Feature not implemented yet
    - Platform compatibility issues
    - API version limitations
    - Configuration restrictions
    - Service capability limits
    """

    def __init__(
        self,
        message: str,
        feature: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize unsupported feature error with feature context.

        Args:
            message: Feature support error description
            feature: The unsupported feature name
            details: Additional feature support details
        """
        super().__init__(message, "UNSUPPORTED_FEATURE_ERROR", details)
        self.feature = feature


class ResourceNotFoundError(GroqSpeechError):
    """
    Raised when a requested resource is not found.

    CRITICAL: This exception handles resource access failures,
    providing information about what resource was requested
    and why it couldn't be found. It's essential for
    resource management and user guidance.

    Resource not found errors can occur due to:
    - File or directory doesn't exist
    - API endpoint not available
    - Configuration file missing
    - Audio device not found
    - Model or resource unavailable
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize resource not found error with resource context.

        Args:
            message: Resource error description
            resource_type: The type of resource that wasn't found
            resource_id: The ID of the missing resource
            details: Additional resource error details
        """
        super().__init__(message, "RESOURCE_NOT_FOUND_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Convenience functions for error handling
def handle_api_error(response, context: Optional[str] = None) -> None:
    """
    Handle API errors and raise appropriate exceptions.

    CRITICAL: This function provides centralized API error handling
    that converts HTTP response codes into appropriate exception
    types. It's essential for consistent error handling across
    all API interactions.

    Args:
        response: The API response object with status code and headers
        context: Additional context for the error (e.g., "recognition")

    Raises:
        AuthenticationError: If authentication fails (401)
        RateLimitError: If rate limits are exceeded (429)
        NetworkError: If there's a server error (5xx)
        RecognitionError: If recognition fails (4xx)

    This function enables:
    - Consistent error handling across all API calls
    - Automatic exception type selection
    - Detailed error context and information
    - Rate limit retry timing information
    - Centralized error handling logic
    """
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key or authentication failed")
    elif response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RateLimitError(retry_after=int(retry_after) if retry_after else None)
    elif response.status_code >= 500:
        raise NetworkError(f"Server error: {response.status_code}")
    elif response.status_code >= 400:
        raise RecognitionError(f"Recognition failed: {response.status_code}")


def validate_api_key(api_key: Optional[str]) -> None:
    """
    Validate API key format and presence.

    CRITICAL: This function provides comprehensive API key validation
    that checks for presence, format, and basic structure. It's
    essential for preventing authentication errors early in
    # the configuration process.

    Args:
        api_key: The API key to validate

    Raises:
        APIKeyError: If the API key is invalid or missing

    Validation checks:
    - API key presence (not None or empty)
    - API key type (must be string)
    - API key format (Groq format validation)
    - Basic structure validation

    This function enables:
    - Early error detection and prevention
    - Clear error messages for configuration issues
    - Consistent API key validation across the system
    - User guidance for proper API key format
    """
    if not api_key:
        raise APIKeyError("API key is required")

    if not isinstance(api_key, str):
        raise APIKeyError("API key must be a string")

    if len(api_key.strip()) == 0:
        raise APIKeyError("API key cannot be empty")

    # Basic format validation (Groq API keys typically start with 'gsk_')
    if not api_key.startswith("gsk_"):
        raise APIKeyError(
            "Invalid API key format. Groq API keys should start with 'gsk_'"
        )
