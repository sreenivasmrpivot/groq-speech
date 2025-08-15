"""
Speech configuration for Groq Speech services.

This module provides the SpeechConfig class that manages all speech recognition
and translation configuration settings. It serves as the bridge between the
configuration system and the speech recognizer, handling API credentials,
language settings, and model parameters.

ARCHITECTURE OVERVIEW:
1. CONFIGURATION INTEGRATION
   - Integrates with the main Config class for environment-based settings
   - Provides programmatic access to speech-specific configuration
   - Manages property-based configuration system

2. CREDENTIAL MANAGEMENT
   - API key handling with environment variable fallback
   - Authorization token support for alternative authentication
   - Custom endpoint and host configuration

3. SPEECH RECOGNITION SETTINGS
   - Language configuration for recognition and translation
   - Model parameter management
   - Custom property system for extensibility

4. VALIDATION AND SAFETY
   - Configuration validation before use
   - Error handling for missing credentials
   - Safe defaults for all settings

KEY FEATURES:
- Environment-based configuration with programmatic overrides
- Comprehensive property system for custom settings
- Language management for recognition and translation
- Model configuration integration
- Authorization header generation
- Configuration validation and error handling

USAGE EXAMPLES:
    # Basic configuration with API key
    config = SpeechConfig(api_key="your_api_key")

    # Configuration with custom endpoint
    config = SpeechConfig(
        api_key="your_api_key",
        endpoint="https://custom.groq.com"
    )

    # Translation configuration
    config = SpeechConfig()
    config.enable_translation = True
    config.set_translation_target_language("en")

    # Custom property configuration
    config.set_property(PropertyId.Speech_Recognition_Temperature, "0.1")
"""

from typing import Optional
from .property_id import PropertyId
from .config import Config


class SpeechConfig:
    """
    Configuration class for Groq Speech services.

    CRITICAL: This class manages all speech recognition and translation
    configuration settings. It's the primary interface for configuring
    the speech recognition system and provides:

    1. Credential Management: API keys, tokens, and authorization
    2. Language Configuration: Recognition and translation languages
    3. Model Settings: AI model parameters and behavior
    4. Custom Properties: Extensible configuration system
    5. Validation: Configuration integrity checking

    The SpeechConfig integrates with the main Config class to provide:
    - Environment-based defaults with programmatic overrides
    - Property-based configuration for extensibility
    - Validation and error handling for production use
    - Authorization header generation for API calls
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        host: Optional[str] = None,
        authorization_token: Optional[str] = None,
    ):
        """
        Initialize SpeechConfig with Groq credentials and settings.

        CRITICAL: This initialization sets up the complete speech recognition
        configuration. It handles credential management, default settings,
        and property initialization:

        Args:
            api_key: Groq API key (can also be set via GROQ_API_KEY env var)
            region: Groq region (optional, defaults to 'us-east-1')
            endpoint: Custom endpoint URL for alternative Groq instances
            host: Custom host address for network configuration
            authorization_token: Authorization token (alternative to API key)

        Initialization process:
        1. Credential setup with environment variable fallback
        2. Default language and translation settings
        3. Property system initialization with defaults
        4. Model configuration integration
        5. Validation and error checking
        """
        # Get API key from environment if not provided
        # This allows users to set API keys via environment variables
        self.api_key = api_key or Config.get_api_key()
        if not self.api_key and not authorization_token:
            raise ValueError("API key or authorization token must be provided")

        # Network and endpoint configuration
        self.region = region or "us-east-1"
        self.endpoint = endpoint or Config.GROQ_CUSTOM_ENDPOINT
        self.host = host
        self.authorization_token = authorization_token

        # Speech recognition and translation settings
        self.speech_recognition_language = Config.DEFAULT_LANGUAGE
        self.translation_target_language = (
            "en"  # Default target language for translation
        )
        self.enable_translation = False  # Whether to use translation mode
        self.endpoint_id = None

        # Properties dictionary for custom settings
        # This provides extensibility for future configuration options
        self._properties = {}

        # Set default properties from environment configuration
        self._set_default_properties()

    def _set_default_properties(self):
        """
        Set default configuration properties from environment settings.

        CRITICAL: This method initializes the property system with sensible
        defaults and environment-based configuration. It ensures the speech
        recognizer has all necessary settings before operation.

        Default properties include:
        - Speech segmentation strategy (silence-based)
        - Logging configuration for debugging
        - Model parameters from environment configuration
        - Timestamp granularity settings
        """
        # Set speech processing defaults
        self.set_property(PropertyId.Speech_SegmentationStrategy, "Silence")
        self.set_property(PropertyId.Speech_LogFilename, "")
        self.set_property(PropertyId.Speech_ServiceConnection_LogFilename, "")

        # Get model configuration from environment
        # This ensures consistency with the main configuration system
        model_config = Config.get_model_config()

        # Set Groq API properties from environment config
        # These control the AI model behavior and response format
        self.set_property(
            PropertyId.Speech_Recognition_GroqModelId, model_config["model_id"]
        )
        self.set_property(
            PropertyId.Speech_Recognition_ResponseFormat,
            model_config["response_format"],
        )
        temp_prop = PropertyId.Speech_Recognition_Temperature
        self.set_property(
            temp_prop, str(model_config["temperature"])
        )
        word_timestamps_prop = PropertyId.Speech_Recognition_EnableWordLevelTimestamps
        self.set_property(
            word_timestamps_prop,
            str(model_config["enable_word_timestamps"]).lower(),
        )
        segment_timestamps_prop = PropertyId.Speech_Recognition_EnableSegmentTimestamps
        self.set_property(
            segment_timestamps_prop,
            str(model_config["enable_segment_timestamps"]).lower(),
        )

    def set_property(self, property_id: PropertyId, value: str):
        """
        Set a configuration property for custom behavior.

        CRITICAL: This method provides the extensible configuration system
        that allows applications to customize speech recognition behavior
        beyond the standard settings.

        Args:
            property_id: The property identifier from PropertyId enum
            value: The string value to set for the property

        The property system enables:
        - Custom model parameters
        - Advanced audio processing settings
        - Debugging and logging configuration
        - Future feature enablement
        """
        self._properties[property_id] = value

    def get_property(self, property_id: PropertyId) -> str:
        """
        Get a configuration property value.

        CRITICAL: This method provides access to the property-based
        configuration system. It's used throughout the speech recognizer
        to access custom settings and model parameters.

        Args:
            property_id: The property identifier from PropertyId enum

        Returns:
            The property value or empty string if not set

        This method ensures:
        - Safe access to configuration properties
        - Consistent default behavior for unset properties
        - Integration with the property system
        """
        return self._properties.get(property_id, "")

    def set_speech_recognition_language(self, language: str):
        """
        Set the speech recognition language.

        CRITICAL: This method configures the language for speech recognition.
        Note that Groq's API automatically detects the language, so this
        setting is primarily for compatibility and
        # future features.

        Args:
            language: Language code (e.g., 'en-US', 'de-DE', 'fr-FR')

        Language codes should follow standard format:
        - Primary language code (e.g., 'en', 'de', 'fr')
        - Optional region code (e.g., 'US', 'DE', 'FR')
        - Combined format: 'en-US', 'de-DE', 'fr-FR'
        """
        self.speech_recognition_language = language

    def set_translation_target_language(self, target_language: str):
        """
        Set the target language for translation.

        CRITICAL: This method configures the target language when using
        translation mode. The speech recognizer will translate speech
        from any language to the specified target language.

        Args:
            target_language: Target language code (e.g., 'en', 'es', 'fr')

        Translation features:
        - Converts speech in any language to the target language
        - Supports all major world languages
        - Provides natural, context-aware translations
        - Maintains speech timing and structure
        """
        self.translation_target_language = target_language

    def set_endpoint_id(self, endpoint_id: str):
        """
        Set custom endpoint ID for custom speech models.

        CRITICAL: This method allows the use of custom speech recognition
        models and endpoints. It's essential for enterprise applications
        that require specialized models or custom deployments.

        Args:
            endpoint_id: The custom endpoint identifier

        Custom endpoints enable:
        - Specialized domain models (medical, legal, technical)
        - Custom vocabulary and terminology
        - Enhanced accuracy for specific use cases
        - Enterprise-grade speech recognition
        """
        self.endpoint_id = endpoint_id

    def get_authorization_headers(self) -> dict:
        """
        Get authorization headers for API requests.

        CRITICAL: This method generates the proper authorization headers
        required for all Groq API calls. It handles both API key and
        token-based authentication methods.

        Returns:
            Dictionary with authorization headers ready for HTTP requests

        Raises:
            ValueError: If no authorization credentials are available

        Authentication methods supported:
        - API key-based authentication (Bearer token)
        - Custom authorization tokens
        - Environment variable fallback
        """
        if self.authorization_token:
            return {"Authorization": f"Bearer {self.authorization_token}"}
        elif self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        else:
            raise ValueError("No authorization credentials available")

    def get_base_url(self) -> str:
        """
        Get the base URL for API requests.

        CRITICAL: This method determines the endpoint URL for all API
        calls. It supports custom endpoints, custom hosts, and the
        default Groq API endpoint.

        Returns:
            Base URL string for API requests

        URL resolution priority:
        1. Custom endpoint (if specified)
        2. Custom host (if specified)
        3. Default Groq API endpoint

        This enables:
        - Custom Groq deployments
        - Alternative API endpoints
        - Network-specific configurations
        """
        if self.endpoint:
            return self.endpoint
        elif self.host:
            return f"https://{self.host}"
        else:
            return "https://api.groq.com"

    def validate(self):
        """
        Validate the configuration before use.

        CRITICAL: This method ensures the configuration is complete
        and valid before the speech recognizer attempts to use it.
        It prevents runtime errors and provides clear feedback about
        configuration issues.

        Raises:
            ValueError: If configuration is invalid or incomplete

        Validation checks:
        - Authorization credentials are present
        - Required language settings are configured
        - Property system is properly initialized
        - Network configuration is valid

        This method should be called before:
        - Initializing the speech recognizer
        - Making API calls
        - Starting recognition sessions
        """
        if not self.api_key and not self.authorization_token:
            raise ValueError("API key or authorization token is required")

        if not self.speech_recognition_language:
            raise ValueError("Speech recognition language must be set")
