"""
Speech configuration for Groq Speech services.
"""

import os
from typing import Optional
from .property_id import PropertyId
from .config import Config


class SpeechConfig:
    """
    Configuration class for Groq Speech services.
    Handles API keys, regions, endpoints, and other configuration options.
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

        Args:
            api_key: Groq API key (can also be set via GROQ_API_KEY env var)
            region: Groq region (optional, defaults to 'us-east-1')
            endpoint: Custom endpoint URL
            host: Custom host address
            authorization_token: Authorization token (alternative to API key)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or Config.get_api_key()
        if not self.api_key and not authorization_token:
            raise ValueError("API key or authorization token must be provided")

        self.region = region or "us-east-1"
        self.endpoint = endpoint or Config.GROQ_CUSTOM_ENDPOINT
        self.host = host
        self.authorization_token = authorization_token

        # Speech recognition settings
        self.speech_recognition_language = Config.DEFAULT_LANGUAGE
        self.translation_target_language = (
            "en"  # Default target language for translation
        )
        self.enable_translation = False  # Whether to use translation mode
        self.endpoint_id = None

        # Properties dictionary for custom settings
        self._properties = {}

        # Set default properties
        self._set_default_properties()

    def _set_default_properties(self):
        """Set default configuration properties."""
        self.set_property(PropertyId.Speech_SegmentationStrategy, "Silence")
        self.set_property(PropertyId.Speech_LogFilename, "")
        self.set_property(PropertyId.Speech_ServiceConnection_LogFilename, "")

        # Get model configuration from environment
        model_config = Config.get_model_config()

        # Set Groq API properties from environment config
        self.set_property(
            PropertyId.Speech_Recognition_GroqModelId, model_config["model_id"]
        )
        self.set_property(
            PropertyId.Speech_Recognition_ResponseFormat,
            model_config["response_format"],
        )
        self.set_property(
            PropertyId.Speech_Recognition_Temperature, str(model_config["temperature"])
        )
        self.set_property(
            PropertyId.Speech_Recognition_EnableWordLevelTimestamps,
            str(model_config["enable_word_timestamps"]).lower(),
        )
        self.set_property(
            PropertyId.Speech_Recognition_EnableSegmentTimestamps,
            str(model_config["enable_segment_timestamps"]).lower(),
        )

    def set_property(self, property_id: PropertyId, value: str):
        """
        Set a configuration property.

        Args:
            property_id: The property to set
            value: The value to set
        """
        self._properties[property_id] = value

    def get_property(self, property_id: PropertyId) -> str:
        """
        Get a configuration property.

        Args:
            property_id: The property to get

        Returns:
            The property value or empty string if not set
        """
        return self._properties.get(property_id, "")

    def set_speech_recognition_language(self, language: str):
        """
        Set the speech recognition language.

        Args:
            language: Language code (e.g., 'en-US', 'de-DE', 'fr-FR')
        """
        self.speech_recognition_language = language

    def set_translation_target_language(self, target_language: str):
        """
        Set the target language for translation.

        Args:
            target_language: Target language code (e.g., 'en', 'es', 'fr')
        """
        self.translation_target_language = target_language

    def set_endpoint_id(self, endpoint_id: str):
        """
        Set custom endpoint ID for custom speech models.

        Args:
            endpoint_id: The custom endpoint ID
        """
        self.endpoint_id = endpoint_id

    def get_authorization_headers(self) -> dict:
        """
        Get authorization headers for API requests.

        Returns:
            Dictionary with authorization headers
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

        Returns:
            Base URL string
        """
        if self.endpoint:
            return self.endpoint
        elif self.host:
            return f"https://{self.host}"
        else:
            return "https://api.groq.com"

    def validate(self):
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_key and not self.authorization_token:
            raise ValueError("API key or authorization token is required")

        if not self.speech_recognition_language:
            raise ValueError("Speech recognition language must be set")
