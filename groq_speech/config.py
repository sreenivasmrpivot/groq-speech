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
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    GROQ_API_BASE_URL = os.getenv('GROQ_API_BASE_URL', 'https://api.groq.com/openai/v1')
    GROQ_CUSTOM_ENDPOINT = os.getenv('GROQ_CUSTOM_ENDPOINT', None)
    
    # Speech Recognition Settings
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en-US')
    DEFAULT_SAMPLE_RATE = int(os.getenv('DEFAULT_SAMPLE_RATE', '16000'))
    DEFAULT_CHANNELS = int(os.getenv('DEFAULT_CHANNELS', '1'))
    DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '1024'))
    
    # Audio Device Settings
    DEFAULT_DEVICE_INDEX = os.getenv('DEFAULT_DEVICE_INDEX', 'None')
    DEFAULT_FRAMES_PER_BUFFER = int(os.getenv('DEFAULT_FRAMES_PER_BUFFER', '1024'))
    
    # Recognition Timeouts
    DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '30'))
    DEFAULT_PHRASE_TIMEOUT = int(os.getenv('DEFAULT_PHRASE_TIMEOUT', '3'))
    DEFAULT_SILENCE_TIMEOUT = int(os.getenv('DEFAULT_SILENCE_TIMEOUT', '1'))
    
    # Advanced Features
    ENABLE_SEMANTIC_SEGMENTATION = os.getenv('ENABLE_SEMANTIC_SEGMENTATION', 'true').lower() == 'true'
    ENABLE_LANGUAGE_IDENTIFICATION = os.getenv('ENABLE_LANGUAGE_IDENTIFICATION', 'true').lower() == 'true'
    
    @classmethod
    def get_device_index(cls) -> Optional[int]:
        """Get the device index, handling 'None' string."""
        if cls.DEFAULT_DEVICE_INDEX == 'None':
            return None
        try:
            return int(cls.DEFAULT_DEVICE_INDEX)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate that API key is set."""
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != 'your_groq_api_key_here')
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get the API key with validation."""
        if not cls.validate_api_key():
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your .env file or environment variables."
            )
        return cls.GROQ_API_KEY


# Convenience function to get config
def get_config() -> Config:
    """Get the configuration instance."""
    return Config 