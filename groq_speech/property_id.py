"""
Property ID enums for Groq Speech services.

This module provides the PropertyId enumeration that defines all configurable
properties for the speech recognition system. These property IDs enable
fine-grained control over recognition behavior, audio processing, and
system configuration.

ARCHITECTURE OVERVIEW:
1. PROPERTY CATEGORIZATION
   - Speech recognition core properties
   - Audio processing and enhancement
   - Connection and network configuration
   - Logging and debugging settings
   - Groq API specific parameters

2. PROPERTY SYSTEM INTEGRATION
   - Used by SpeechConfig for configuration management
   - Enables runtime property modification
   - Provides consistent property naming
   - Supports extensible configuration

3. CONFIGURATION FLEXIBILITY
   - Environment-based property overrides
   - Runtime property modification
   - Default value management
   - Property validation and error handling

KEY FEATURES:
- Comprehensive coverage of all configurable aspects
- Consistent naming convention for properties
- Support for both standard and Groq-specific properties
- Easy extensibility for future configuration options
- Integration with configuration management system

USAGE EXAMPLES:
    # Set recognition language
    config.set_property(PropertyId.Speech_Recognition_Language, "en-US")

    # Enable word-level timestamps
    word_timestamps_prop = PropertyId.Speech_Recognition_EnableWordLevelTimestamps
    config.set_property(word_timestamps_prop, "true")

    # Configure Groq model
    groq_model_prop = PropertyId.Speech_Recognition_GroqModelId
    config.set_property(groq_model_prop, "whisper-large-v3")

    # Set confidence threshold
    confidence_prop = PropertyId.Speech_Recognition_ConfidenceThreshold
    config.set_property(confidence_prop, "0.8")
"""

from enum import Enum


class PropertyId(Enum):
    """
    Enumeration of speech service property IDs.

    CRITICAL: This enum defines all configurable properties for the speech
    recognition system. It provides a unified interface for configuring
    every aspect of the system's behavior and performance.

    The PropertyId system enables:
    1. Fine-grained control over recognition behavior
    2. Runtime configuration modification
    3. Environment-based property overrides
    4. Consistent property naming across the system
    5. Extensible configuration for future features

    Properties are organized into logical categories:
    - Core recognition settings
    - Audio processing parameters
    - Network and connection configuration
    - Logging and debugging options
    - Groq API specific parameters

    Each property has specific:
    - Default values from configuration
    - Validation rules and constraints
    - Impact on system behavior
    - Performance implications
    """

    # Speech recognition properties
    # Core language and model configuration
    # for recognition
    Speech_Recognition_Language = "SpeechServiceConnection_RecoLanguage"
    Speech_Recognition_EndpointId = "SpeechServiceConnection_EndpointId"
    Speech_Recognition_ModelId = "SpeechServiceConnection_ModelId"

    # Audio properties
    # Audio enhancement and processing
    # configuration
    Speech_AudioInput_Processing_EnableAec = "Speech-AudioInput-Processing-EnableAec"
    Speech_AudioInput_Processing_EnableAgc = "Speech-AudioInput-Processing-EnableAgc"
    Speech_AudioInput_Processing_EnableNs = "Speech-AudioInput-Processing-EnableNs"

    # Segmentation properties
    # Speech segmentation strategy for
    # continuous recognition
    Speech_SegmentationStrategy = "Speech-SegmentationStrategy"

    # Logging properties
    # Logging configuration for debugging and monitoring
    Speech_LogFilename = "SpeechServiceConnection_LogFilename"
    Speech_ServiceConnection_LogFilename = "SpeechServiceConnection_LogFilename"

    # Connection properties
    # Network and proxy configuration for API communication
    Speech_ServiceConnection_Url = "SpeechServiceConnection_Url"
    Speech_ServiceConnection_ProxyHostName = "SpeechServiceConnection_ProxyHostName"
    Speech_ServiceConnection_ProxyPort = "SpeechServiceConnection_ProxyPort"
    Speech_ServiceConnection_ProxyUserName = "SpeechServiceConnection_ProxyUserName"
    Speech_ServiceConnection_ProxyPassword = "SpeechServiceConnection_ProxyPassword"

    # Recognition properties
    # Advanced recognition features and behavior
    Speech_Recognition_EnableDictation = "Speech_Recognition_EnableDictation"
    Speech_Recognition_EnableWordLevelTimestamps = "Speech_Recognition_EnableWordLevelTimestamps"
    Speech_Recognition_EnableIntermediateResults = "Speech_Recognition_EnableIntermediateResults"

    # Confidence properties
    # Confidence threshold configuration for result filtering
    Speech_Recognition_ConfidenceThreshold = "Speech_Recognition_ConfidenceThreshold"

    # Language identification properties
    # Automatic language detection and identification
    Speech_Recognition_EnableLanguageIdentification = "Speech_Recognition_EnableLanguageIdentification"
    Speech_Recognition_LanguageIdentificationMode = "Speech_Recognition_LanguageIdentificationMode"

    # Custom properties
    # Custom model and endpoint configuration
    Speech_Recognition_CustomEndpointId = "Speech_Recognition_CustomEndpointId"
    Speech_Recognition_CustomModelId = "Speech_Recognition_CustomModelId"

    # Real-time properties
    # Real-time processing and output configuration
    Speech_Recognition_EnableRealTimeTranscription = "Speech_Recognition_EnableRealTimeTranscription"
    Speech_Recognition_EnablePunctuation = "Speech_Recognition_EnablePunctuation"
    Speech_Recognition_EnableProfanityFilter = "Speech_Recognition_EnableProfanityFilter"

    # Groq API specific properties
    # Groq-specific configuration for optimal API usage
    Speech_Recognition_GroqModelId = "Speech_Recognition_GroqModelId"
    Speech_Recognition_ResponseFormat = "Speech_Recognition_ResponseFormat"
    Speech_Recognition_Temperature = "Speech_Recognition_Temperature"
    Speech_Recognition_Prompt = "Speech_Recognition_Prompt"
    Speech_Recognition_EnableSegmentTimestamps = "Speech_Recognition_EnableSegmentTimestamps"
