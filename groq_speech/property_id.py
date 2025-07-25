"""
Property ID enums for Groq Speech services.
"""

from enum import Enum


class PropertyId(Enum):
    """
    Enumeration of speech service property IDs.
    """
    # Speech recognition properties
    Speech_Recognition_Language = "SpeechServiceConnection_RecoLanguage"
    Speech_Recognition_EndpointId = "SpeechServiceConnection_EndpointId"
    Speech_Recognition_ModelId = "SpeechServiceConnection_ModelId"
    
    # Audio properties
    Speech_AudioInput_Processing_EnableAec = "Speech-AudioInput-Processing-EnableAec"
    Speech_AudioInput_Processing_EnableAgc = "Speech-AudioInput-Processing-EnableAgc"
    Speech_AudioInput_Processing_EnableNs = "Speech-AudioInput-Processing-EnableNs"
    
    # Segmentation properties
    Speech_SegmentationStrategy = "Speech-SegmentationStrategy"
    
    # Logging properties
    Speech_LogFilename = "SpeechServiceConnection_LogFilename"
    Speech_ServiceConnection_LogFilename = "SpeechServiceConnection_LogFilename"
    
    # Connection properties
    Speech_ServiceConnection_Url = "SpeechServiceConnection_Url"
    Speech_ServiceConnection_ProxyHostName = "SpeechServiceConnection_ProxyHostName"
    Speech_ServiceConnection_ProxyPort = "SpeechServiceConnection_ProxyPort"
    Speech_ServiceConnection_ProxyUserName = "SpeechServiceConnection_ProxyUserName"
    Speech_ServiceConnection_ProxyPassword = "SpeechServiceConnection_ProxyPassword"
    
    # Recognition properties
    Speech_Recognition_EnableDictation = "Speech_Recognition_EnableDictation"
    Speech_Recognition_EnableWordLevelTimestamps = "Speech_Recognition_EnableWordLevelTimestamps"
    Speech_Recognition_EnableIntermediateResults = "Speech_Recognition_EnableIntermediateResults"
    
    # Confidence properties
    Speech_Recognition_ConfidenceThreshold = "Speech_Recognition_ConfidenceThreshold"
    
    # Language identification properties
    Speech_Recognition_EnableLanguageIdentification = "Speech_Recognition_EnableLanguageIdentification"
    Speech_Recognition_LanguageIdentificationMode = "Speech_Recognition_LanguageIdentificationMode"
    
    # Custom properties
    Speech_Recognition_CustomEndpointId = "Speech_Recognition_CustomEndpointId"
    Speech_Recognition_CustomModelId = "Speech_Recognition_CustomModelId"
    
    # Real-time properties
    Speech_Recognition_EnableRealTimeTranscription = "Speech_Recognition_EnableRealTimeTranscription"
    Speech_Recognition_EnablePunctuation = "Speech_Recognition_EnablePunctuation"
    Speech_Recognition_EnableProfanityFilter = "Speech_Recognition_EnableProfanityFilter" 