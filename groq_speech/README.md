# Groq Speech SDK - Core Module

This is the core module of the Groq Speech SDK, providing speech recognition, translation, and speaker diarization capabilities.

## 🎯 Critical Entry Points

### 1. SpeechRecognizer (Primary Entry Point)
**Location**: `groq_speech.speech_recognizer.SpeechRecognizer`

The main class for all speech recognition operations:

```python
from groq_speech import SpeechRecognizer, SpeechConfig

config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# File processing
result = recognizer.recognize_file("audio.wav", enable_diarization=True)
result = recognizer.translate_file("audio.wav", enable_diarization=False)

# Microphone processing
result = recognizer.recognize_microphone(duration=10)
result = recognizer.translate_microphone(duration=10)

# Raw audio data
result = recognizer.recognize_audio_data(audio_data)
result = recognizer.translate_audio_data(audio_data)
```

### 2. Config (Configuration Management)
**Location**: `groq_speech.config.Config`

Centralized configuration management:

```python
from groq_speech import Config

# Get configuration
api_key = Config.get_api_key()
model_config = Config.get_model_config()
chunking_config = Config.get_chunking_config()
```

### 3. Result Objects (Data Access)
**Location**: `groq_speech.speech_recognizer.SpeechRecognitionResult`

Structured result data access:

```python
result = recognizer.recognize_file("audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Timestamps: {result.timestamps}")
```

## 🏗️ Architecture

The SDK follows a **simplified single-entry-point architecture**:

### Primary Entry Point
- **SpeechRecognizer**: Main orchestrator for all speech operations

### Service Classes (Internal)
- **DiarizationService**: Handles speaker diarization with smart grouping
- **AudioProcessor**: Manages audio processing and chunking
- **VADService**: Voice Activity Detection with multiple fallback options
- **GroqAPIClient**: Handles API communication
- **ResponseParser**: Parses API responses

### Configuration
- **SpeechConfig**: Speech recognition settings
- **Config**: Centralized configuration management
- **DiarizationConfig**: Diarization parameters

## 📊 Performance

### Big O Complexity
- **File Processing**: O(n) where n is audio length
- **Diarization**: O(n log n) due to Pyannote.audio processing
- **API Calls**: O(1) per chunk, O(k) total where k is number of chunks
- **Memory Usage**: O(1) for streaming, O(n) for file processing

### Optimization Features
- **Smart Grouping**: 24MB-optimized speaker segment grouping
- **Voice Activity Detection**: Intelligent silence detection
- **Caching**: Model and configuration caching
- **Retry Logic**: Automatic retry for transient errors

## 🚨 Error Handling

### Exception Hierarchy
```python
from groq_speech.exceptions import (
    GroqSpeechException,
    ConfigurationError,
    APIError,
    AudioError,
    DiarizationError
)

try:
    result = recognizer.recognize_file("audio.wav")
except APIError as e:
    print(f"API Error: {e}")
except AudioError as e:
    print(f"Audio Error: {e}")
```

### Fallback Mechanisms
- **Diarization Fallback**: Falls back to basic transcription if diarization fails
- **API Retry**: Automatic retry for transient API errors
- **Audio Processing**: Graceful handling of audio format issues

## 📚 API Reference

For detailed API documentation, see:
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture overview
- [DATAFLOWS.md](DATAFLOWS.md) - Data flow diagrams

## 🔄 Migration from Previous Versions

### Breaking Changes
- **AudioConfig removed**: Functionality moved to SpeechRecognizer
- **EnhancedDiarizer removed**: Consolidated into Diarizer
- **Multiple entry points**: Simplified to single SpeechRecognizer
- **Constructor parameters**: Simplified and standardized

### Migration Guide
```python
# Old way
from groq_speech import AudioConfig, EnhancedDiarizer
audio_config = AudioConfig()
diarizer = EnhancedDiarizer()

# New way
from groq_speech import SpeechRecognizer
recognizer = SpeechRecognizer()
# All functionality through SpeechRecognizer
```