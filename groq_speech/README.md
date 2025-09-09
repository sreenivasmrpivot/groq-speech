# Groq Speech SDK - Complete Documentation

This is the core module of the Groq Speech SDK, providing speech recognition, translation, and speaker diarization capabilities with a simplified single-entry-point architecture.

## Demo Usage

The SDK includes a clean, simple demo script that showcases all capabilities:

```bash
# File-based transcription (SDK handles all complexity internally)
python examples/speech_demo.py --file audio.wav

# File-based translation with diarization
python examples/speech_demo.py --file audio.wav --operation translation --diarize

# Microphone single mode
python examples/speech_demo.py --microphone-mode single

# Microphone continuous mode
python examples/speech_demo.py --microphone-mode continuous --diarize
```

**Key Benefits:**
- **54% fewer lines of code** compared to complex implementations
- **No fallback logic** - SDK handles everything internally
- **No manual audio preprocessing** - AudioProcessor handles it automatically
- **Simple API calls** - Just call `recognizer.recognize_file()` or `recognizer.translate_file()`

## Table of Contents
1. [Quick Start](#quick-start)
2. [Critical Entry Points](#critical-entry-points)
3. [Architecture Overview](#architecture-overview)
4. [Data Flows](#data-flows)
5. [Configuration System](#configuration-system)
6. [Error Handling](#error-handling)
7. [Performance](#performance)
8. [API Reference](#api-reference)
9. [Migration Guide](#migration-guide)

## Quick Start

```python
from groq_speech import SpeechRecognizer, SpeechConfig

# Basic transcription
config = SpeechConfig()
recognizer = SpeechRecognizer(config)
result = recognizer.recognize_file("audio.wav")

# Translation to English
config.enable_translation = True
recognizer = SpeechRecognizer(config)
result = recognizer.translate_file("audio.wav")

# With speaker diarization
result = recognizer.recognize_file("audio.wav", enable_diarization=True)
```

## Critical Entry Points

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

**Key Public Methods**:
- `recognize_file(audio_file, enable_diarization=True)` - Process audio files
- `translate_file(audio_file, enable_diarization=True)` - Translate audio files
- `recognize_audio_data(audio_data)` - Process raw audio data
- `translate_audio_data(audio_data)` - Translate raw audio data
- `recognize_microphone(duration=None)` - Microphone recognition
- `translate_microphone(duration=None)` - Microphone translation

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

## Architecture Overview

The SDK follows a **simplified single-entry-point architecture** with SOLID principles:

### Primary Entry Point
- **SpeechRecognizer**: Main orchestrator for all speech operations

### Service Classes (Internal)
- **DiarizationService**: Handles speaker diarization with smart grouping
- **AudioProcessor**: Manages audio processing and chunking
- **VADService**: Voice Activity Detection with multiple fallback options
- **GroqAPIClient**: Handles API communication
- **ResponseParser**: Parses API responses
- **EventManager**: Manages real-time events
- **PerformanceTracker**: Monitors performance metrics

### Configuration
- **SpeechConfig**: Speech recognition settings
- **Config**: Centralized configuration management
- **DiarizationConfig**: Diarization parameters

### SOLID Principles Implementation
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend without modifying existing code
- **Liskov Substitution**: Interfaces can be substituted
- **Interface Segregation**: Small, focused interfaces
- **Dependency Inversion**: High-level modules depend on abstractions

## Data Flows

### 1. File Processing (No Diarization)
**Command**: `python speech_demo.py --file audio.wav`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.recognize_file(enable_diarization=False)
  ↓
AudioProcessor → GroqAPIClient → ResponseParser
  ↓
SpeechRecognitionResult
  ↓
User Response (transcribed text)
```

### 2. File Processing (With Diarization)
**Command**: `python speech_demo.py --file audio.wav --diarize`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.recognize_file(enable_diarization=True)
  ↓
DiarizationService → Pyannote.audio → Speaker Detection
  ↓
Smart Grouping (24MB chunks) → GroqAPIClient (per segment)
  ↓
DiarizationResult (with speaker attribution)
  ↓
User Response (speaker-separated transcriptions)
```

### 3. Microphone Single Mode
**Command**: `python speech_demo.py --microphone-mode single`

```
User
  ↓
Microphone Input (real-time recording)
  ↓
SpeechRecognizer.recognize_microphone()
  ↓
AudioProcessor → GroqAPIClient → ResponseParser
  ↓
SpeechRecognitionResult
  ↓
User Response (transcribed text)
```

### 4. Microphone Continuous Mode
**Command**: `python speech_demo.py --microphone-mode continuous`

```
User
  ↓
Microphone Input (continuous streaming)
  ↓
SpeechRecognizer.recognize_microphone() (continuous loop)
  ↓
AudioProcessor → VADService → Audio Chunking
  ↓
For each chunk: GroqAPIClient → ResponseParser
  ↓
SpeechRecognitionResult (per chunk)
  ↓
User Response (continuous transcriptions)
```

### 5. Translation Processing
**Command**: `python speech_demo.py --file audio.wav --operation translation`

```
User
  ↓
Audio Input → SpeechRecognizer.translate_file/translate_audio_data()
  ↓
GroqAPIClient (translation mode) → ResponseParser
  ↓
SpeechRecognitionResult (translated text)
  ↓
User Response (English translation)
```

## Configuration System

### Environment Variables
```bash
# Required
GROQ_API_KEY=your_api_key_here

# Optional
HF_TOKEN=your_huggingface_token_here  # For diarization
GROQ_MODEL_ID=whisper-large-v3        # Model selection
GROQ_TEMPERATURE=0.0                  # Model temperature
```

### Configuration Classes
- **SpeechConfig**: Speech recognition settings
- **DiarizationConfig**: Diarization parameters
- **VADConfig**: Voice Activity Detection settings

### Configuration Validation
```python
from groq_speech import SpeechConfig

# Get current settings
settings = SpeechConfig.get_diarization_config()
print(f"Chunk strategy: {settings['chunk_strategy']}")
```

## Error Handling

### Exception Hierarchy
```python
from groq_speech.exceptions import (
    GroqSpeechException,      # Base exception
    ConfigurationError,       # Configuration issues
    APIError,                 # API communication errors
    AudioError,               # Audio processing errors
    DiarizationError          # Diarization-specific errors
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
- **VAD Fallback**: Multiple VAD implementations with fallback chain

## Performance

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
- **Parallel Processing**: Concurrent processing where applicable

### Thread Safety
The SDK is designed to be thread-safe:
- **Thread-safe request tracking**
- **Lock-protected result collection**
- **Concurrent request management**
- **Safe configuration access**

### Memory Management
- **Efficient audio data handling**
- **Configurable processing modes**
- **Automatic cleanup of temporary data**
- **Streaming support for large files**

## API Reference

For detailed API documentation, see:
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference

## Migration Guide

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

## Future Enhancements

### Planned Features
1. **Real-time Streaming**: Optimize for live microphone input
2. **Speaker Persistence**: Maintain speaker identity across sessions
3. **Advanced Chunking**: Intelligent audio segmentation
4. **Performance Monitoring**: Real-time pipeline metrics

### Integration Opportunities
1. **Custom Models**: Support for other speaker detection models
2. **Multi-language**: Enhanced language support
3. **Cloud Processing**: Distributed processing capabilities
4. **API Extensions**: Additional Groq API features