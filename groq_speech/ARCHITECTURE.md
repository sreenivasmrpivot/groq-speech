# Groq Speech SDK - Architecture Overview

## Table of Contents
1. [Critical Entry Points](#critical-entry-points)
2. [Core Components](#core-components)
3. [Data Flows](#data-flows)
4. [Configuration System](#configuration-system)
5. [Error Handling](#error-handling)

## Critical Entry Points

The Groq Speech SDK provides a **single primary entry point** for external consumers:

### 1. SpeechRecognizer (Primary Entry Point)
**Location**: `groq_speech.speech_recognizer.SpeechRecognizer`
**Purpose**: Main orchestrator for all speech recognition operations

**Key Public Methods**:
- `recognize_file(audio_file, enable_diarization=True)` - Process audio files
- `translate_file(audio_file, enable_diarization=True)` - Translate audio files
- `recognize_audio_data(audio_data)` - Process raw audio data
- `translate_audio_data(audio_data)` - Translate raw audio data
- `recognize_microphone(duration=None)` - Microphone recognition
- `translate_microphone(duration=None)` - Microphone translation

**Key Private Methods** (Internal Use):
- `_process_audio_with_diarization()` - Handle diarization pipeline
- `_recognize_with_diarization()` - Diarization-based recognition
- `_translate_with_diarization()` - Diarization-based translation

### 2. Config (Configuration Management)
**Location**: `groq_speech.config.Config`
**Purpose**: Centralized configuration management

**Key Methods**:
- `get_api_key()` - Get API key from environment
- `get_model_config()` - Get model configuration
- `get_chunking_config()` - Get audio chunking settings

### 3. Result Objects (Data Access)
**Location**: `groq_speech.speech_recognizer.SpeechRecognitionResult`
**Purpose**: Structured result data access

**Key Properties**:
- `text` - Recognized text
- `confidence` - Recognition confidence
- `timestamps` - Word-level timestamps
- `reason` - Result status

## Core Components

### SpeechRecognizer
The main class that orchestrates all speech recognition operations. It follows SOLID principles with clear separation of concerns:

- **Single Responsibility**: Handles speech recognition and translation
- **Open/Closed**: Extensible through service injection
- **Liskov Substitution**: Consistent interface for all operations
- **Interface Segregation**: Clean, focused public API
- **Dependency Inversion**: Depends on abstractions, not concretions

### Service Classes (Internal)
- **DiarizationService**: Handles speaker diarization operations
- **AudioProcessor**: Manages audio processing and chunking
- **GroqAPIClient**: Handles API communication
- **ResponseParser**: Parses API responses
- **EventManager**: Manages real-time events
- **PerformanceTracker**: Monitors performance metrics

### Diarization System
- **Diarizer**: Main diarization class (consolidated from EnhancedDiarizer)
- **SpeakerSegment**: Individual speaker segment data
- **DiarizationResult**: Complete diarization result with speaker attribution

## Data Flows

### 1. File Processing (No Diarization)
```
User → File Input → SpeechRecognizer.recognize_file(enable_diarization=False)
     → AudioProcessor → GroqAPIClient → ResponseParser → SpeechRecognitionResult
```

### 2. File Processing (With Diarization)
```
User → File Input → SpeechRecognizer.recognize_file(enable_diarization=True)
     → DiarizationService → Pyannote.audio → Speaker Detection
     → Audio Chunking → GroqAPIClient (per segment) → ResponseParser
     → DiarizationResult (with speaker attribution)
```

### 3. Microphone Processing
```
User → Microphone → SpeechRecognizer.recognize_microphone()
     → AudioProcessor → GroqAPIClient → ResponseParser → SpeechRecognitionResult
```

### 4. Translation Processing
```
User → Audio Input → SpeechRecognizer.translate_file/translate_audio_data()
     → GroqAPIClient (translation mode) → ResponseParser → SpeechRecognitionResult
```

## Configuration System

### Environment Variables
- `GROQ_API_KEY`: Required API key
- `HF_TOKEN`: Optional, for speaker diarization
- `GROQ_MODEL_ID`: Optional, model selection
- `GROQ_TEMPERATURE`: Optional, model temperature

### Configuration Classes
- **SpeechConfig**: Speech recognition settings
- **DiarizationConfig**: Diarization parameters
- **Config**: Centralized configuration management

## Error Handling

### Exception Hierarchy
- **GroqSpeechException**: Base exception class
- **ConfigurationError**: Configuration-related errors
- **APIError**: API communication errors
- **AudioError**: Audio processing errors
- **DiarizationError**: Diarization-specific errors

### Fallback Mechanisms
- **Diarization Fallback**: Falls back to basic transcription if diarization fails
- **API Retry**: Automatic retry for transient API errors
- **Audio Processing**: Graceful handling of audio format issues

## Performance Optimizations

### Big O Complexity
- **File Processing**: O(n) where n is audio length
- **Diarization**: O(n log n) due to Pyannote.audio processing
- **API Calls**: O(1) per chunk, O(k) total where k is number of chunks
- **Memory Usage**: O(1) for streaming, O(n) for file processing

### Parallel Processing
- **Diarization**: Parallel processing of speaker segments
- **API Calls**: Concurrent processing where possible
- **Audio Chunking**: Optimized chunking strategies

## Usage Examples

### Basic File Recognition
```python
from groq_speech import SpeechRecognizer, SpeechConfig

config = SpeechConfig()
recognizer = SpeechRecognizer(config)
result = recognizer.recognize_file("audio.wav")
print(f"Recognized: {result.text}")
```

### File Recognition with Diarization
```python
result = recognizer.recognize_file("audio.wav", enable_diarization=True)
for segment in result.segments:
    print(f"Speaker {segment.speaker_id}: {segment.text}")
```

### Translation
```python
result = recognizer.translate_file("audio.wav", enable_diarization=True)
print(f"Translated: {result.text}")
```

### Microphone Recognition
```python
result = recognizer.recognize_microphone(duration=10)  # 10 seconds
print(f"Recognized: {result.text}")
```

## Migration from Previous Versions

### Removed Components
- **AudioConfig**: Functionality moved to SpeechRecognizer
- **EnhancedDiarizer**: Consolidated into Diarizer
- **SpeakerDiarizer**: Consolidated into Diarizer
- **Multiple Entry Points**: Simplified to single SpeechRecognizer

### Breaking Changes
- Constructor parameters simplified
- Method signatures standardized
- Diarization enabled by default for files
- Consistent `enable_diarization` parameter across all methods