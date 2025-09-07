# Groq Speech SDK

A comprehensive Python SDK for Groq's AI-powered speech recognition and translation services, featuring **enhanced speaker diarization with smart grouping and 24MB chunk optimization**.

## 🚀 Quick Start

### Installation
```bash
pip install groq-speech
```

### Basic Usage
```python
from groq_speech import SpeechRecognizer, SpeechConfig

# Configure speech recognition
config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# File recognition
result = recognizer.recognize_file("audio.wav")
print(f"Recognized: {result.text}")

# Translation
result = recognizer.translate_file("audio.wav")
print(f"Translated: {result.text}")
```

### Command Line Interface
```bash
# File processing with diarization
python examples/speech_demo.py --file audio.wav --diarize

# File processing without diarization
python examples/speech_demo.py --file audio.wav

# Microphone single-shot
python examples/speech_demo.py --microphone-mode single

# Microphone continuous
python examples/speech_demo.py --microphone-mode continuous

# Translation modes
python examples/speech_demo.py --file audio.wav --operation translation
python examples/speech_demo.py --microphone-mode single --operation translation
```

## ✨ Features

### Core Capabilities
- **Real-time Speech Recognition**: High-quality transcription using Groq's AI models
- **Speech Translation**: Automatic translation to English from any language
- **Speaker Diarization**: Multi-speaker detection and separation with smart grouping
- **Smart Chunking**: 24MB-optimized audio chunking for efficient API usage
- **Voice Activity Detection**: Intelligent silence detection and audio segmentation
- **Event-driven Architecture**: Real-time callbacks and event handling

### Audio Processing
- **Microphone Input**: Real-time audio capture with configurable parameters
- **File Processing**: Support for various audio formats (WAV, MP3, etc.)
- **Audio Optimization**: Automatic resampling, chunking, and format conversion
- **Device Management**: Audio device enumeration and selection

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

## 📖 Usage Examples

### File Processing

#### Basic Recognition
```python
from groq_speech import SpeechRecognizer, SpeechConfig

config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# Simple file recognition
result = recognizer.recognize_file("audio.wav")
print(f"Recognized: {result.text}")
```

#### Recognition with Diarization
```python
# File recognition with speaker diarization
result = recognizer.recognize_file("audio.wav", enable_diarization=True)

if hasattr(result, "segments"):
    print(f"Speakers detected: {result.num_speakers}")
    for segment in result.segments:
        print(f"Speaker {segment.speaker_id}: {segment.text}")
```

#### Translation
```python
# File translation
result = recognizer.translate_file("audio.wav")
print(f"Translated: {result.text}")

# Translation with diarization
result = recognizer.translate_file("audio.wav", enable_diarization=True)
if hasattr(result, "segments"):
    for segment in result.segments:
        print(f"Speaker {segment.speaker_id}: {segment.text}")
```

### Microphone Processing

#### Single Recording
```python
# Record for 10 seconds
result = recognizer.recognize_microphone(duration=10)
print(f"Recognized: {result.text}")

# Translation from microphone
result = recognizer.translate_microphone(duration=10)
print(f"Translated: {result.text}")
```

#### Continuous Recognition
```python
# Start continuous recognition
recognizer.start_continuous_recognition()

# Handle events
def on_recognized(result):
    print(f"Recognized: {result.text}")

recognizer.add_event_handler("recognized", on_recognized)

# Stop when done
recognizer.stop_continuous_recognition()
```

### Raw Audio Data Processing

```python
import soundfile as sf

# Load audio file
audio_data, sample_rate = sf.read("audio.wav")

# Process raw audio
result = recognizer.recognize_audio_data(audio_data)
print(f"Recognized: {result.text}")

# Translate raw audio
result = recognizer.translate_audio_data(audio_data)
print(f"Translated: {result.text}")
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required
export GROQ_API_KEY="your-api-key-here"

# Optional
export HF_TOKEN="your-huggingface-token"  # For diarization
export GROQ_MODEL_ID="whisper-large-v3"  # Model selection
export GROQ_TEMPERATURE="0.0"            # Model temperature
```

### SpeechConfig
```python
from groq_speech import SpeechConfig

config = SpeechConfig()
config.api_key = "your-api-key"
config.enable_translation = True
config.translation_target_language = "en"
```

### DiarizationConfig
```python
from groq_speech import DiarizationConfig

diarization_config = DiarizationConfig()
diarization_config.max_speakers = 10
```

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

## 🔧 Command Line Interface

The SDK includes a command-line interface for testing and demonstration:

```bash
# File recognition
python examples/speech_demo.py --file audio.wav

# File recognition with diarization
python examples/speech_demo.py --file audio.wav --diarize

# File translation
python examples/speech_demo.py --file audio.wav --operation translation

# Microphone recognition
python examples/speech_demo.py --microphone-mode single

# Microphone translation
python examples/speech_demo.py --microphone-mode single --operation translation

# Continuous recognition
python examples/speech_demo.py --microphone-mode continuous
```

## 📚 Documentation

For detailed documentation, see:
- [API Reference](groq_speech/API_REFERENCE.md) - Complete API reference
- [Architecture](groq_speech/ARCHITECTURE.md) - Architecture overview
- [Data Flows](groq_speech/DATAFLOWS.md) - Data flow diagrams

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the examples in the `examples/` directory