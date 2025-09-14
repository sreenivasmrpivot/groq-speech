# Groq Speech SDK

A comprehensive Python SDK for Groq's AI-powered speech recognition and translation services, featuring **real-time processing, speaker diarization, and web interface support**.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd groq-speech

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from groq_speech import SpeechRecognizer, SpeechConfig

# Configure speech recognition
config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# File recognition (async)
result = await recognizer.recognize_file("audio.wav")
print(f"Recognized: {result.text}")

# Translation (async)
result = await recognizer.translate_file("audio.wav")
print(f"Translated: {result.text}")

# With speaker diarization (async)
result = await recognizer.recognize_file("audio.wav", enable_diarization=True)
if hasattr(result, 'segments'):
    for segment in result.segments:
        print(f"Speaker {segment.speaker_id}: {segment.text}")
```

### Command Line Interface
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

## ✨ Features

### Core Capabilities
- **Real-time Speech Recognition**: High-quality transcription using Groq's AI models
- **Speech Translation**: Automatic translation to English from any language
- **Speaker Diarization**: Multi-speaker detection and separation using Pyannote.audio
- **Voice Activity Detection**: Intelligent silence detection and audio segmentation
- **Web Interface**: Modern React-based UI for easy testing and demonstration
- **REST API**: FastAPI backend with both REST and WebSocket endpoints

### Audio Processing
- **Microphone Input**: Real-time audio capture with configurable parameters
- **File Processing**: Support for various audio formats (WAV, MP3, etc.)
- **Audio Optimization**: Automatic resampling, chunking, and format conversion
- **Continuous Processing**: Real-time streaming with intelligent chunking

### Web Interface
- **Modern UI**: React-based interface with real-time feedback
- **Multiple Modes**: File upload, microphone recording, continuous processing
- **Real-time Results**: WebSocket-based streaming for continuous recognition
- **Performance Metrics**: Live timing and performance monitoring

## 🏗️ Architecture

The SDK follows a **3-layer architecture with parallel client interfaces**:

### Layer 1: Core SDK (`groq_speech/`)
- **SpeechRecognizer**: Main orchestrator for all speech operations
- **SpeakerDiarization**: Speaker diarization using Pyannote.audio
- **VADService**: Voice Activity Detection and audio chunking
- **SpeechConfig**: Configuration management

### Layer 2: Client Interfaces (Parallel)
- **CLI Client** (`speech_demo.py`): Command-line interface
- **API Client** (`api/server.py`): FastAPI REST + WebSocket server

### Layer 3: Web Interface (`groq-speech-ui/`)
- **React Frontend**: Modern web interface with real-time processing
- **WebSocket Integration**: Real-time streaming for continuous recognition

### Key Design Principles
- **Single Responsibility**: Each component has one clear purpose
- **Dependency Injection**: Services are injected for better testability
- **Event-Driven**: Real-time callbacks and event handling
- **SOLID Principles**: Clean, maintainable, and extensible code

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
# Record and process audio data
import pyaudio
import numpy as np

# Record audio (example with PyAudio)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True)
audio_data = stream.read(16000)  # 1 second
stream.close()
p.terminate()

# Convert to numpy array
audio_array = np.frombuffer(audio_data, dtype=np.float32)

# Process with recognizer
result = recognizer.recognize_audio_data(audio_array)
print(f"Recognized: {result.text}")

# Translation
result = recognizer.recognize_audio_data(audio_array, is_translation=True)
print(f"Translated: {result.text}")
```

#### Continuous Recognition (Web Interface)
The web interface provides continuous recognition through WebSocket connections:
- Real-time audio streaming
- Automatic chunking and processing
- Live results display
- Performance metrics

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

## 🌐 Web Interface

### Quick Start
```bash
# Start the backend API server
python -m api.server

# In another terminal, start the frontend
cd examples/groq-speech-ui
npm install
npm run dev
```

### Features
- **File Upload**: Drag and drop audio files for processing
- **Microphone Recording**: Real-time audio capture and processing
- **Continuous Mode**: Stream processing with WebSocket
- **Speaker Diarization**: Visual speaker separation and attribution
- **Performance Metrics**: Real-time timing and performance data
- **Multiple Languages**: Support for transcription and translation

### API Endpoints
- `POST /api/v1/recognize` - File transcription
- `POST /api/v1/translate` - File translation
- `WebSocket /ws/recognize` - Real-time processing
- `GET /health` - Health check

## ⚙️ Configuration

### Environment Variables
```bash
# Required
export GROQ_API_KEY="your-api-key-here"

# Optional (for diarization)
export HF_TOKEN="your-huggingface-token"
```

### SpeechConfig
```python
from groq_speech import SpeechConfig

config = SpeechConfig()
config.api_key = "your-api-key"
config.enable_translation = True
config.set_translation_target_language("en")
```

## 📊 Performance

### Processing Modes
- **File Processing**: Batch processing with automatic chunking
- **Real-time Processing**: WebSocket-based streaming with low latency
- **Diarization**: Pyannote.audio-based speaker detection and separation
- **Voice Activity Detection**: Intelligent silence detection and audio segmentation

### Optimization Features
- **Intelligent Chunking**: Automatic audio segmentation for optimal processing
- **Voice Activity Detection**: Skip silent segments to improve efficiency
- **WebSocket Streaming**: Real-time processing with minimal latency
- **Error Handling**: Automatic retry and graceful degradation

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

### Architecture & Technical Details
- [Architecture Analysis](ARCHITECTURE_ANALYSIS.md) - Complete system architecture with Mermaid diagrams
- [Code Analysis](CODE_ANALYSIS.md) - Detailed code analysis and implementation patterns
- [API Reference](groq_speech/API_REFERENCE.md) - Complete API reference

### Web Interface
- [Frontend README](examples/groq-speech-ui/README.md) - Web interface documentation
- [Backend Setup](examples/groq-speech-ui/BACKEND_SETUP.md) - Backend configuration guide

### Development
- [Contributing Guide](CONTRIBUTING.md) - Development guidelines and standards
- [Changelog](CHANGELOG.md) - Version history and changes

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