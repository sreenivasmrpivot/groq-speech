# Groq Speech SDK

A Python SDK for Groq's speech services, providing real-time speech-to-text capabilities. This SDK mirrors the Azure AI Speech SDK functionality but uses Groq's API for speech recognition.

## Features

- **Real-time Speech Recognition**: Convert speech to text in real-time
- **Single-shot Recognition**: Recognize a single utterance
- **Continuous Recognition**: Recognize speech continuously with event handling
- **File-based Recognition**: Recognize speech from audio files
- **Language Support**: Support for multiple languages and language identification
- **Semantic Segmentation**: Advanced segmentation for better recognition results
- **Error Handling**: Comprehensive error handling and cancellation support
- **Event-driven Architecture**: Event-based recognition with callbacks

## Installation

### Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd groq-speech
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings:
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Update the `.env` file with your Groq API key:
```env
GROQ_API_KEY=your-groq-api-key-here
```

You can get your API key from [Groq Console](https://console.groq.com/).

### Docker Deployment

For production deployment, use Docker:

```bash
# Build and run with Docker Compose
docker-compose -f deployment/docker/docker-compose.yml up -d

# Access the API server
curl http://localhost:8000/health

# Access the web demo
open http://localhost:5000
```

### API Server

Run the FastAPI server for production use:

```bash
# Start the API server
python -m api.server

# Or with uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The API server provides:
- REST API endpoints
- WebSocket real-time recognition
- Interactive documentation at `/docs`
- Health monitoring at `/health`

## Quick Start

### Basic Speech Recognition

```python
import groq_speech as speechsdk

def from_microphone():
    # Uses settings from .env file
    speech_config = speechsdk.SpeechConfig()
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async()
    print(speech_recognition_result.text)

from_microphone()
```

### Recognition from File

```python
import groq_speech as speechsdk

def from_file():
    # Uses settings from .env file
    speech_config = speechsdk.SpeechConfig()
    audio_config = speechsdk.AudioConfig(filename="your_file_name.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async()
    print(speech_recognition_result.text)

from_file()
```

### Error Handling

```python
if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized: {}".format(speech_recognition_result.text))
elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_recognition_result.cancellation_details
    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))
```

## Configuration

The SDK uses a `.env` file for configuration. Create a `.env` file in the project root with the following settings:

```env
# Required: Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_BASE_URL=https://api.groq.com/openai/v1

# Speech Recognition Settings
DEFAULT_LANGUAGE=en-US
DEFAULT_SAMPLE_RATE=16000
DEFAULT_CHANNELS=1
DEFAULT_CHUNK_SIZE=1024

# Audio Device Settings
DEFAULT_DEVICE_INDEX=None
DEFAULT_FRAMES_PER_BUFFER=1024

# Recognition Timeouts
DEFAULT_TIMEOUT=30
DEFAULT_PHRASE_TIMEOUT=3
DEFAULT_SILENCE_TIMEOUT=1

# Optional: Advanced Features
ENABLE_SEMANTIC_SEGMENTATION=true
ENABLE_LANGUAGE_IDENTIFICATION=true
```

### Configuration Class

You can also access configuration programmatically:

```python
from groq_speech import Config, get_config

# Access configuration values
api_key = Config.get_api_key()
language = Config.DEFAULT_LANGUAGE
sample_rate = Config.DEFAULT_SAMPLE_RATE

# Validate API key
if Config.validate_api_key():
    print("API key is set")
```

## API Reference

### SpeechConfig

Configuration class for Groq Speech services.

```python
# Uses .env settings automatically
speech_config = SpeechConfig()

# Or specify manually
speech_config = SpeechConfig(
    api_key="your-api-key",
    region="us-east-1",  # optional
    endpoint="custom-endpoint",  # optional
    host="custom-host",  # optional
    authorization_token="token"  # alternative to api_key
)
```

#### Methods

- `set_property(property_id, value)`: Set a configuration property
- `get_property(property_id)`: Get a configuration property
- `set_speech_recognition_language(language)`: Set recognition language
- `set_endpoint_id(endpoint_id)`: Set custom endpoint ID
- `validate()`: Validate configuration

### AudioConfig

Configuration class for audio input/output.

```python
audio_config = AudioConfig(
    filename="audio.wav",  # optional, for file input
    device_id=0,  # optional, microphone device ID
    sample_rate=16000,  # audio sample rate
    channels=1,  # number of audio channels
    format_type="wav"  # audio format
)
```

#### Methods

- `get_audio_devices()`: Get list of available audio devices
- `start_microphone_stream()`: Start microphone audio stream
- `stop_microphone_stream()`: Stop microphone audio stream
- `read_audio_chunk(chunk_size)`: Read audio chunk from microphone
- `get_file_audio_data()`: Get audio data from file
- `create_audio_chunks(chunk_duration)`: Create audio chunks for streaming

### SpeechRecognizer

Main class for speech recognition.

```python
speech_recognizer = SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config  # optional
)
```

#### Methods

- `recognize_once_async()`: Perform single-shot recognition
- `recognize_once()`: Synchronous single-shot recognition
- `start_continuous_recognition()`: Start continuous recognition
- `stop_continuous_recognition()`: Stop continuous recognition
- `connect(event_type, handler)`: Connect event handler
- `is_recognizing()`: Check if recognition is active

#### Event Types

- `recognizing`: Intermediate recognition results
- `recognized`: Final recognition results
- `session_started`: Session started event
- `session_stopped`: Session stopped event
- `canceled`: Recognition canceled event

### SpeechRecognitionResult

Result of a speech recognition operation.

#### Properties

- `text`: Recognized text
- `reason`: Result reason (ResultReason enum)
- `confidence`: Confidence score (0.0 to 1.0)
- `language`: Detected language
- `cancellation_details`: Details if canceled
- `no_match_details`: Details if no speech detected

## Advanced Features

### Continuous Recognition

```python
import groq_speech as speechsdk
import time

def continuous_recognition():
    speech_config = speechsdk.SpeechConfig(api_key="YourGroqApiKey")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    
    done = False
    
    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
    
    def recognized_cb(evt):
        print('RECOGNIZED: {}'.format(evt))
        if evt.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Final result: {}".format(evt.text))
    
    speech_recognizer.connect('recognized', recognized_cb)
    speech_recognizer.connect('session_stopped', stop_cb)
    
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
```

### Language Identification

```python
speech_config.set_property(speechsdk.PropertyId.Speech_Recognition_EnableLanguageIdentification, "true")
speech_config.set_property(speechsdk.PropertyId.Speech_Recognition_LanguageIdentificationMode, "Continuous")
```

### Semantic Segmentation

```python
speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationStrategy, "Semantic")
```

### Custom Endpoints

```python
speech_config = speechsdk.SpeechConfig(api_key="YourGroqApiKey")
speech_config.endpoint_id = "YourEndpointId"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
```

## Examples & Demos

The SDK includes comprehensive real-world demonstration applications:

### Desktop Applications

- **Voice Assistant Demo** (`examples/voice_assistant_demo.py`): Interactive GUI application with command processing
- **Transcription Workbench** (`examples/transcription_workbench.py`): Professional transcription tool with analysis

### Web Applications

- **Web Demo** (`examples/web_demo.py`): Modern web interface with real-time statistics

### Basic Examples

- `basic_recognition.py`: Basic single-shot recognition examples
- `continuous_recognition.py`: Continuous recognition with event handling

### Running Demos

```bash
# Voice Assistant (Desktop GUI)
python examples/voice_assistant_demo.py

# Transcription Workbench (Desktop GUI)
python examples/transcription_workbench.py

# Web Demo (Browser-based)
python examples/web_demo.py
```

## Supported Languages

The SDK supports all languages supported by Groq's Whisper models. Common language codes include:

- `en-US`: English (US)
- `de-DE`: German
- `fr-FR`: French
- `es-ES`: Spanish
- `it-IT`: Italian
- `pt-BR`: Portuguese (Brazil)
- `ja-JP`: Japanese
- `ko-KR`: Korean
- `zh-CN`: Chinese (Simplified)
- `ru-RU`: Russian

## Error Handling

The SDK provides comprehensive error handling through the `ResultReason` and `CancellationReason` enums:

### ResultReason
- `RecognizedSpeech`: Successful recognition
- `NoMatch`: No speech detected
- `Canceled`: Recognition was canceled

### CancellationReason
- `Error`: Error occurred during recognition
- `EndOfStream`: Recognition ended normally
- `Timeout`: Recognition timed out
- `NetworkError`: Network-related error
- `ServiceError`: Service-related error
- `InvalidAudio`: Invalid audio input
- `LanguageNotSupported`: Language not supported

## Configuration Properties

Use `PropertyId` enum to set various configuration properties:

```python
speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationStrategy, "Semantic")
speech_config.set_property(speechsdk.PropertyId.Speech_Recognition_EnableDictation, "true")
speech_config.set_property(speechsdk.PropertyId.Speech_Recognition_EnableWordLevelTimestamps, "true")
```

## Audio Requirements

- **Sample Rate**: 16 kHz recommended
- **Channels**: Mono (1 channel) recommended
- **Format**: WAV, MP3, FLAC, OGG supported
- **Bit Depth**: 16-bit recommended

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Groq API key is set correctly
2. **Audio Device Issues**: Check available audio devices with `AudioConfig().get_audio_devices()`
3. **Network Issues**: Ensure stable internet connection for API calls
4. **Audio Format Issues**: Use supported audio formats and proper sample rates

### Debug Mode

Enable debug logging:

```python
speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "debug.log")
speech_config.set_property(speechsdk.PropertyId.Speech_ServiceConnection_LogFilename, "service.log")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the examples
- Open an issue on GitHub

## Acknowledgments

This SDK is inspired by the Azure AI Speech SDK and adapted for use with Groq's speech services. 