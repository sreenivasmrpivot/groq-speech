# Groq Speech SDK - API Reference

## Table of Contents
1. [SpeechRecognizer](#speechrecognizer)
2. [AudioConfig](#audioconfig)
3. [SpeakerDiarizer](#speakerdiarizer)
4. [EnhancedDiarizer](#enhanceddiarizer)
5. [Configuration Classes](#configuration-classes)
6. [Result Objects](#result-objects)
7. [Error Handling](#error-handling)

## SpeechRecognizer

**Location**: `groq_speech.speech_recognizer.SpeechRecognizer`

### Constructor
```python
SpeechRecognizer(speech_config: SpeechConfig, audio_config: Optional[AudioConfig] = None)
```

**Parameters**:
- `speech_config`: Speech configuration including API keys and model settings
- `audio_config`: Audio configuration for microphone and file processing (optional)

### Core Methods

#### recognize_once_async()
```python
def recognize_once_async(self) -> SpeechRecognitionResult
```
**Purpose**: Perform single-shot speech recognition from microphone
**Returns**: `SpeechRecognitionResult` with recognition data
**Usage**: Primary method for one-time speech recognition

#### start_continuous_recognition()
```python
def start_continuous_recognition(self)
```
**Purpose**: Start continuous speech recognition mode
**Returns**: None
**Usage**: Initiates real-time continuous recognition with event firing

#### stop_continuous_recognition()
```python
def stop_continuous_recognition(self)
```
**Purpose**: Stop continuous speech recognition
**Returns**: None
**Usage**: Safely stops continuous recognition and cleans up resources

#### recognize_audio_data()
```python
def recognize_audio_data(self, audio_data: np.ndarray, is_translation: bool = False) -> SpeechRecognitionResult
```
**Parameters**:
- `audio_data`: Audio data as numpy array
- `is_translation`: Whether to use translation endpoint (default: False)
**Returns**: `SpeechRecognitionResult` with recognition data
**Usage**: Process pre-recorded audio data

#### translate_audio_data()
```python
def translate_audio_data(self, audio_data: np.ndarray) -> SpeechRecognitionResult
```
**Parameters**:
- `audio_data`: Audio data as numpy array
**Returns**: `SpeechRecognitionResult` with English translation
**Usage**: Translate speech in any language to English

### Event System

#### connect()
```python
def connect(self, event_name: str, handler: Callable)
```
**Parameters**:
- `event_name`: Event name ("session_started", "recognizing", "recognized", "session_stopped", "canceled")
- `handler`: Callback function to handle the event
**Usage**: Register event handlers for real-time processing

#### disconnect()
```python
def disconnect(self, event_name: str, handler: Callable)
```
**Purpose**: Unregister event handlers

### Performance Methods

#### get_performance_stats()
```python
def get_performance_stats(self) -> dict
```
**Returns**: Dictionary with performance metrics
**Usage**: Get comprehensive performance statistics

#### is_recognizing()
```python
def is_recognizing(self) -> bool
```
**Returns**: Boolean indicating if recognition is active
**Usage**: Check recognition state

## AudioConfig

**Location**: `groq_speech.audio_config.AudioConfig`

### Constructor
```python
AudioConfig(
    device_id: Optional[int] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 1024,
    filename: Optional[str] = None
)
```

**Parameters**:
- `device_id`: Audio device ID (optional, auto-detect if None)
- `sample_rate`: Audio sample rate (default: 16000)
- `channels`: Number of audio channels (default: 1)
- `chunk_size`: Audio chunk size for streaming (default: 1024)
- `filename`: Audio file path for file-based processing (optional)

### Core Methods

#### read_audio_chunk()
```python
def read_audio_chunk(self, size: int) -> bytes
```
**Parameters**:
- `size`: Number of bytes to read
**Returns**: Audio data as bytes
**Usage**: Read audio chunk from microphone

#### get_file_audio_data()
```python
def get_file_audio_data(self) -> np.ndarray
```
**Returns**: Audio data as numpy array
**Usage**: Load audio data from file

#### start_microphone_stream()
```python
def start_microphone_stream(self)
```
**Purpose**: Initialize microphone stream
**Usage**: Start audio capture from microphone

#### stop_microphone_stream()
```python
def stop_microphone_stream(self)
```
**Purpose**: Stop microphone stream
**Usage**: Stop audio capture and cleanup resources

### Context Manager
```python
with AudioConfig() as audio:
    chunk = audio.read_audio_chunk(1024)
```

## SpeakerDiarizer

**Location**: `groq_speech.speaker_diarization.SpeakerDiarizer`

### Constructor
```python
SpeakerDiarizer(
    config: Optional[DiarizationConfig] = None,
    speech_config: Optional[SpeechConfig] = None
)
```

**Parameters**:
- `config`: Diarization configuration (optional)
- `speech_config`: Speech configuration for transcription (optional)

### Core Methods

#### diarize_audio()
```python
def diarize_audio(
    self, 
    audio_data: np.ndarray, 
    sample_rate: int = 16000,
    diarization_config: Optional[DiarizationConfig] = None
) -> DiarizationResult
```
**Parameters**:
- `audio_data`: Audio data as numpy array
- `sample_rate`: Sample rate of audio data (default: 16000)
- `diarization_config`: Diarization configuration (optional)
**Returns**: `DiarizationResult` with speaker segments
**Usage**: Perform speaker diarization on audio data

#### diarize_with_accurate_transcription()
```python
def diarize_with_accurate_transcription(
    self, 
    audio_file: str, 
    mode: str, 
    speech_recognizer: Optional[SpeechRecognizer] = None
) -> DiarizationResult
```
**Parameters**:
- `audio_file`: Path to audio file
- `mode`: Processing mode ("transcription" or "translation")
- `speech_recognizer`: Speech recognizer instance (optional)
**Returns**: `DiarizationResult` with accurate transcriptions
**Usage**: Diarization with high-quality transcription

## EnhancedDiarizer

**Location**: `groq_speech.enhanced_diarization.EnhancedDiarizer`

### Constructor
```python
EnhancedDiarizer(
    config: Optional[EnhancedDiarizationConfig] = None,
    speech_config: Optional[SpeechConfig] = None
)
```

### Core Methods

#### diarize_with_enhanced_flow()
```python
def diarize_with_enhanced_flow(
    self, 
    audio_file: str, 
    mode: str, 
    speech_recognizer: Optional[SpeechRecognizer] = None
) -> DiarizationResult
```
**Parameters**:
- `audio_file`: Path to audio file
- `mode`: Processing mode ("transcription" or "translation")
- `speech_recognizer`: Speech recognizer instance (optional)
**Returns**: `DiarizationResult` with optimized processing
**Usage**: Advanced diarization with parallel processing

#### get_performance_stats()
```python
def get_performance_stats(self) -> dict
```
**Returns**: Dictionary with enhanced processing statistics
**Usage**: Get performance metrics for enhanced diarization

## Configuration Classes

### SpeechConfig
**Location**: `groq_speech.speech_config.SpeechConfig`

```python
SpeechConfig(
    api_key: Optional[str] = None,
    model: str = "whisper-large-v3-turbo",
    enable_translation: bool = False,
    translation_target: str = "en"
)
```

### DiarizationConfig
**Location**: `groq_speech.speaker_diarization.DiarizationConfig`

```python
DiarizationConfig(
    min_speakers: int = 1,
    max_speakers: int = 10,
    min_segment_duration: float = 0.5,
    max_segment_duration: float = 30.0
)
```

### EnhancedDiarizationConfig
**Location**: `groq_speech.enhanced_diarization.EnhancedDiarizationConfig`

```python
EnhancedDiarizationConfig(
    max_group_size_mb: float = 25.0,
    max_parallel_requests: int = 5,
    grouping_strategy: str = "speaker_continuity"
)
```

## Result Objects

### SpeechRecognitionResult
**Location**: `groq_speech.speech_recognizer.SpeechRecognitionResult`

**Properties**:
- `text`: Recognized text
- `reason`: Result reason (ResultReason enum)
- `confidence`: Confidence score (0.0 to 1.0)
- `language`: Detected language
- `timestamps`: Word-level timestamps
- `timing_metrics`: Performance metrics

### DiarizationResult
**Location**: `groq_speech.speaker_diarization.DiarizationResult`

**Properties**:
- `segments`: List of SpeakerSegment objects
- `speaker_mapping`: Dictionary mapping speaker IDs to names
- `total_duration`: Total audio duration
- `num_speakers`: Number of detected speakers
- `overall_confidence`: Overall confidence score

### SpeakerSegment
**Location**: `groq_speech.speaker_diarization.SpeakerSegment`

**Properties**:
- `start_time`: Segment start time
- `end_time`: Segment end time
- `speaker_id`: Speaker identifier
- `text`: Transcribed text for segment
- `confidence`: Transcription confidence

## Error Handling

### ResultReason
**Location**: `groq_speech.result_reason.ResultReason`

**Values**:
- `RecognizedSpeech`: Successful recognition
- `NoMatch`: No speech detected
- `Canceled`: Recognition was canceled

### CancellationReason
**Location**: `groq_speech.result_reason.CancellationReason`

**Values**:
- `Error`: General error
- `EndOfStream`: End of audio stream
- `CancelledByUser`: User canceled operation

### Exceptions
**Location**: `groq_speech.exceptions`

- `DiarizationError`: Diarization-specific errors
- `ConfigurationError`: Configuration-related errors
- `AudioError`: Audio processing errors

## Usage Examples

### Basic Recognition
```python
from groq_speech import SpeechConfig, SpeechRecognizer

config = SpeechConfig()
recognizer = SpeechRecognizer(config)
result = recognizer.recognize_once_async()
print(f"Recognized: {result.text}")
```

### Continuous Recognition
```python
def on_recognized(event_data):
    print(f"Recognized: {event_data.text}")

recognizer.connect("recognized", on_recognized)
recognizer.start_continuous_recognition()
# ... processing ...
recognizer.stop_continuous_recognition()
```

### File Processing
```python
import soundfile as sf

audio_data, sample_rate = sf.read("audio.wav")
result = recognizer.recognize_audio_data(audio_data)
print(f"Transcription: {result.text}")
```

### Speaker Diarization
```python
from groq_speech import SpeakerDiarizer

diarizer = SpeakerDiarizer()
result = diarizer.diarize_audio(audio_data)
for segment in result.segments:
    print(f"Speaker {segment.speaker_id}: {segment.text}")
```

### Translation
```python
result = recognizer.translate_audio_data(audio_data)
print(f"Translation: {result.text}")
```
