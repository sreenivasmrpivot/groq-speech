# Groq Speech SDK - API Reference

## Table of Contents
1. [SpeechRecognizer](#speechrecognizer)
2. [Diarizer](#diarizer)
3. [Configuration Classes](#configuration-classes)
4. [Result Objects](#result-objects)
5. [Error Handling](#error-handling)

## SpeechRecognizer

**Location**: `groq_speech.speech_recognizer.SpeechRecognizer`

### Constructor
```python
SpeechRecognizer(
    speech_config: Optional[SpeechConfig] = None,
    api_key: Optional[str] = None,
    enable_diarization: bool = False,
    translation_target_language: str = "en",
    sample_rate: int = 16000,
    api_client: Optional[IAPIClient] = None,
    audio_processor: Optional[IAudioProcessor] = None,
    diarization_service: Optional[IDiarizationService] = None
)
```

**Parameters**:
- `speech_config`: Speech configuration object (optional)
- `api_key`: Groq API key for authentication (optional)
- `enable_diarization`: Whether to enable speaker diarization (default: False)
- `translation_target_language`: Target language for translation (default: "en")
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `api_client`: API client implementation (for testing)
- `audio_processor`: Audio processor implementation (for testing)
- `diarization_service`: Diarization service implementation (for testing)

### Core Methods

#### recognize_audio_data()
```python
def recognize_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000, is_translation: bool = False) -> SpeechRecognitionResult
```
**Purpose**: Recognize speech from audio data
**Parameters**:
- `audio_data`: Audio data as numpy array
- `sample_rate`: Audio sample rate (default: 16000)
- `is_translation`: Whether to use translation endpoint (default: False)
**Returns**: `SpeechRecognitionResult` with recognition data
**Usage**: Primary method for processing audio data

#### translate_audio_data()
```python
def translate_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> SpeechRecognitionResult
```
**Purpose**: Translate audio to target language
**Parameters**:
- `audio_data`: Audio data as numpy array
- `sample_rate`: Audio sample rate (default: 16000)
**Returns**: `SpeechRecognitionResult` with translated text
**Usage**: Translate audio to specified target language

#### recognize_file()
```python
def recognize_file(self, audio_file: str, enable_diarization: bool = True) -> Union[SpeechRecognitionResult, DiarizationResult]
```

**Purpose**: Process audio file for recognition
**Parameters**:
- `audio_file`: Path to audio file
- `enable_diarization`: Whether to enable speaker diarization (default: True)
**Returns**: `SpeechRecognitionResult` or `DiarizationResult` with recognition data
**Usage**: Primary method for file-based processing

#### translate_file()
```python
def translate_file(self, audio_file: str, enable_diarization: bool = True) -> Union[SpeechRecognitionResult, DiarizationResult]
```

**Purpose**: Process audio file for translation
**Parameters**:
- `audio_file`: Path to audio file
- `enable_diarization`: Whether to enable speaker diarization (default: True)
**Returns**: `SpeechRecognitionResult` or `DiarizationResult` with translated text
**Usage**: Translate audio files to target language

#### process_file()
```python
async def process_file(self, audio_file: str, enable_diarization: bool = True, is_translation: bool = False) -> Union[SpeechRecognitionResult, DiarizationResult]
```
**Purpose**: Recognize speech from audio file
**Parameters**:
- `audio_file`: Path to audio file
- `enable_diarization`: Whether to enable speaker diarization (default: True)
**Returns**: `SpeechRecognitionResult` or `DiarizationResult` with recognition data
**Usage**: Process audio files with optional diarization

#### recognize_file()
```python
async def recognize_file(self, audio_file: str, enable_diarization: bool = True) -> Union[SpeechRecognitionResult, DiarizationResult]
```
**Purpose**: Recognize speech from audio file (convenience method)
**Parameters**:
- `audio_file`: Path to audio file
- `enable_diarization`: Whether to enable speaker diarization (default: True)
**Returns**: `SpeechRecognitionResult` or `DiarizationResult` with recognition data
**Usage**: Convenience method for basic recognition

#### translate_file()
```python
async def translate_file(self, audio_file: str, enable_diarization: bool = True) -> Union[SpeechRecognitionResult, DiarizationResult]
```
**Purpose**: Translate audio file to target language
**Parameters**:
- `audio_file`: Path to audio file
- `enable_diarization`: Whether to enable speaker diarization (default: True)
**Returns**: `SpeechRecognitionResult` or `DiarizationResult` with translated text
**Usage**: Translate audio files with optional diarization

#### start_continuous_recognition()
```python
def start_continuous_recognition(self) -> None
```
**Purpose**: Start continuous speech recognition mode
**Returns**: None
**Usage**: Initiates real-time continuous recognition with event firing

#### stop_continuous_recognition()
```python
def stop_continuous_recognition(self) -> None
```
**Purpose**: Stop continuous speech recognition
**Returns**: None
**Usage**: Safely stops continuous recognition and cleans up resources
## Diarizer

**Location**: `groq_speech.speaker_diarization.Diarizer`

### Constructor
```python
Diarizer(config: Optional[DiarizationConfig] = None)
```

**Parameters**:
- `config`: Diarization configuration parameters (optional)

### Core Methods

#### diarize()
```python
def diarize(self, audio_file: str, mode: str, speech_recognizer=None) -> DiarizationResult
```
**Purpose**: Perform speaker diarization on audio file
**Parameters**:
- `audio_file`: Path to audio file
- `mode`: Processing mode ("transcription" or "translation")
- `speech_recognizer`: Speech recognizer instance (optional)
**Returns**: `DiarizationResult` with speaker segments and transcriptions
**Usage**: Identify speakers and transcribe their speech

## Configuration Classes

### SpeechConfig
**Location**: `groq_speech.speech_config.SpeechConfig`

### DiarizationConfig
**Location**: `groq_speech.speaker_diarization.DiarizationConfig`

### Config
**Location**: `groq_speech.config.Config`

## Result Objects

### SpeechRecognitionResult
**Location**: `groq_speech.speech_recognizer.SpeechRecognitionResult`

### DiarizationResult
**Location**: `groq_speech.speaker_diarization.DiarizationResult`

### SpeakerSegment
**Location**: `groq_speech.speaker_diarization.SpeakerSegment`

## Error Handling

### ResultReason
**Location**: `groq_speech.result_reason.ResultReason`

### CancellationReason
**Location**: `groq_speech.result_reason.CancellationReason`
