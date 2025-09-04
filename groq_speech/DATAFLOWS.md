# Groq Speech SDK - Data Flows

## Overview

This document describes the data flows for the Groq Speech SDK, showing how data moves from user input through the system to final output. All flows start with the user and end with a response back to the user.

## Flow 1: File Processing (No Diarization)

**Command**: `python speech_demo.py --file audio.wav`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.recognize_file(enable_diarization=False)
  ↓
AudioProcessor.load_audio_file()
  ↓
GroqAPIClient.transcribe()
  ↓
ResponseParser.parse_transcription_response()
  ↓
SpeechRecognitionResult
  ↓
User Response (transcribed text)
```

**Key Points**:
- Direct Groq API processing
- No speaker diarization
- Single transcription result
- Fastest processing path

## Flow 2: File Processing (With Diarization)

**Command**: `python speech_demo.py --file audio.wav --diarize`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.recognize_file(enable_diarization=True)
  ↓
DiarizationService.diarize_file()
  ↓
Diarizer.diarize() → Pyannote.audio → Speaker Detection
  ↓
Audio Chunking → Speaker-specific segments
  ↓
For each segment:
  GroqAPIClient.transcribe() → ResponseParser
  ↓
DiarizationResult (with speaker attribution)
  ↓
User Response (speaker-separated transcriptions)
```

**Key Points**:
- Pyannote.audio for speaker detection
- Parallel processing of segments
- Speaker attribution for each segment
- More detailed output

## Flow 3: Microphone Single Mode

**Command**: `python speech_demo.py --microphone-mode single`

```
User
  ↓
Microphone Input (real-time recording)
  ↓
SpeechRecognizer.recognize_microphone()
  ↓
AudioProcessor.record_audio()
  ↓
GroqAPIClient.transcribe()
  ↓
ResponseParser.parse_transcription_response()
  ↓
SpeechRecognitionResult
  ↓
User Response (transcribed text)
```

**Key Points**:
- Real-time audio capture
- Single recording session
- Direct transcription
- No diarization by default

## Flow 4: Microphone Continuous Mode

**Command**: `python speech_demo.py --microphone-mode continuous`

```
User
  ↓
Microphone Input (continuous streaming)
  ↓
SpeechRecognizer.recognize_microphone() (continuous loop)
  ↓
AudioProcessor.stream_audio() → Audio Chunking
  ↓
For each chunk:
  GroqAPIClient.transcribe() → ResponseParser
  ↓
SpeechRecognitionResult (per chunk)
  ↓
User Response (continuous transcriptions)
```

**Key Points**:
- Continuous audio streaming
- Chunked processing
- Real-time output
- Runs until Ctrl+C

## Flow 5: Translation (File)

**Command**: `python speech_demo.py --file audio.wav --operation translation`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.translate_file(enable_diarization=False)
  ↓
AudioProcessor.load_audio_file()
  ↓
GroqAPIClient.translate()
  ↓
ResponseParser.parse_translation_response()
  ↓
SpeechRecognitionResult (translated text)
  ↓
User Response (English translation)
```

**Key Points**:
- Translation to English
- No diarization by default
- Single translation result
- Language detection automatic

## Flow 6: Translation (File with Diarization)

**Command**: `python speech_demo.py --file audio.wav --diarize --operation translation`

```
User
  ↓
File Input (audio.wav)
  ↓
SpeechRecognizer.translate_file(enable_diarization=True)
  ↓
DiarizationService.diarize_file()
  ↓
Diarizer.diarize() → Pyannote.audio → Speaker Detection
  ↓
Audio Chunking → Speaker-specific segments
  ↓
For each segment:
  GroqAPIClient.translate() → ResponseParser
  ↓
DiarizationResult (with speaker-attributed translations)
  ↓
User Response (speaker-separated translations)
```

**Key Points**:
- Translation with speaker diarization
- Each speaker's speech translated separately
- Maintains speaker attribution
- Most comprehensive output

## Flow 7: Microphone Translation

**Command**: `python speech_demo.py --microphone-mode single --operation translation`

```
User
  ↓
Microphone Input (real-time recording)
  ↓
SpeechRecognizer.translate_microphone()
  ↓
AudioProcessor.record_audio()
  ↓
GroqAPIClient.translate()
  ↓
ResponseParser.parse_translation_response()
  ↓
SpeechRecognitionResult (translated text)
  ↓
User Response (English translation)
```

**Key Points**:
- Real-time translation
- Microphone input
- Single translation result
- Language detection automatic

## Error Handling Flows

### Diarization Fallback
```
DiarizationService.diarize_file()
  ↓
Diarizer.diarize() → FAILS
  ↓
_create_fallback_result()
  ↓
Basic transcription without diarization
  ↓
User Response (fallback transcription)
```

### API Error Handling
```
GroqAPIClient.transcribe() → FAILS
  ↓
Retry mechanism (if transient error)
  ↓
Error propagation to user
  ↓
User Response (error message)
```

## Performance Characteristics

### File Processing
- **No Diarization**: O(n) where n is audio length
- **With Diarization**: O(n log n) due to Pyannote.audio processing
- **Memory Usage**: O(n) for file loading

### Microphone Processing
- **Single Mode**: O(1) per recording
- **Continuous Mode**: O(1) per chunk
- **Memory Usage**: O(1) for streaming

### API Calls
- **File Processing**: O(1) per file (or per segment for diarization)
- **Microphone**: O(k) where k is number of chunks
- **Network**: Depends on audio size and network conditions

## Configuration Impact

### Environment Variables
- `GROQ_API_KEY`: Required for all flows
- `HF_TOKEN`: Required for diarization flows
- `GROQ_MODEL_ID`: Affects processing speed and accuracy
- `GROQ_TEMPERATURE`: Affects output variability

### Audio Settings
- **Sample Rate**: 16kHz (standard)
- **Chunk Size**: Configurable for memory/performance trade-off
- **Format**: WAV (internal), supports various input formats

## Best Practices

### For File Processing
1. Use diarization for multi-speaker content
2. Skip diarization for single-speaker content (faster)
3. Consider file size limits (24MB for Groq API)

### For Microphone Processing
1. Use single mode for short recordings
2. Use continuous mode for long sessions
3. Consider audio quality and environment noise

### For Translation
1. Ensure clear audio for better translation accuracy
2. Consider speaker diarization for multi-speaker translation
3. Be aware of language detection limitations