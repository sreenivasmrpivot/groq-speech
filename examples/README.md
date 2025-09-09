# Groq Speech SDK - Examples

This directory contains examples demonstrating how to use the Groq Speech SDK with its clean, simplified architecture.

## Quick Start

### Prerequisites
1. Install dependencies: `pip install -r groq_speech/requirements.txt`
2. Set up environment variables in `groq_speech/.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here  # Optional, for diarization
   ```

### Basic Usage

```bash
# File-based transcription (SDK handles all complexity internally)
python speech_demo.py --file audio.wav

# File-based translation with diarization
python speech_demo.py --file audio.wav --operation translation --diarize

# Microphone single mode
python speech_demo.py --microphone-mode single

# Microphone continuous mode
python speech_demo.py --microphone-mode continuous --diarize
```

## Architecture Benefits

The demo script showcases the SDK's clean architecture:

### Before (Complex Consumer Code)
```python
def process_audio_file(audio_file, mode, recognizer, enable_diarization=True):
    # 80+ lines of complex fallback logic, audio preprocessing, error handling
    if enable_diarization:
        try:
            result = recognizer.diarize_file(audio_file)
            if not result or not result.segments:
                # Complex fallback logic...
                pass
        except Exception as e:
            # More error handling...
            pass
    # ... 70+ more lines of complex code
```

### After (Simple Consumer Code)
```python
def process_audio_file(audio_file, mode, recognizer, enable_diarization=True):
    # 20 lines of simple API calls - SDK handles everything!
    try:
        if mode == "translation":
            result = recognizer.translate_file(audio_file, enable_diarization=enable_diarization)
        else:
            result = recognizer.recognize_file(audio_file, enable_diarization=enable_diarization)
        return result
    except Exception as e:
        print(f"❌ File processing failed: {e}")
        return None
```

## Key Improvements

1. **54% Code Reduction**: From 766 lines to 350 lines
2. **No Fallback Logic**: SDK handles all fallback scenarios internally
3. **No Manual Audio Preprocessing**: AudioProcessor handles it automatically
4. **Simple Error Handling**: SDK provides consistent error responses
5. **Clean API**: Just call `recognizer.recognize_file()` or `recognizer.translate_file()`

## Available Commands

### File Processing
- `--file audio.wav` - Process audio file
- `--operation transcription` - Speech-to-text (default)
- `--operation translation` - Speech-to-text in target language
- `--diarize` - Enable speaker diarization

### Microphone Processing
- `--microphone-mode single` - Record until Ctrl+C, then process
- `--microphone-mode continuous` - Real-time processing with VAD
- `--diarize` - Enable speaker diarization
- `--operation translation` - Translate to target language

### Debug Options
- `--verbose` - Enable verbose debug logging

## Examples

### Basic Transcription
```bash
python speech_demo.py --file audio.wav
```

### Translation with Diarization
```bash
python speech_demo.py --file audio.wav --operation translation --diarize
```

### Single Microphone Mode
```bash
python speech_demo.py --microphone-mode single
```

### Continuous Microphone Mode
```bash
python speech_demo.py --microphone-mode continuous --diarize
```

## SDK Integration

The demo shows how to integrate the SDK into your own applications:

```python
from groq_speech import SpeechRecognizer, SpeechConfig

# Create recognizer
config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# File processing
result = recognizer.recognize_file("audio.wav", enable_diarization=True)
result = recognizer.translate_file("audio.wav", enable_diarization=False)

# Audio data processing
result = recognizer.recognize_audio_data(audio_data, sample_rate)
result = recognizer.translate_audio_data(audio_data, sample_rate)

# VAD integration (for advanced use cases)
audio_level = recognizer.get_audio_level(audio_data)
should_create, reason = recognizer.should_create_chunk(audio_data, sample_rate, max_duration)
```

## Error Handling

The SDK provides consistent error handling:

```python
try:
    result = recognizer.recognize_file("audio.wav")
    if result and result.text:
        print(f"Transcription: {result.text}")
    else:
        print("No speech detected")
except Exception as e:
    print(f"Recognition failed: {e}")
```

## Performance

The SDK is optimized for performance:
- **O(1) audio preprocessing** with caching
- **O(n) diarization** with parallel processing
- **Memory-efficient** audio chunking
- **Connection pooling** for API calls

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r groq_speech/requirements.txt
   ```

2. **Missing API Keys**
   - Set `GROQ_API_KEY` in `groq_speech/.env`
   - Set `HF_TOKEN` for diarization features

3. **Audio Format Issues**
   - SDK automatically handles format conversion
   - Supports WAV, MP3, and other common formats

4. **Microphone Issues**
   - Install PyAudio: `pip install pyaudio`
   - Check microphone permissions

### Debug Mode

Use `--verbose` flag for detailed logging:

```bash
python speech_demo.py --file audio.wav --verbose
```

## Support

For issues and questions:
1. Check the main SDK documentation in `groq_speech/README.md`
2. Review the API reference in `groq_speech/API_REFERENCE.md`
3. Check error messages and logs for specific issues