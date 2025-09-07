# Examples - Groq Speech SDK

This directory contains examples demonstrating the Groq Speech SDK's capabilities, including speech recognition, translation, and speaker diarization.

## 📁 Available Examples

### 1. `speech_demo.py` - Main Demo Script
**Purpose**: Comprehensive command-line demo for all SDK features

**Features:**
- ✅ **File Processing**: Audio file recognition and translation
- ✅ **Microphone Processing**: Real-time audio capture and processing
- ✅ **Speaker Diarization**: Multi-speaker detection and separation
- ✅ **Smart Grouping**: 24MB-optimized audio chunking
- ✅ **Voice Activity Detection**: Intelligent silence detection

**Usage:**
```bash
# File processing with diarization
python examples/speech_demo.py --file audio.wav --diarize

# File processing without diarization
python examples/speech_demo.py --file audio.wav

# File translation
python examples/speech_demo.py --file audio.wav --operation translation

# Microphone single-shot
python examples/speech_demo.py --microphone-mode single

# Microphone continuous
python examples/speech_demo.py --microphone-mode continuous

# Microphone with diarization
python examples/speech_demo.py --microphone-mode single --diarize

# Microphone translation
python examples/speech_demo.py --microphone-mode single --operation translation
```

### 2. `groq-speech-ui/` - Web Interface
**Purpose**: Modern web-based user interface for basic transcription and translation

**Features:**
- 🌐 **Web Interface**: Browser-based access
- 📁 **File Upload**: Drag-and-drop audio file processing
- 🎤 **Microphone Input**: Real-time audio processing
- 🔄 **Basic Functionality**: Simple transcription and translation

**Setup:**
```bash
cd examples/groq-speech-ui
npm install
npm run dev
```

## 🚀 Quick Start

### 1. Test File Processing
```bash
# Find a sample audio file
find . -name "*.wav" -o -name "*.mp3" | head -1

# Process with diarization
python examples/speech_demo.py --file <audio_file> --diarize
```

**Expected Output:**
```
🎭 Diarization Pipeline: Pyannote.audio FIRST, then Groq API per segment
✅ CORRECT diarization completed in 5.92s
🎭 Speakers detected: 2
📊 Total segments: 3
⏱️  Total duration: 30.0s
🎯 Overall confidence: 0.950

🎤 Speaker Groups with Accurate Transcription:
🎤 SPEAKER_00: Hello, how are you?
🎤 SPEAKER_01: I'm doing well, thank you.
```

### 2. Test Microphone Input
```bash
# Single-shot microphone recording
python examples/speech_demo.py --microphone-mode single

# Continuous microphone processing
python examples/speech_demo.py --microphone-mode continuous
```

### 3. Test Translation
```bash
# File translation
python examples/speech_demo.py --file audio.wav --operation translation

# Microphone translation
python examples/speech_demo.py --microphone-mode single --operation translation
```

## 🔧 Configuration Requirements

### Required Setup
```bash
# 1. Environment variables
cp groq_speech/env.template groq_speech/.env

# 2. Edit groq_speech/.env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here

# 3. Install dependencies
pip install -r examples/requirements.txt
```

### HF_TOKEN Setup (for diarization)
1. **Get Token**: Visit https://huggingface.co/settings/tokens
2. **Accept License**: Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. **Set Token**: Add to `groq_speech/.env`
4. **Restart**: Restart the application

## 📊 Performance Characteristics

### Accuracy Features
- **Speaker Attribution**: High accuracy with Pyannote.audio
- **Text Quality**: Accurate transcription per speaker
- **Timing Precision**: Precise speaker segment boundaries
- **Smart Grouping**: 24MB-optimized chunking for efficient API usage

### Processing Efficiency
- **Voice Activity Detection**: Intelligent silence detection
- **Smart Chunking**: Optimized audio segmentation
- **Fallback Mechanisms**: Multiple VAD implementations
- **Memory Efficiency**: Streaming support for large files

## 🎯 When to Use Each Mode

### Use File Mode When:
- ✅ You need **accurate speaker identification**
- ✅ You want **detailed speaker segments**
- ✅ You're doing **post-processing analysis**
- ✅ You have **pre-recorded audio**
- ✅ You need **professional quality results**

### Use Microphone Mode When:
- ✅ You need **real-time processing**
- ✅ You want **live transcription**
- ✅ You're doing **interactive applications**
- ✅ You need **continuous audio processing**

### Use Translation Mode When:
- ✅ You need **English translation** from other languages
- ✅ You're working with **multilingual content**
- ✅ You want **automatic language detection**

## 🔍 Troubleshooting

### Common Issues

#### 1. HF_TOKEN Not Configured
```
⚠️  HF_TOKEN not configured - Cannot perform proper diarization
💡 For microphone diarization, configure HF_TOKEN first
🔄 Falling back to basic transcription...
```

**Solution:**
- Set HF_TOKEN in `groq_speech/.env`
- Accept model license at HuggingFace
- Restart the application

#### 2. Audio File Not Found
```
❌ Audio file not found: audio.wav
```

**Solution:**
- Check file path and permissions
- Ensure file exists and is readable
- Use absolute paths if needed

#### 3. Microphone Not Detected
```
❌ PyAudio not available. Install with: pip install pyaudio
```

**Solution:**
- Install PyAudio: `pip install pyaudio`
- Check system audio permissions
- Verify audio device configuration

## 💡 Pro Tips

### 1. Optimal Audio Quality
- **Use WAV format** for best speaker detection
- **Ensure good recording quality** (clear audio, minimal background noise)
- **Record in quiet environments** for best results

### 2. Speaker Detection Optimization
- **Longer audio segments** provide better speaker detection
- **Multiple speakers** work best with clear speech patterns
- **Consistent audio levels** improve detection accuracy

### 3. Performance Optimization
- **File Mode**: Best for detailed analysis and accuracy
- **Microphone Mode**: Best for real-time applications
- **Smart Grouping**: Automatically optimizes for 24MB API limits

## 🚀 Advanced Usage

### 1. Batch Processing
```bash
# Process multiple files
for file in *.wav; do
    python examples/speech_demo.py --file "$file" --diarize
done
```

### 2. Custom Integration
```python
from groq_speech import SpeechRecognizer, SpeechConfig

# Create recognizer
config = SpeechConfig()
recognizer = SpeechRecognizer(config)

# Use diarization
result = recognizer.recognize_file("audio.wav", enable_diarization=True)

# Process results
if hasattr(result, "segments"):
    for segment in result.segments:
        print(f"Speaker {segment.speaker_id}: {segment.text}")
```

## ✅ Summary

The examples demonstrate the Groq Speech SDK's capabilities:

1. **Perfect Accuracy**: High-quality speaker attribution and transcription
2. **Unified Experience**: Consistent quality across all modes
3. **Better Performance**: Efficient, optimized processing
4. **Smart Features**: 24MB chunking, VAD, and fallback mechanisms
5. **Easy Integration**: Simple API for custom applications

**Key Benefits:**
- ✅ **Accurate speaker detection** with Pyannote.audio
- ✅ **Smart audio chunking** for optimal API usage
- ✅ **Voice activity detection** for intelligent processing
- ✅ **Multiple fallback mechanisms** for reliability
- ✅ **Professional quality** results