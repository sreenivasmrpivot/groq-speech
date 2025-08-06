# 🎤 Groq Speech Examples

This directory contains end-to-end examples demonstrating how to use the Groq Speech SDK.

## 📁 Examples

### 1. **Web UI Example** (`groq-speech-ui/`)
A complete Next.js web application with React frontend for real-time speech recognition and translation.

**Features:**
- ✅ Real-time transcription and translation
- ✅ Single-shot and continuous recognition modes
- ✅ Performance metrics and timing visualization
- ✅ Mock mode for testing without API key
- ✅ Responsive and modern UI
- ✅ WebSocket support for continuous streaming
- ✅ Debug tools and comprehensive testing

**Quick Start:**
```bash
cd groq-speech-ui
npm install
npm run dev
# Open http://localhost:3000
```

**Usage:**
1. Open http://localhost:3000
2. Enable "Use Mock API" for immediate testing
3. Or configure your GROQ_API_KEY and click "Refresh"
4. Click "Start Recording" to begin speech recognition

### 2. **CLI Example** (`cli_speech_recognition.py`)
A command-line interface that directly uses the `groq_speech` library for speech recognition and translation.

**Features:**
- ✅ Direct library usage (no web server required)
- ✅ Microphone input for real-time recognition
- ✅ Audio file processing
- ✅ Transcription and translation modes
- ✅ Multiple language support
- ✅ Model selection
- ✅ Confidence scoring

**Quick Start:**
```bash
# Install dependencies
pip install groq-speech

# Set your API key
export GROQ_API_KEY="your_groq_api_key_here"

# Run CLI example
python cli_speech_recognition.py --mode transcription --language en-US
```

**Usage Examples:**
```bash
# Transcribe from microphone
python cli_speech_recognition.py --mode transcription --language en-US

# Translate from microphone
python cli_speech_recognition.py --mode translation --language es-ES

# Transcribe from audio file
python cli_speech_recognition.py --file audio.wav --language en-US

# List available models and languages
python cli_speech_recognition.py --list-models
python cli_speech_recognition.py --list-languages
```

## 🔧 Setup Requirements

### For Web UI Example:
- Node.js 18+ and npm
- GROQ_API_KEY configured in backend `.env` file
- Backend server running (`python -m api.server`)

### For CLI Example:
- Python 3.8+
- GROQ_API_KEY environment variable
- Microphone access (for live recognition)

## 📊 Comparison

| Feature | Web UI | CLI |
|---------|--------|-----|
| **User Interface** | Modern web UI | Command line |
| **Real-time** | ✅ WebSocket streaming | ✅ Continuous recognition |
| **File Processing** | ✅ Upload support | ✅ Direct file input |
| **Translation** | ✅ Both modes | ✅ Both modes |
| **Performance Metrics** | ✅ Visual charts | ✅ Timing display |
| **Mock Mode** | ✅ For testing | ❌ Direct API only |
| **Deployment** | Web server required | Standalone script |
| **Dependencies** | Node.js, React, Next.js | Python only |

## 🚀 Getting Started

### Option 1: Web UI (Recommended for demos)
```bash
# Start backend
cd groq-speech
source .venv/bin/activate
python -m api.server

# Start frontend (in another terminal)
cd examples/groq-speech-ui
npm run dev
```

### Option 2: CLI (Recommended for development)
```bash
# Set API key
export GROQ_API_KEY="your_key_here"

# Run CLI
cd examples
python cli_speech_recognition.py --mode transcription --language en-US
```

## 🧪 Testing

### Web UI Testing:
- Open http://localhost:3000
- Use mock mode for immediate testing
- Test all features: transcription, translation, continuous recognition

### CLI Testing:
```bash
# Test microphone recognition
python cli_speech_recognition.py --mode transcription

# Test file processing (if you have an audio file)
python cli_speech_recognition.py --file test.wav --language en-US
```

## 📝 Notes

- **Web UI**: Best for demonstrations, user-friendly interface, comprehensive features
- **CLI**: Best for development, direct library usage, lightweight deployment
- Both examples demonstrate the full capabilities of the Groq Speech SDK
- Both support all languages and models supported by Groq
- Both include proper error handling and user feedback

## 🎯 Use Cases

**Web UI Example:**
- Product demonstrations
- User-facing applications
- Real-time speech interfaces
- Performance monitoring and visualization

**CLI Example:**
- Development and testing
- Server-side processing
- Automated speech recognition
- Integration into existing Python workflows 