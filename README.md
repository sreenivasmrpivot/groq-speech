# Groq Speech SDK

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/your-repo/groq-speech)

High-performance speech recognition using Groq's AI services with comprehensive timing metrics and accuracy improvements.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd groq-speech

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### Basic Usage

```python
from groq_speech import SpeechConfig, SpeechRecognizer

# Initialize
speech_config = SpeechConfig()
recognizer = SpeechRecognizer(speech_config=speech_config)

# Single recognition
result = recognizer.recognize_once_async()
print(f"Recognized: {result.text}")

# Continuous recognition
def on_recognized(result):
    print(f"'{result.text}' - {result.confidence:.2f}")

recognizer.connect("recognized", on_recognized)
recognizer.start_continuous_recognition()
```

### Examples

The project includes two complete end-to-end examples:

#### 1. Web UI Example
A modern Next.js web application with real-time speech recognition:

```bash
# Start backend
python -m api.server

# Start frontend (in another terminal)
cd examples/groq-speech-ui
npm run dev
# Open http://localhost:3000
```

#### 2. CLI Example
A command-line interface that directly uses the library:

```bash
# Set your API key
export GROQ_API_KEY="your_groq_api_key_here"

# Run CLI example
cd examples
python cli_speech_recognition.py --mode transcription --language en-US
```

See `examples/README.md` for detailed usage instructions.

## ✨ Key Features

- **🎤 Real-time Speech Recognition**: Single and continuous recognition modes
- **📊 Timing Metrics**: Detailed performance tracking for each pipeline stage
- **🎯 High Accuracy**: 95%+ confidence with optimized VAD and audio processing
- **📈 Performance Monitoring**: Real-time charts and visual indicators
- **🌐 Web Interface**: Interactive demo with comprehensive metrics
- **🧪 Comprehensive Testing**: Accuracy and performance test suites

## 📊 Performance Highlights

- **API Call Time**: ~295ms average (excellent performance)
- **Total Response Time**: Under 1 second
- **Accuracy**: 95% confidence
- **Memory Usage**: Optimized buffer management
- **Network Efficiency**: Audio compression and connection pooling

## 🔧 Configuration

Create a `.env` file:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional - Performance tuning
AUDIO_CHUNK_DURATION=1.0
AUDIO_BUFFER_SIZE=16384
AUDIO_SILENCE_THRESHOLD=0.005
DEFAULT_PHRASE_TIMEOUT=5
DEFAULT_SILENCE_TIMEOUT=2
```

## 🧪 Testing

```bash
# Run timing metrics test
python test_timing_metrics.py

# Run accuracy tests
python tests/test_transcription_accuracy.py

# Run all tests
python -m pytest tests/
```

## 📚 Documentation

- **[📖 Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md)** - Complete documentation covering all features
- **[🏗️ Architecture Design](docs/architecture-design.md)** - System architecture and design patterns
- **[🚀 Deployment Guide](docs/deployment-guide.md)** - Production deployment instructions
- **[⚡ Performance Optimization](docs/performance-optimization.md)** - Performance tuning and optimization

## 🎯 Recent Improvements

### Version 2.0.0 - Major Enhancements

- **⏱️ Timing Metrics**: Comprehensive performance tracking for microphone capture, API calls, and response processing
- **🎯 Accuracy Improvements**: Enhanced VAD, better audio processing, and optimized configuration
- **📊 Visual Analytics**: Real-time charts showing timing breakdown and performance trends
- **🔧 Performance Optimization**: Faster processing pipeline with better memory management
- **🧪 Enhanced Testing**: Comprehensive test suites for accuracy and performance

### Key Fixes

- ✅ Fixed wrong transcriptions (e.g., "I just" → "He's")
- ✅ Reduced missed transcriptions between segments
- ✅ Eliminated extra word repetitions
- ✅ Improved speech segmentation and boundary detection

## 🏗️ Architecture

```
Microphone → AudioConfig → AudioProcessor → VAD → SpeechRecognizer → Groq API → Response Processing → Result
```

### Core Components

- **SpeechRecognizer**: Main orchestration engine
- **AudioProcessor**: Optimized audio processing with VAD
- **TimingMetrics**: Comprehensive performance tracking
- **Web Demo**: Interactive interface with real-time charts

## 🚀 Quick Examples

### Basic Recognition
```python
from groq_speech import SpeechConfig, SpeechRecognizer

recognizer = SpeechRecognizer(SpeechConfig())
result = recognizer.recognize_once_async()

if result.timing_metrics:
    timing = result.timing_metrics.get_metrics()
    print(f"Total time: {timing['total_time']*1000:.1f}ms")
```

### Continuous Recognition with Timing
```python
def on_recognized(result):
    if result.timing_metrics:
        timing = result.timing_metrics.get_metrics()
        print(f"'{result.text}' - {timing['total_time']*1000:.1f}ms")

recognizer.connect("recognized", on_recognized)
recognizer.start_continuous_recognition()
```

### Web Demo with Charts
```bash
python examples/web_demo_timing.py
# Open http://localhost:5000 for interactive demo with timing metrics
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **📖 Documentation**: [Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md)
- **🐛 Issues**: Report bugs on GitHub
- **💡 Examples**: Check the `examples/` directory
- **🧪 Tests**: Run tests to verify functionality

---

**Built with ❤️ using Groq's AI services for high-performance speech recognition.** 