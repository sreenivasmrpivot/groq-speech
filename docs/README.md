# Groq Speech SDK - Documentation

## 📚 **Documentation Index**

This directory contains comprehensive documentation for the Groq Speech SDK.

### **🏗️ Architecture & Design**
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture analysis
- **[CODE_ANALYSIS.md](CODE_ANALYSIS.md)** - Detailed code analysis and technical decisions

### **🚀 Deployment & Operations**
- **[Deployment Guide](../deployment/README.md)** - Docker, Cloud Run, and production deployment
- **[API Status Report](API_STATUS_REPORT.md)** - Current API endpoint status and functionality

### **🔧 Development & Testing**
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines and contribution process
- **[Postman Testing Guide](POSTMAN_TESTING_GUIDE.md)** - API testing with Postman

### **📖 User Guides**
- **[Main README](../README.md)** - Quick start and overview
- **[SDK API Reference](../groq_speech/API_REFERENCE.md)** - Complete SDK API documentation

## 🎯 **Quick Navigation**

### **For Users**
1. Start with [Main README](../README.md) for quick start
2. Check [Deployment Guide](../deployment/README.md) for setup
3. Review [SDK API Reference](../groq_speech/API_REFERENCE.md) for usage

### **For Developers**
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system understanding
2. Study [CODE_ANALYSIS.md](CODE_ANALYSIS.md) for implementation details
3. Follow [Contributing Guide](CONTRIBUTING.md) for development

### **For DevOps**
1. Use [Deployment Guide](../deployment/README.md) for production setup
2. Check [API Status Report](API_STATUS_REPORT.md) for monitoring
3. Review [Postman Testing Guide](POSTMAN_TESTING_GUIDE.md) for testing

## 🏗️ **System Overview**

The Groq Speech SDK is a comprehensive speech recognition and translation system with three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: User Interfaces                │
├─────────────────────────────────────────────────────────────┤
│  CLI Client (speech_demo.py)  │  Web UI (groq-speech-ui)   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 2: API Layer                      │
├─────────────────────────────────────────────────────────────┤
│                    FastAPI Server (api/)                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: Core SDK                       │
├─────────────────────────────────────────────────────────────┤
│              groq_speech/ (Python SDK)                     │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 **Key Features**

### **Speech Processing**
- ✅ **File Transcription** - Process audio files with high accuracy
- ✅ **File Translation** - Translate audio to different languages
- ✅ **Speaker Diarization** - Identify and separate multiple speakers
- ✅ **Microphone Processing** - Real-time audio processing

### **Advanced Features**
- ✅ **Voice Activity Detection** - Client-side real-time VAD
- ✅ **Continuous Processing** - Long-form audio with chunking
- ✅ **GPU Acceleration** - CUDA support for diarization
- ✅ **Performance Monitoring** - Built-in metrics and analytics

### **Deployment Options**
- ✅ **Local Development** - Docker Compose with hot reload
- ✅ **Production** - Docker containers with GPU support
- ✅ **Cloud Run** - GCP Cloud Run with GPU acceleration

## 📊 **Current Status**

### **Working Features**
- **CLI Interface**: All 10 command types working perfectly
- **Web Interface**: Complete feature parity with CLI
- **API Server**: REST API with all endpoints functional
- **VAD Processing**: Client-side real-time silence detection
- **Diarization**: GPU-accelerated speaker diarization
- **Translation**: Multi-language translation support

### **Performance**
- **CLI**: Direct SDK access, no network overhead
- **Web UI**: Client-side VAD for real-time processing
- **API**: REST API with efficient audio processing
- **GPU**: Automatic CUDA detection and usage

## 🔧 **Technical Highlights**

### **Architecture Decisions**
1. **Client-Side VAD** - Real-time processing without network latency
2. **Unified Components** - Single classes handle multiple modes
3. **REST API Only** - Simplified architecture, easier maintenance
4. **SDK Factory Methods** - Centralized configuration creation

### **Performance Optimizations**
1. **Chunked Processing** - Handles large files efficiently
2. **Memory Management** - Optimized for both short and long audio
3. **GPU Support** - Automatic detection and usage
4. **Real-Time VAD** - 15-second silence detection

## 📈 **Getting Started**

### **1. Quick Start**
```bash
# Clone repository
git clone <repository-url>
cd groq-speech

# Install dependencies
pip install -r requirements.txt
cd examples/groq-speech-ui && npm install

# Configure environment
cp groq_speech/env.template groq_speech/.env
# Edit with your API keys

# Run CLI
python examples/speech_demo.py --file audio.wav

# Run Web UI
cd api && python server.py &
cd examples/groq-speech-ui && npm run dev
```

### **2. Docker Deployment**
```bash
# Standard deployment
docker-compose -f deployment/docker/docker-compose.yml up

# GPU-enabled deployment
docker-compose -f deployment/docker/docker-compose.gpu.yml up
```

### **3. Cloud Run Deployment**
```bash
# Deploy to GCP Cloud Run
cd deployment/gcp
./deploy.sh
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Code Standards**
- Follow existing code patterns
- Add comprehensive documentation
- Include tests for new features
- Update documentation as needed

## 📞 **Support**

For issues and questions:
1. Check the [documentation](.) for answers
2. Review [existing issues](https://github.com/your-repo/issues)
3. Create a new issue with detailed information

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Groq, Pyannote.audio, and modern web technologies.**