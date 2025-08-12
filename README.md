# Groq Speech SDK

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/your-repo/groq-speech)

High-performance speech recognition using Groq's AI services with comprehensive timing metrics and accuracy improvements.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** and pip
- **Node.js 18+** and npm (for web UI)
- **Groq API Key** from [Groq Console](https://console.groq.com/)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd groq-speech

# Set up environment
echo "GROQ_API_KEY=your_actual_groq_api_key_here" > .env
```

## 🎯 **Three Ways to Run**

### **Option 1: One-Command Local Development (Recommended for Development)**

```bash
# Single command to start everything
./run-dev.sh
```

This script will:
- ✅ Check and configure your environment
- ✅ Install all Python and Node.js dependencies
- ✅ Start the FastAPI backend server
- ✅ Start the Next.js frontend
- ✅ Open your browser to http://localhost:3000

**What you get:**
- 🌐 Frontend: http://localhost:3000
- 🔧 Backend: http://localhost:8000
- 📖 API Docs: http://localhost:8000/docs

### **Option 2: Docker Development (Recommended for Consistent Environment)**

```bash
# Start both backend and frontend in Docker
cd deployment/docker
docker-compose -f docker-compose.full.yml up --build
```

**What you get:**
- 🌐 Frontend: http://localhost:3000
- 🔧 Backend: http://localhost:8000
- 🐳 Everything runs in containers
- 🔄 Hot reload for both frontend and backend

### **Option 3: Production Docker Deployment**

```bash
# Production deployment with monitoring
cd deployment/docker
docker-compose up --build
```

**What you get:**
- 🚀 Production-ready setup
- 📊 Monitoring with Prometheus/Grafana
- 🔒 Redis caching
- 🛡️ Health checks and auto-restart

## 🔧 **Configuration**

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional - API Configuration
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL_ID=whisper-large-v3-turbo

# Optional - Performance Tuning
AUDIO_CHUNK_DURATION=1.0
AUDIO_BUFFER_SIZE=16384
AUDIO_SILENCE_THRESHOLD=0.005
DEFAULT_PHRASE_TIMEOUT=5
DEFAULT_SILENCE_TIMEOUT=2
```

## 📁 **Project Structure**

```
groq-speech/
├── groq_speech/           # Core SDK
├── api/                   # FastAPI server
├── examples/
│   └── groq-speech-ui/   # Next.js frontend
├── deployment/docker/     # Docker configurations
├── requirements.txt       # Development dependencies
├── requirements-dev.txt   # Development tools
├── run-dev.sh            # One-command development script
└── setup.py              # Package setup
```

## 🎤 **Usage Examples**

### Basic Recognition
```python
from groq_speech import SpeechConfig, SpeechRecognizer

recognizer = SpeechRecognizer(SpeechConfig())
result = recognizer.recognize_once_async()

if result.timing_metrics:
    timing = result.timing_metrics.get_metrics()
    print(f"Total time: {timing['total_time']*1000:.1f}ms")
```

### Web UI Demo
1. Run one of the three options above
2. Open http://localhost:3000
3. Click "Start Recording" to begin speech recognition
4. Use "Mock Mode" for testing without API key

### CLI Example
```bash
# Test CLI functionality
python examples/cli_speech_recognition.py --help

# List available models
python examples/cli_speech_recognition.py --list-models

# List available languages
python examples/cli_speech_recognition.py --list-languages
```

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=groq_speech --cov=api

# Run timing metrics test
python test_timing_metrics.py
```

## 🚀 **Development Workflow**

### **For Backend Changes:**
```bash
# Option 1: Local development
./run-dev.sh

# Option 2: Docker development
cd deployment/docker
docker-compose -f docker-compose.full.yml up --build
```

### **For Frontend Changes:**
```bash
# Option 1: Local development
./run-dev.sh

# Option 2: Docker development (with volume mounts for hot reload)
cd deployment/docker
docker-compose -f docker-compose.full.yml up --build
```

### **For Production Deployment:**
```bash
cd deployment/docker
docker-compose up --build
```

## 📊 **Performance Highlights**

- **API Call Time**: ~295ms average
- **Total Response Time**: Under 1 second
- **Accuracy**: 95% confidence
- **Memory Usage**: Optimized buffer management
- **Network Efficiency**: Audio compression and connection pooling

## 🔒 **Security Features**

- API keys stored securely in backend
- CORS configured for development and production
- Environment-based configuration
- No sensitive data exposed to frontend

## 🆘 **Troubleshooting**

### **Common Issues:**

**Backend won't start:**
- Check `.env` file has valid `GROQ_API_KEY`
- Ensure port 8000 is free
- Run `pip install -r requirements.txt`

**Frontend can't connect:**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify CORS configuration

**Docker issues:**
- Ensure Docker and Docker Compose are installed
- Check container logs: `docker-compose logs`
- Rebuild containers: `docker-compose up --build`

## 📚 **Documentation**

- **[📖 Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md)** - Complete technical documentation
- **[🏗️ Architecture Design](docs/architecture-design.md)** - System architecture
- **[🚀 Quick Start Guide](QUICKSTART.md)** - 5-minute setup guide

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Groq's AI services for high-performance speech recognition.** 