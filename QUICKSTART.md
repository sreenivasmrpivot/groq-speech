# 🚀 Quick Start Guide

## **Get groq-speech-ui running in under 5 minutes!**

### **Prerequisites**
- Python 3.8+ and pip
- Node.js 18+ and npm
- Groq API key from [console.groq.com](https://console.groq.com)

### **Step 1: Setup**
```bash
git clone <repository-url>
cd groq-speech
echo "GROQ_API_KEY=your_actual_key_here" > .env
```

### **Step 2: Choose Your Method**

#### **🎯 Method 1: One Command (Easiest)**
```bash
./run-dev.sh
```
**Opens browser automatically to http://localhost:3000**

#### **🐳 Method 2: Docker (Most Reliable)**
```bash
cd deployment/docker
docker-compose -f docker-compose.full.yml up --build
```
**Opens browser to http://localhost:3000**

#### **🔧 Method 3: Manual (For Developers)**
```bash
# Terminal 1: Backend
pip install -r groq_speech/requirements.txt
pip install -e .
pip install -r api/requirements.txt
python -m api.server

# Terminal 2: Frontend
cd examples/groq-speech-ui
npm install
npm run dev
```

### **Step 3: Use the App**
1. Open http://localhost:3000
2. Click "Start Recording"
3. Speak into your microphone
4. See real-time transcription!

### **🎉 That's it! You're ready to use groq-speech-ui!**

---

## **📋 What Each Method Does**

### **Method 1: `./run-dev.sh`**
- ✅ Installs dependencies in correct order (SDK → API → Examples)
- ✅ Starts backend server on port 8000
- ✅ Starts frontend on port 3000
- ✅ Handles all setup automatically

### **Method 2: Docker**
- ✅ Builds containers with proper dependency chain
- ✅ Runs both services with hot reload
- ✅ Includes Redis caching
- ✅ Production-ready configuration

### **Method 3: Manual**
- ✅ Full control over each component
- ✅ Good for development and debugging
- ✅ Understands the dependency structure

---

## **🔗 Dependency Chain**

```
groq_speech/ (Core SDK)
    ↓
examples/cli_speech_recognition.py (consumes groq_speech)
    ↓
api/server.py (consumes groq_speech)
    ↓
examples/groq-speech-ui (consumes api via HTTP)
```

---

**Need help?** Check the [main README](README.md) for detailed instructions or the [Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md) for technical details.
