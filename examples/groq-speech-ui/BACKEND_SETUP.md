# Backend Setup Guide

This guide explains how to configure the Groq API key in the backend instead of the frontend UI.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   Backend API   │    │   Groq API      │
│   (Next.js)     │───▶│   (FastAPI)     │───▶│   (External)    │
│                 │    │                 │    │                 │
│ • No API key    │    │ • API key       │    │ • Requires key  │
│ • User friendly │    │ • Secure        │    │ • Speech API    │
│ • Mock mode     │    │ • Proxy         │    │ • Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Why This Approach is Better

### ✅ **Security Benefits**
- API key never exposed to frontend
- No client-side storage of sensitive data
- Centralized key management
- Better for production deployments

### ✅ **User Experience**
- Users don't need to manage API keys
- Simpler onboarding process
- No risk of exposing keys in browser
- Works with mock mode for demos

### ✅ **Development Benefits**
- Easier environment management
- Consistent configuration across environments
- Better for team development
- Simplified deployment

## 🚀 Setup Instructions

### 1. Configure Backend API Key

Create a `.env` file in the root directory of the `groq-speech` project:

```bash
cd groq-speech
```

Create `.env` file:
```env
# Groq API Configuration
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional: Custom API endpoint (default: https://api.groq.com/openai/v1)
GROQ_API_BASE_URL=https://api.groq.com/openai/v1

# Optional: Model configuration
GROQ_MODEL_ID=whisper-large-v3-turbo
GROQ_RESPONSE_FORMAT=verbose_json
GROQ_TEMPERATURE=0.0

# Optional: Audio processing settings
DEFAULT_SAMPLE_RATE=16000
DEFAULT_CHANNELS=1
AUDIO_CHUNK_DURATION=1.0
AUDIO_BUFFER_SIZE=16384
```

### 2. Start the Backend Server

```bash
# From the groq-speech directory
python -m api.server
```

The server will start on `http://localhost:8000`

### 3. Verify Backend Configuration

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0",
  "api_key_configured": true
}
```

### 4. Start the Frontend

```bash
# From the groq-speech-ui directory
npm run dev
```

The frontend will start on `http://localhost:3000`

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `GROQ_API_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `GROQ_MODEL_ID` | `whisper-large-v3-turbo` | Speech model to use |
| `GROQ_RESPONSE_FORMAT` | `verbose_json` | Response format |
| `GROQ_TEMPERATURE` | `0.0` | Model temperature |
| `DEFAULT_LANGUAGE` | `en-US` | Default recognition language |
| `DEFAULT_SAMPLE_RATE` | `16000` | Audio sample rate |
| `DEFAULT_CHANNELS` | `1` | Audio channels (mono) |

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and API key status |
| `/api/v1/recognize` | POST | Single-shot transcription |
| `/api/v1/translate` | POST | Single-shot translation |
| `/ws/recognize` | WebSocket | Continuous recognition |
| `/api/v1/models` | GET | Available models |
| `/api/v1/languages` | GET | Supported languages |

## 🧪 Testing the Setup

### 1. Test Backend Health
```bash
curl http://localhost:8000/health
```

### 2. Test Frontend Connection
Open `http://localhost:3000` and check the backend status indicator.

### 3. Test Speech Recognition
- Enable "Mock Mode" for testing without API calls
- Or use real mode with configured backend

## 🔒 Security Best Practices

### 1. Environment Management
```bash
# Development
cp .env.example .env
# Edit .env with your actual API key

# Production
# Set environment variables in your deployment platform
```

### 2. API Key Security
- Never commit API keys to version control
- Use environment variables in production
- Rotate keys regularly
- Monitor API usage

### 3. CORS Configuration
The backend is configured to allow all origins for development. For production:

```python
# In api/server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚨 Troubleshooting

### Backend Issues

**Problem**: `GROQ_API_KEY not set`
**Solution**: Create `.env` file with your API key

**Problem**: `Failed to initialize Groq client`
**Solution**: Check API key format and network connectivity

**Problem**: `ModuleNotFoundError: No module named 'groq'`
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Frontend Issues

**Problem**: "Cannot connect to backend server"
**Solution**: 
1. Ensure backend is running on port 8000
2. Check firewall settings
3. Verify CORS configuration

**Problem**: "Backend API key not configured"
**Solution**: 
1. Check backend health endpoint
2. Verify `.env` file configuration
3. Restart backend server

### API Issues

**Problem**: "API request failed: 401"
**Solution**: Check API key validity and permissions

**Problem**: "API request failed: 429"
**Solution**: Rate limit exceeded, wait and retry

## 📊 Monitoring

### Backend Logs
```bash
# Start with verbose logging
python -m api.server --log-level debug
```

### Health Monitoring
```bash
# Check API key status
curl http://localhost:8000/health | jq '.api_key_configured'

# Monitor backend uptime
watch -n 5 'curl -s http://localhost:8000/health | jq ".status"'
```

## 🎯 Production Deployment

### 1. Environment Variables
Set in your deployment platform:
```bash
GROQ_API_KEY=your_production_api_key
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
```

### 2. CORS Configuration
Update CORS settings for your domain:
```python
allow_origins=["https://yourdomain.com"]
```

### 3. Health Checks
Monitor the health endpoint:
```bash
curl https://your-backend-domain.com/health
```

## 📝 Summary

This setup provides:
- ✅ Secure API key management
- ✅ Better user experience
- ✅ Simplified deployment
- ✅ Mock mode for demos
- ✅ Comprehensive monitoring
- ✅ Production-ready configuration

The API key is now properly configured in the backend, eliminating the need for users to manage API keys in the frontend UI. 