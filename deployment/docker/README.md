# Groq Speech Docker Deployment

This directory contains a clean, simplified Docker setup for deploying the Groq Speech API and UI as separate containers.

## 🏗️ Architecture

- **API Container**: FastAPI server with Groq Speech SDK
- **UI Container**: Next.js frontend application
- **Network**: Both containers communicate via Docker network
- **GPU Support**: Optional NVIDIA GPU acceleration for diarization

## 📁 Files

- `Dockerfile.api` - Optimized API container
- `Dockerfile.ui` - Optimized UI container  
- `docker-compose.yml` - Standard deployment setup
- `docker-compose.gpu.yml` - GPU-enabled deployment
- `docker-compose.dev.yml` - Development with hot reload
- `env.api.template` - API environment variables template
- `env.ui.template` - UI environment variables template
- `deploy-local.sh` - Local deployment script

## 🚀 Quick Start

### 1. Set up environment variables

```bash
# Copy the templates
cp deployment/docker/env.api.template .env.api
cp deployment/docker/env.ui.template .env.ui

# Edit with your API keys
nano .env.api
nano .env.ui
```

**Required in `.env.api`:**
- `GROQ_API_KEY` - Your Groq API key
- `HF_TOKEN` - Your Hugging Face token (optional, for diarization)

**Required in `.env.ui`:**
- `NEXT_PUBLIC_API_URL` - API URL (default: `http://groq-speech-api:8000`)

### 2. Deploy locally

```bash
# Standard deployment
docker-compose -f deployment/docker/docker-compose.yml up

# GPU-enabled deployment
docker-compose -f deployment/docker/docker-compose.gpu.yml up

# Development with hot reload
docker-compose -f deployment/docker/docker-compose.dev.yml up
```

This will:
- Build both containers
- Start the services
- Run health checks
- Display service URLs

### 3. Access the services

- **API**: http://localhost:8000
- **UI**: https://localhost:3443
- **API Docs**: http://localhost:8000/docs

**Note**: The UI uses HTTPS for microphone access. Your browser will show a security warning for the self-signed certificate. Click "Advanced" and "Proceed to localhost" to continue.

### 4. Check GPU support (if using GPU deployment)

```bash
# Check GPU availability
docker-compose -f deployment/docker/docker-compose.gpu.yml exec api python test_gpu_support.py

# Expected output
✅ CUDA available: True
✅ PyTorch CUDA: True
✅ GPU count: 1
✅ GPU name: NVIDIA GeForce RTX 4090
```

## 🛠️ Manual Commands

### Build and run with Docker Compose

```bash
# Standard deployment
docker-compose -f deployment/docker/docker-compose.yml up --build -d

# GPU-enabled deployment
docker-compose -f deployment/docker/docker-compose.gpu.yml up --build -d

# Development with hot reload
docker-compose -f deployment/docker/docker-compose.dev.yml up --build -d

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down
```

### Build individual containers

```bash
# Build API
docker build -f deployment/docker/Dockerfile.api -t groq-speech-api .

# Build UI
docker build -f deployment/docker/Dockerfile.ui -t groq-speech-ui .

# Run API
docker run -p 8000:8000 --env-file .env groq-speech-api

# Run UI (HTTPS)
docker run -p 3443:3443 -e NEXT_PUBLIC_API_URL=http://localhost:8000 groq-speech-ui

# Run with GPU support
docker run --gpus all -p 8000:8000 --env-file .env groq-speech-api
```

## 🔧 Configuration

### Environment Variables

#### API (.env.api)
| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Required** Groq API key |
| `HF_TOKEN` | - | Hugging Face token for diarization |
| `GROQ_MODEL_ID` | whisper-large-v3 | Groq model to use |
| `AUDIO_SAMPLE_RATE` | 16000 | Audio sample rate |
| `MAX_AUDIO_FILE_SIZE` | 25000000 | Max file size (25MB) |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |

#### UI (.env.ui)
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | http://groq-speech-api:8000 | **Required** API URL for UI |
| `NEXT_PUBLIC_FRONTEND_URL` | http://localhost:3000 | Frontend URL |
| `NEXT_PUBLIC_VERBOSE` | false | Enable verbose logging |
| `NEXT_PUBLIC_DEBUG` | false | Enable debug mode |

### Resource Limits

**API Container:**
- CPU: 2 cores
- Memory: 4GB
- Port: 8000

**UI Container:**
- CPU: 1 core  
- Memory: 1GB
- Port: 3443 (HTTPS)

## 🐛 Troubleshooting

### Check container health

```bash
# Check API health
curl http://localhost:8000/health

# Check UI health (HTTPS)
curl -k https://localhost:3443
```

### View container logs

```bash
# View all logs
docker-compose -f deployment/docker/docker-compose.yml logs

# View specific service logs
docker-compose -f deployment/docker/docker-compose.yml logs groq-speech-api
docker-compose -f deployment/docker/docker-compose.yml logs groq-speech-ui
```

### Common issues

1. **API not starting**: Check GROQ_API_KEY is set
2. **UI can't connect to API**: Check NEXT_PUBLIC_API_URL
3. **Build failures**: Ensure Docker has enough memory (4GB+)

## 🚀 Production Deployment

For production deployment to GCP Cloud Run or GKE, see the `../gcp/` directory.

### **Deployment Options:**
- **GCP Cloud Run**: Serverless deployment with auto-scaling
- **GKE GPU**: Kubernetes deployment with GPU acceleration
- **Docker Swarm**: Container orchestration for on-premises
- **Kubernetes**: Self-managed Kubernetes deployment

## 📊 Monitoring

The containers include health checks and logging:

- **API Health**: `/health` endpoint
- **API Docs**: `/docs` endpoint
- **Logs**: Available via `docker logs` or docker-compose logs
- **GPU Status**: Available via `test_gpu_support.py` script
- **Performance Metrics**: Built-in performance monitoring
- **Error Tracking**: Comprehensive error logging and reporting
