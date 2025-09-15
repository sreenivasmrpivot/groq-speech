# Groq Speech SDK - Deployment Guide

## 🚀 **Deployment Options**

The Groq Speech SDK supports multiple deployment options:

1. **Local Development** - Docker Compose with hot reload
2. **Production** - Docker containers with GPU support
3. **Cloud Run** - GCP Cloud Run with GPU acceleration

## 🐳 **Docker Deployment**

### **Prerequisites**
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- Groq API key and Hugging Face token

### **Environment Setup**
```bash
# Copy environment template
cp groq_speech/env.template groq_speech/.env

# Edit with your API keys
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### **1. Standard Deployment**
```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yml up

# Or run in background
docker-compose -f deployment/docker/docker-compose.yml up -d
```

**Services:**
- **API Server**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **Health Check**: http://localhost:8000/health

### **2. GPU-Enabled Deployment**
```bash
# Start with GPU support
docker-compose -f deployment/docker/docker-compose.gpu.yml up

# Check GPU availability
docker-compose -f deployment/docker/docker-compose.gpu.yml exec api python test_gpu_support.py
```

**GPU Features:**
- **Pyannote.audio**: CUDA acceleration for diarization
- **Automatic Detection**: Falls back to CPU if GPU unavailable
- **Memory Optimization**: Efficient GPU memory usage

### **3. Development Deployment**
```bash
# Start with hot reload
docker-compose -f deployment/docker/docker-compose.dev.yml up

# Frontend hot reload enabled
# API server restarts on code changes
```

**Development Features:**
- **Hot Reload**: Code changes trigger automatic restarts
- **Volume Mounts**: Local code changes reflected immediately
- **Debug Mode**: Enhanced logging and error reporting

## ☁️ **GCP Cloud Run Deployment**

### **Prerequisites**
- Google Cloud SDK installed
- GCP project with Cloud Run API enabled
- Docker registry access (Artifact Registry)

### **1. Build and Push Images**
```bash
# Build API server image
docker build -f deployment/docker/Dockerfile.gpu -t gcr.io/PROJECT_ID/groq-speech-api .

# Build frontend image
cd examples/groq-speech-ui
docker build -t gcr.io/PROJECT_ID/groq-speech-ui .

# Push to registry
docker push gcr.io/PROJECT_ID/groq-speech-api
docker push gcr.io/PROJECT_ID/groq-speech-ui
```

### **2. Deploy to Cloud Run**
```bash
# Deploy API server
gcloud run deploy groq-speech-api \
  --image gcr.io/PROJECT_ID/groq-speech-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4

# Deploy frontend
gcloud run deploy groq-speech-ui \
  --image gcr.io/PROJECT_ID/groq-speech-ui \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

### **3. Configure Environment Variables**
```bash
# Set API server environment
gcloud run services update groq-speech-api \
  --set-env-vars GROQ_API_KEY=your_groq_api_key,HF_TOKEN=your_huggingface_token

# Set frontend environment
gcloud run services update groq-speech-ui \
  --set-env-vars NEXT_PUBLIC_API_URL=https://groq-speech-api-xxx.run.app
```

## 🔧 **Configuration Options**

### **Docker Compose Files**

#### **`docker-compose.yml`** - Standard Production
```yaml
services:
  api:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
  
  frontend:
    build:
      context: ../../examples/groq-speech-ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
    depends_on:
      - api
```

#### **`docker-compose.gpu.yml`** - GPU-Enabled
```yaml
services:
  api:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile.gpu
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### **`docker-compose.dev.yml`** - Development
```yaml
services:
  api:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ../../api:/app/api
      - ../../groq_speech:/app/groq_speech
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
  
  frontend:
    build:
      context: ../../examples/groq-speech-ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ../../examples/groq-speech-ui/src:/app/src
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
```

### **Dockerfiles**

#### **`Dockerfile`** - Standard API Server
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY groq_speech/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API server
COPY api/ ./api/
COPY groq_speech/ ./groq_speech/

# Set working directory
WORKDIR /app/api

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "server.py"]
```

#### **`Dockerfile.gpu`** - GPU-Enabled API Server
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY groq_speech/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy API server and SDK
COPY api/ ./api/
COPY groq_speech/ ./groq_speech/
COPY test_gpu_support.py ./

# Set working directory
WORKDIR /app/api

# Expose port
EXPOSE 8000

# Start server
CMD ["python3", "server.py"]
```

## 🔍 **Health Checks**

### **API Server Health Check**
```bash
# Check API server status
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "groq_api": "connected",
    "hf_api": "connected"
  }
}
```

### **Frontend Health Check**
```bash
# Check frontend status
curl http://localhost:3000

# Should return HTML page
```

### **GPU Support Check**
```bash
# Check GPU availability
docker-compose exec api python test_gpu_support.py

# Expected output
✅ CUDA available: True
✅ PyTorch CUDA: True
✅ GPU count: 1
✅ GPU name: NVIDIA GeForce RTX 4090
```

## 📊 **Monitoring and Logging**

### **API Server Logs**
```bash
# View API server logs
docker-compose logs -f api

# View specific service logs
docker-compose logs -f api | grep "ERROR"
```

### **Frontend Logs**
```bash
# View frontend logs
docker-compose logs -f frontend

# View build logs
docker-compose logs frontend | grep "build"
```

### **Performance Monitoring**
- **API Response Times**: Available in `/api/v1/status`
- **GPU Usage**: Monitor with `nvidia-smi`
- **Memory Usage**: Monitor with `docker stats`

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. GPU Not Available**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# If not available, install NVIDIA Docker runtime
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### **2. API Key Issues**
```bash
# Check environment variables
docker-compose exec api env | grep GROQ_API_KEY

# Verify API key format
docker-compose exec api python -c "import os; print('API Key set:', bool(os.getenv('GROQ_API_KEY')))"
```

#### **3. Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Change ports in docker-compose.yml if needed
```

#### **4. Memory Issues**
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### **Debug Mode**
```bash
# Enable debug logging
docker-compose -f deployment/docker/docker-compose.dev.yml up

# Check detailed logs
docker-compose logs -f api | grep "DEBUG"
```

## 🔄 **Updates and Maintenance**

### **Update Images**
```bash
# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### **Update Environment Variables**
```bash
# Update .env file
vim groq_speech/.env

# Restart services
docker-compose restart
```

### **Backup Configuration**
```bash
# Backup environment
cp groq_speech/.env groq_speech/.env.backup

# Backup Docker Compose files
cp deployment/docker/docker-compose*.yml ./backup/
```

## 📈 **Scaling**

### **Horizontal Scaling**
```yaml
# Scale API server
services:
  api:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### **Load Balancing**
```yaml
# Add load balancer
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

This deployment guide provides comprehensive instructions for deploying the Groq Speech SDK in various environments with proper configuration and monitoring.