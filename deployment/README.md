# Groq Speech SDK - Deployment Guide

## 🚀 **Deployment Options**

The Groq Speech SDK supports multiple deployment options:

1. **Local Development** - Docker Compose with hot reload
2. **Production** - Docker containers with GPU support
3. **GCP Cloud Run** - Serverless deployment with auto-scaling
4. **GKE GPU** - Kubernetes deployment with GPU acceleration

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

### **4. Production Deployment**
```bash
# Production deployment with GPU support
docker-compose -f deployment/docker/docker-compose.gpu.yml up

# Check GPU availability
docker-compose -f deployment/docker/docker-compose.gpu.yml exec api python test_gpu_support.py
```

**Production Features:**
- **GPU Support**: CUDA acceleration for diarization
- **Health Checks**: Comprehensive monitoring
- **Security**: Non-root containers and minimal images
- **Performance**: Optimized for production workloads

## ☁️ **GCP Cloud Run Deployment**

### **Prerequisites**
- Google Cloud SDK installed
- GCP project with Cloud Run API enabled
- Docker registry access (Artifact Registry)

### **Quick Deployment**
```bash
# Deploy to Cloud Run (CPU only)
cd deployment/gcp
./deploy.sh
```

**Features:**
- **Auto-scaling**: Automatically scales based on demand
- **Pay-per-use**: Only pay for actual usage
- **Global deployment**: Deploy to multiple regions
- **Integrated monitoring**: Built-in logging and monitoring

## 🚀 **GKE GPU Deployment**

### **Prerequisites**
- Google Cloud SDK installed
- GCP project with GKE API enabled
- Docker registry access (Artifact Registry)

### **Quick Deployment**
```bash
# Deploy to GKE with GPU support
cd deployment/gcp
./deploy-simple-gke.sh
```

**Features:**
- **GPU acceleration**: NVIDIA T4 GPUs for fast diarization
- **Kubernetes orchestration**: Full container orchestration
- **High availability**: Multi-zone deployment
- **Custom scaling**: Fine-grained control over resources

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

### **Cloud Monitoring**
```bash
# Cloud Run logs
gcloud run services logs read groq-speech-api --region=us-central1
gcloud run services logs read groq-speech-ui --region=us-central1

# GKE logs
kubectl logs -l app=groq-speech-api
kubectl logs -l app=groq-speech-ui
```

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

### **Docker Scaling**
```yaml
# Scale API server
services:
  api:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### **Cloud Run Scaling**
```bash
# Auto-scaling is built-in
# Min instances: 1
# Max instances: 10
# Scales based on demand
```

### **GKE Scaling**
```bash
# Scale API deployment
kubectl scale deployment groq-speech-api --replicas=3

# Scale UI deployment
kubectl scale deployment groq-speech-ui --replicas=2

# Auto-scaling (if enabled)
kubectl autoscale deployment groq-speech-api --min=1 --max=5 --cpu-percent=70
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