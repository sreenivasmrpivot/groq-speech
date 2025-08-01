# Groq Speech SDK - Deployment Guide

## Overview

This guide covers deploying the Groq Speech SDK in various environments, from local development to production-scale deployments.

## Deployment Options

### 1. Local Development

#### Prerequisites

- Python 3.11+
- Groq API key
- Audio device (for microphone input)

#### Setup

```bash
# Clone repository
git clone <repository-url>
cd groq-speech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

#### Running Applications

```bash
# Run demos
python examples/voice_assistant_demo.py
python examples/transcription_workbench.py
python examples/web_demo.py

# Run API server
python -m api.server
```

### 2. Docker Deployment

#### Single Container

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t groq-speech .

# Run container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_api_key \
  groq-speech
```

#### Multi-Service Deployment

```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yml up -d

# Check status
docker-compose -f deployment/docker/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f
```

#### Services Overview

- **groq-speech-api**: FastAPI server (port 8000)
- **groq-speech-web-demo**: Flask web demo (port 5000)
- **redis**: Session management (port 6379)
- **nginx**: Load balancer (ports 80, 443)
- **prometheus**: Monitoring (port 9090)
- **grafana**: Visualization (port 3000)

### 3. Kubernetes Deployment

#### Prerequisites

- Kubernetes cluster
- kubectl configured
- Helm (optional)

#### Deploy with kubectl

```bash
# Create namespace
kubectl create namespace groq-speech

# Apply configurations
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/secret.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

#### Deploy with Helm

```bash
# Add Helm repository
helm repo add groq-speech https://charts.groq-speech.com

# Install release
helm install groq-speech groq-speech/groq-speech \
  --namespace groq-speech \
  --set groq.apiKey=your_api_key
```

### 4. Cloud Deployment

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker build -t groq-speech .
docker tag groq-speech:latest your-account.dkr.ecr.us-east-1.amazonaws.com/groq-speech:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/groq-speech:latest

# Deploy with ECS
aws ecs create-service \
  --cluster your-cluster \
  --service-name groq-speech \
  --task-definition groq-speech:1 \
  --desired-count 2
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/groq-speech
gcloud run deploy groq-speech \
  --image gcr.io/your-project/groq-speech \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Deploy to ACI
az container create \
  --resource-group your-rg \
  --name groq-speech \
  --image your-registry.azurecr.io/groq-speech:latest \
  --ports 8000 \
  --environment-variables GROQ_API_KEY=your_api_key
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GROQ_API_KEY` | Groq API key | - | Yes |
| `GROQ_API_BASE_URL` | Groq API base URL | `https://api.groq.com/openai/v1` | No |
| `DEFAULT_LANGUAGE` | Default recognition language | `en-US` | No |
| `DEFAULT_SAMPLE_RATE` | Audio sample rate | `16000` | No |
| `DEFAULT_CHANNELS` | Audio channels | `1` | No |
| `DEFAULT_TIMEOUT` | Recognition timeout | `30` | No |
| `ENABLE_SEMANTIC_SEGMENTATION` | Enable semantic segmentation | `true` | No |
| `ENABLE_LANGUAGE_IDENTIFICATION` | Enable language detection | `true` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `ENVIRONMENT` | Deployment environment | `production` | No |

### Configuration Files

#### .env File

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
DEFAULT_LANGUAGE=en-US
DEFAULT_SAMPLE_RATE=16000
DEFAULT_CHANNELS=1
DEFAULT_CHUNK_SIZE=1024
DEFAULT_TIMEOUT=30
ENABLE_SEMANTIC_SEGMENTATION=true
ENABLE_LANGUAGE_IDENTIFICATION=true
LOG_LEVEL=INFO
ENVIRONMENT=production
```

#### Docker Compose Environment

```yaml
# docker-compose.yml
environment:
  - GROQ_API_KEY=${GROQ_API_KEY}
  - GROQ_API_BASE_URL=${GROQ_API_BASE_URL:-https://api.groq.com/openai/v1}
  - DEFAULT_LANGUAGE=${DEFAULT_LANGUAGE:-en-US}
  - LOG_LEVEL=${LOG_LEVEL:-INFO}
  - ENVIRONMENT=${ENVIRONMENT:-production}
```

## Monitoring & Observability

### Health Checks

#### API Health Endpoint

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "api_key_configured": true
}
```

#### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Metrics Collection

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'groq-speech-api'
    static_configs:
      - targets: ['groq-speech-api:8000']
    metrics_path: '/metrics'
```

#### Grafana Dashboards

- API request metrics
- Recognition success rates
- Response time distributions
- Error rate monitoring
- Resource utilization

### Logging

#### Log Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

#### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General application information
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## Security

### API Security

#### Authentication

```python
# API key validation
def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key != 'your_groq_api_key_here')
```

#### Rate Limiting

```python
from fastapi import HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/recognize")
@limiter.limit("10/minute")
async def recognize_speech(request: RecognitionRequest):
    # Implementation
    pass
```

#### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Container Security

#### Non-root User

```dockerfile
# Create non-root user
RUN groupadd -r groq && useradd -r -g groq groq
USER groq
```

#### Security Scanning

```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image groq-speech:latest
```

## Scaling

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream groq_speech_api {
    server groq-speech-api-1:8000;
    server groq-speech-api-2:8000;
    server groq-speech-api-3:8000;
}

server {
    listen 80;
    server_name api.groq-speech.com;
    
    location / {
        proxy_pass http://groq_speech_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: groq-speech-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: groq-speech-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Performance Optimization

#### Connection Pooling

```python
import asyncio
from aiohttp import ClientSession

# Reuse HTTP sessions
session = ClientSession()
```

#### Caching

```python
import redis

# Redis caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(key: str, result: dict, ttl: int = 3600):
    redis_client.setex(key, ttl, json.dumps(result))
```

## Troubleshooting

### Common Issues

#### API Key Issues

```bash
# Check API key configuration
echo $GROQ_API_KEY

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

#### Audio Device Issues

```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)['name']}') for i in range(p.get_device_count())]"
```

#### Network Issues

```bash
# Test connectivity
curl -f http://localhost:8000/health

# Check logs
docker-compose logs groq-speech-api
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
python -m api.server --debug
```

## Backup & Recovery

### Data Backup

```bash
# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz \
  .env \
  logs/ \
  uploads/ \
  deployment/
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzf backup-20240101.tar.gz

# Restart services
docker-compose restart
```

## Maintenance

### Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

### Monitoring

```bash
# Check service status
docker-compose ps

# Monitor resource usage
docker stats

# View recent logs
docker-compose logs --tail=100
```

## Support

### Getting Help

- **Documentation**: Check the docs/ directory
- **Issues**: Report on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: support@groq-speech.com

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

This deployment guide covers the essential aspects of deploying the Groq Speech SDK in various environments. For specific requirements or custom configurations, refer to the architecture documentation or contact the development team. 