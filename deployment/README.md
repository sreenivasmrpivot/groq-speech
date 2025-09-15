# Groq Speech SDK Deployment Guide

This guide covers deploying the Groq Speech SDK API server with GPU support for optimal performance.

## Prerequisites

### Local Development
- Docker and Docker Compose
- Python 3.8+
- Node.js 18+

### GCP CloudRun Deployment
- Google Cloud SDK (`gcloud`)
- GCP project with billing enabled
- Required APIs enabled (Cloud Build, Cloud Run, Container Registry)

## Local Development with Docker

### CPU-only Development
```bash
# Build and run the API server
cd deployment/docker
docker-compose up --build

# The API will be available at http://localhost:8000
```

### GPU-enabled Development
```bash
# Build and run with GPU support
cd deployment/docker
docker-compose -f docker-compose.gpu.yml up --build

# The API will be available at http://localhost:8000
```

### Frontend Development
```bash
# In a separate terminal
cd examples/groq-speech-ui
npm install
npm run dev

# The UI will be available at http://localhost:3000
```

## GCP CloudRun Deployment

### 1. Set Environment Variables
```bash
export PROJECT_ID="your-gcp-project-id"
export GROQ_API_KEY="your-groq-api-key"
export HF_TOKEN="your-huggingface-token"
```

### 2. Deploy with GPU Support
```bash
cd deployment/gcp
./deploy.sh
```

### 3. Manual Deployment (Alternative)
```bash
# Build the image
gcloud builds submit --tag gcr.io/${PROJECT_ID}/groq-speech-api --file deployment/docker/Dockerfile.gpu .

# Deploy to CloudRun
gcloud run deploy groq-speech-api \
    --image gcr.io/${PROJECT_ID}/groq-speech-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 16Gi \
    --cpu 4 \
    --gpu-type nvidia-t4 \
    --gpu-count 1 \
    --timeout 3600 \
    --concurrency 1 \
    --max-instances 10 \
    --min-instances 0
```

## GPU Support Configuration

The deployment automatically detects and configures GPU support for Pyannote.audio:

- **CUDA Detection**: Automatically detects available CUDA devices
- **Device Selection**: Uses GPU if available, falls back to CPU
- **Memory Management**: Optimized for CloudRun's GPU memory limits
- **Performance**: Significantly faster diarization processing

### GPU Requirements
- **CloudRun**: NVIDIA T4 GPU (recommended)
- **Local**: NVIDIA GPU with CUDA 11.8+ support
- **Memory**: Minimum 4GB GPU memory for diarization

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Groq API key for speech recognition | Yes | - |
| `HF_TOKEN` | Hugging Face token for Pyannote.audio | Yes | - |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | No | `0` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

## Monitoring and Logs

### View Logs
```bash
# CloudRun logs
gcloud logs read --service=groq-speech-api --limit=50

# Docker logs
docker-compose logs -f
```

### Health Check
```bash
curl https://your-service-url/health
```

### API Documentation
Visit `https://your-service-url/docs` for interactive API documentation.

## Performance Optimization

### GPU Optimization
- Use `nvidia-t4` GPU type for best price/performance
- Set `concurrency=1` to avoid GPU memory conflicts
- Monitor GPU utilization in CloudRun metrics

### Memory Optimization
- 16GB RAM recommended for diarization workloads
- Adjust `max-instances` based on expected load
- Use `min-instances=0` for cost optimization

### Cost Optimization
- Set appropriate `max-instances` limit
- Use `min-instances=0` for auto-scaling
- Monitor usage patterns and adjust accordingly

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   - Check CloudRun region supports GPUs
   - Verify GPU quota in GCP console
   - Ensure proper GPU type selection

2. **Memory Issues**
   - Increase memory allocation
   - Reduce concurrency settings
   - Check for memory leaks in logs

3. **API Timeouts**
   - Increase timeout settings
   - Check network connectivity
   - Monitor processing times

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Test API endpoints
curl -X POST "https://your-service-url/api/v1/recognize" \
  -H "Content-Type: application/json" \
  -d '{"audio_data": [0.1, 0.2, 0.3], "sample_rate": 16000}'

# View detailed logs
gcloud logs read --service=groq-speech-api --severity=ERROR
```

## Security Considerations

- Use GCP Secret Manager for sensitive data
- Enable authentication if needed
- Configure CORS properly for production
- Monitor API usage and set rate limits

## Scaling

### Horizontal Scaling
- Adjust `max-instances` based on load
- Use load balancing for multiple regions
- Monitor costs and performance

### Vertical Scaling
- Increase memory/CPU allocation
- Use more powerful GPU types
- Optimize processing algorithms

## Support

For issues and questions:
- Check the logs first
- Review this documentation
- Test locally before deploying
- Monitor CloudRun metrics and logs
