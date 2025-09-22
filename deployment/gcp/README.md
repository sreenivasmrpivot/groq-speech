# Groq Speech GCP Deployment

Deploy Groq Speech to Google Cloud using the **exact same Docker images** that work locally.

## 🚀 Quick Start

### Option 1: Cloud Run (CPU only, fastest deployment)
```bash
cd deployment/gcp
./deploy.sh
```

### Option 2: GKE with GPU (for fast diarization)
```bash
cd deployment/gcp
./deploy-gke.sh
```

## 📋 Prerequisites

1. **Google Cloud CLI** installed and authenticated
2. **Working local Docker images**:
   - `docker-groq-speech-api:latest`
   - `docker-groq-speech-ui:latest`
3. **Environment variables** configured in `.env`

## 🔧 Setup

1. **Configure environment**:
   ```bash
   ./setup-env.sh
   ```

2. **Edit `.env`** with your values:
   ```bash
   PROJECT_ID=your-project-id
   REGION=us-central1
   GROQ_API_KEY=your-groq-api-key
   HF_TOKEN=your-hf-token
   ```

## 🎯 Deployment Options

### Cloud Run (Recommended for most users)
- **Pros**: Fast deployment, auto-scaling, pay-per-use
- **Cons**: CPU only (diarization slower)
- **Use case**: Development, testing, low-volume production

### GKE with GPU (Recommended for production)
- **Pros**: GPU support for fast diarization, full control
- **Cons**: More complex, higher cost
- **Use case**: Production with high diarization usage

## 📁 Clean Directory Structure

```
deployment/gcp/
├── deploy.sh           # Cloud Run deployment (CPU)
├── deploy-gke.sh       # GKE deployment (GPU)
├── setup-env.sh        # Environment setup
├── env.template        # Environment template
└── README.md          # This file
```

## 🔍 How It Works

1. **Uses exact same Docker images** that work locally
2. **Tags and pushes** images to Google Container Registry
3. **Deploys** to Cloud Run or GKE
4. **No Dockerfile changes** needed

## 🎯 Key Benefits

- ✅ **Same images locally and in cloud**
- ✅ **No Dockerfile modifications**
- ✅ **GPU support when needed**
- ✅ **Simple and reliable**
- ✅ **Easy to debug**

## 📊 Monitoring

### Check deployment status:
```bash
# Cloud Run
gcloud run services list --region=us-central1

# GKE
kubectl get pods
kubectl get services
```

### View logs:
```bash
# Cloud Run
gcloud run services logs read groq-speech-api --region=us-central1
gcloud run services logs read groq-speech-ui --region=us-central1

# GKE
kubectl logs -l app=groq-speech-api
kubectl logs -l app=groq-speech-ui
```

## 🚨 Troubleshooting

### Common Issues:
1. **Permission denied**: Run `gcloud auth login`
2. **Image not found**: Ensure local images exist with `docker images`
3. **Build fails**: Use the exact same images that work locally

### Reset deployment:
```bash
# Cloud Run
gcloud run services delete groq-speech-api --region=us-central1
gcloud run services delete groq-speech-ui --region=us-central1

# GKE
gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE
```

## 💡 Why This Approach?

- **Reliability**: Uses proven working images
- **Simplicity**: No complex Dockerfile modifications
- **Consistency**: Same behavior locally and in cloud
- **Debugging**: Easy to reproduce issues locally