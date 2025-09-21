#!/bin/bash

# Groq Speech GCP Cloud Run Deployment Script
# This script builds and deploys both API and UI to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    print_error "PROJECT_ID environment variable is not set"
    print_error "Please set it with: export PROJECT_ID=your-project-id"
    exit 1
fi

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

print_status "Starting GCP Cloud Run deployment for project: $PROJECT_ID"

# Set default region
REGION=${REGION:-us-central1}
print_status "Using region: $REGION"

# Enable required APIs
print_status "Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create secrets if they don't exist
print_status "Setting up secrets..."
if ! gcloud secrets describe groq-speech-secrets --project=$PROJECT_ID >/dev/null 2>&1; then
    print_status "Creating secrets..."
    gcloud secrets create groq-speech-secrets --project=$PROJECT_ID
fi

# Check if secrets are set
if ! gcloud secrets versions list groq-speech-secrets --project=$PROJECT_ID --format="value(name)" | grep -q .; then
    print_warning "No secret versions found. Please add your secrets:"
    print_warning "gcloud secrets versions add groq-speech-secrets --data-file=- <<< 'your-groq-api-key'"
    print_warning "gcloud secrets versions add groq-speech-secrets --data-file=- <<< 'your-hf-token'"
    exit 1
fi

# Build and push API image
print_status "Building and pushing API image..."
cd ../..
gcloud builds submit --tag gcr.io/$PROJECT_ID/groq-speech-api:latest -f deployment/docker/Dockerfile.api .

# Deploy API to Cloud Run
print_status "Deploying API to Cloud Run..."
gcloud run deploy groq-speech-api \
    --image gcr.io/$PROJECT_ID/groq-speech-api:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --concurrency 10 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars "GROQ_API_BASE=https://api.groq.com/openai/v1,GROQ_MODEL_ID=whisper-large-v3,GROQ_TEMPERATURE=0.0,AUDIO_SAMPLE_RATE=16000,AUDIO_CHANNELS=1,MAX_AUDIO_FILE_SIZE=25000000,DIARIZATION_MIN_SEGMENT_DURATION=2.0,DIARIZATION_SILENCE_THRESHOLD=0.8,DIARIZATION_MAX_SEGMENTS_PER_CHUNK=8,DIARIZATION_CHUNK_STRATEGY=adaptive,DIARIZATION_MAX_SPEAKERS=5,API_HOST=0.0.0.0,API_PORT=8000,API_WORKERS=1,API_LOG_LEVEL=info,LOG_LEVEL=INFO,ENVIRONMENT=production" \
    --set-secrets "GROQ_API_KEY=groq-speech-secrets:latest,HF_TOKEN=groq-speech-secrets:latest"

# Get API URL
API_URL=$(gcloud run services describe groq-speech-api --platform managed --region $REGION --format 'value(status.url)')
print_success "API deployed to: $API_URL"

# Build and push UI image
print_status "Building and pushing UI image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/groq-speech-ui:latest -f deployment/docker/Dockerfile.ui .

# Deploy UI to Cloud Run
print_status "Deploying UI to Cloud Run..."
gcloud run deploy groq-speech-ui \
    --image gcr.io/$PROJECT_ID/groq-speech-ui:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --concurrency 100 \
    --max-instances 10 \
    --timeout 60 \
    --set-env-vars "NEXT_PUBLIC_API_URL=$API_URL,NEXT_PUBLIC_FRONTEND_URL=https://groq-speech-ui-XXXXX-uc.a.run.app,NEXT_PUBLIC_VERBOSE=false,NEXT_PUBLIC_DEBUG=false,NODE_ENV=production,NEXT_TELEMETRY_DISABLED=1"

# Get UI URL
UI_URL=$(gcloud run services describe groq-speech-ui --platform managed --region $REGION --format 'value(status.url)')
print_success "UI deployed to: $UI_URL"

# Update UI with correct frontend URL
print_status "Updating UI with correct frontend URL..."
gcloud run services update groq-speech-ui \
    --platform managed \
    --region $REGION \
    --set-env-vars "NEXT_PUBLIC_FRONTEND_URL=$UI_URL"

print_success "Deployment completed successfully!"
echo ""
echo "🌐 Services are running:"
echo "   API: $API_URL"
echo "   UI:  $UI_URL"
echo "   API Docs: $API_URL/docs"
echo ""
echo "📋 Useful commands:"
echo "   View API logs: gcloud run logs read groq-speech-api --region $REGION"
echo "   View UI logs: gcloud run logs read groq-speech-ui --region $REGION"
echo "   Delete services: gcloud run services delete groq-speech-api groq-speech-ui --region $REGION"
