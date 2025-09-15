#!/bin/bash

# GCP CloudRun Deployment Script for Groq Speech SDK with GPU Support
# This script builds and deploys the API server to Google Cloud Run with GPU support

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="groq-speech-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting GCP CloudRun deployment with GPU support${NC}"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}❌ Please set PROJECT_ID environment variable${NC}"
    echo "Usage: PROJECT_ID=your-project-id ./deploy.sh"
    exit 1
fi

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
    exit 1
fi

# Set the project
echo -e "${YELLOW}📋 Setting project to ${PROJECT_ID}${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com

# Build the Docker image
echo -e "${YELLOW}🐳 Building Docker image with GPU support${NC}"
gcloud builds submit --tag ${IMAGE_NAME} --file deployment/docker/Dockerfile.gpu .

# Create secrets if they don't exist
echo -e "${YELLOW}🔐 Setting up secrets${NC}"
if ! gcloud secrets describe groq-secrets &> /dev/null; then
    echo -e "${YELLOW}Creating secrets...${NC}"
    gcloud secrets create groq-secrets --data-file=- <<EOF
groq-api-key=${GROQ_API_KEY}
hf-token=${HF_TOKEN}
EOF
else
    echo -e "${YELLOW}Secrets already exist, updating...${NC}"
    gcloud secrets versions add groq-secrets --data-file=- <<EOF
groq-api-key=${GROQ_API_KEY}
hf-token=${HF_TOKEN}
EOF
fi

# Deploy to CloudRun
echo -e "${YELLOW}🚀 Deploying to CloudRun${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8000 \
    --memory 16Gi \
    --cpu 4 \
    --gpu-type nvidia-t4 \
    --gpu-count 1 \
    --timeout 3600 \
    --concurrency 1 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars CUDA_VISIBLE_DEVICES=0 \
    --set-secrets GROQ_API_KEY=groq-secrets:groq-api-key,HF_TOKEN=groq-secrets:hf-token

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}📊 Health Check: ${SERVICE_URL}/health${NC}"
echo -e "${GREEN}📖 API Docs: ${SERVICE_URL}/docs${NC}"

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment${NC}"
if curl -f "${SERVICE_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Deployment successful! Your Groq Speech API is running on CloudRun with GPU support.${NC}"
