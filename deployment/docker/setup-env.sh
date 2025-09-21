#!/bin/bash

# Groq Speech Environment Setup Script
# This script creates the necessary environment files from templates

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

print_status "Setting up Groq Speech environment files..."

# Change to project root directory
cd ../..

# Create API environment file
if [ ! -f ".env.api" ]; then
    if [ -f "deployment/docker/env.api.template" ]; then
        cp deployment/docker/env.api.template .env.api
        print_success "Created .env.api from template"
    else
        print_error "Template file not found: deployment/docker/env.api.template"
        exit 1
    fi
else
    print_warning ".env.api already exists, skipping..."
fi

# Create UI environment file
if [ ! -f ".env.ui" ]; then
    if [ -f "deployment/docker/env.ui.template" ]; then
        cp deployment/docker/env.ui.template .env.ui
        print_success "Created .env.ui from template"
    else
        print_error "Template file not found: deployment/docker/env.ui.template"
        exit 1
    fi
else
    print_warning ".env.ui already exists, skipping..."
fi

print_success "Environment setup completed!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env.api with your API keys:"
echo "      - GROQ_API_KEY=your_actual_groq_api_key_here"
echo "      - HF_TOKEN=your_actual_huggingface_token_here"
echo ""
echo "   2. Edit .env.ui if needed (defaults should work):"
echo "      - NEXT_PUBLIC_API_URL=http://groq-speech-api:8000"
echo "      - NEXT_PUBLIC_FRONTEND_URL=https://localhost:3443"
echo ""
echo "   3. Run deployment:"
echo "      ./deployment/docker/deploy-local.sh"
