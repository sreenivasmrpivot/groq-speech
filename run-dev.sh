#!/bin/bash

# Groq Speech SDK - Run Development Environment
# This script sets up and runs both backend and frontend

set -e  # Exit on any error

echo "🚀 Starting Groq Speech SDK Development Environment..."

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

# Check if .env file exists and is configured
check_env() {
    if [ ! -f .env ]; then
        print_warning "Creating .env file..."
        echo "GROQ_API_KEY=your_actual_groq_api_key_here" > .env
        print_error "Please edit .env file with your actual Groq API key!"
        print_error "Then run this script again."
        exit 1
    fi

    if grep -q "GROQ_API_KEY=your_actual_groq_api_key_here" .env; then
        print_error "Please set your actual GROQ_API_KEY in .env file"
        exit 1
    fi

    print_success ".env file configured"
}

# Install Python dependencies in dependency order
install_python_deps() {
    print_status "Installing Python dependencies in dependency order..."
    
    # 1. Install core SDK dependencies
    print_status "Installing core SDK dependencies..."
    pip install -r groq_speech/requirements.txt
    
    # 2. Install core SDK in editable mode
    print_status "Installing core SDK in editable mode..."
    pip install -e .
    
    # 3. Install API server dependencies
    print_status "Installing API server dependencies..."
    pip install -r api/requirements.txt
    
    # 4. Install examples dependencies (if any)
    print_status "Installing examples dependencies..."
    if [ -f "examples/requirements.txt" ]; then
        pip install -r examples/requirements.txt
    fi
    
    # 5. Install development dependencies
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
    
    print_success "Python dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    cd examples/groq-speech-ui
    if [ ! -d "node_modules" ]; then
        npm install
        print_success "Node.js dependencies installed"
    else
        print_status "Node.js dependencies already installed"
    fi
    cd ../..
}

# Start backend server
start_backend() {
    print_status "Starting backend server..."
    # Start backend in background
    python -m api.server &
    BACKEND_PID=$!
    
    # Wait for backend to be ready
    print_status "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend server ready at http://localhost:8000"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend server failed to start"
            kill $BACKEND_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
}

# Start frontend
start_frontend() {
    print_status "Starting frontend..."
    cd examples/groq-speech-ui
    npm run dev &
    FRONTEND_PID=$!
    cd ../..
    
    # Wait for frontend to be ready
    print_status "Waiting for frontend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend ready at http://localhost:3000"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Frontend failed to start"
            kill $FRONTEND_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
}

# Setup signal handlers for cleanup
cleanup() {
    print_status "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    print_success "Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_status "Checking environment..."
    check_env
    
    print_status "Installing dependencies..."
    install_python_deps
    install_node_deps
    
    print_status "Starting services..."
    start_backend
    start_frontend
    
    print_success "🎉 Development environment is ready!"
    echo ""
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo "📖 API Docs: http://localhost:8000/docs"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Keep script running
    wait
}

# Run main function
main "$@"
