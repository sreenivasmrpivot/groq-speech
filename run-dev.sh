#!/bin/bash

# Groq Speech SDK - Run Development Environment
# This script sets up and runs both backend and frontend
# Usage: ./run-dev.sh [--verbose] [--help]

set -e  # Exit on any error

# Default values
VERBOSE=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    echo "Groq Speech SDK - Development Environment Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --verbose, -v    Enable verbose logging for all components"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Run in normal mode"
    echo "  $0 --verbose      # Run with verbose logging"
    exit 0
fi

echo "🚀 Starting Groq Speech SDK Development Environment..."
if [ "$VERBOSE" = true ]; then
    echo "🔍 Verbose mode enabled - detailed logs will be shown"
fi

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
        print_warning "Creating .env file from template..."
        cp .env.template .env
        print_error "Please edit .env file with your actual API keys!"
        print_error "Required: GROQ_API_KEY and HF_TOKEN"
        print_error "Then run this script again."
        exit 1
    fi

    if grep -q "GROQ_API_KEY=your_groq_api_key_here" .env; then
        print_error "Please set your actual GROQ_API_KEY in .env file"
        exit 1
    fi

    if grep -q "HF_TOKEN=your_hf_token_here" .env; then
        print_error "Please set your actual HF_TOKEN in .env file"
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
    
    # Set verbose environment variable if verbose mode is enabled
    if [ "$VERBOSE" = true ]; then
        export GROQ_VERBOSE=true
        export GROQ_LOG_LEVEL=DEBUG
        print_status "Backend verbose logging enabled"
    fi
    
    # Start backend in background with appropriate logging
    if [ "$VERBOSE" = true ]; then
        print_status "Starting backend with verbose logging..."
        python -m api.server 2>&1 | while IFS= read -r line; do
            echo -e "${BLUE}[BACKEND]${NC} $line"
        done &
        BACKEND_PID=$!
    else
        python -m api.server > backend.log 2>&1 &
        BACKEND_PID=$!
    fi
    
    # Wait for backend to be ready
    print_status "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend server ready at http://localhost:8000"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend server failed to start"
            if [ "$VERBOSE" = false ]; then
                print_error "Check backend.log for details"
            fi
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
    
    # Set verbose environment variable if verbose mode is enabled
    if [ "$VERBOSE" = true ]; then
        export NEXT_PUBLIC_VERBOSE=true
        export NEXT_PUBLIC_DEBUG=true
        export NEXT_PUBLIC_LOG_LEVEL=DEBUG
        print_status "Frontend verbose logging enabled"
    fi
    
    # Start frontend with appropriate logging
    if [ "$VERBOSE" = true ]; then
        print_status "Starting frontend with verbose logging..."
        NEXT_PUBLIC_VERBOSE=true NEXT_PUBLIC_DEBUG=true NEXT_PUBLIC_LOG_LEVEL=DEBUG npm run dev 2>&1 | while IFS= read -r line; do
            echo -e "${GREEN}[FRONTEND]${NC} $line"
        done &
        FRONTEND_PID=$!
    else
        npm run dev > frontend.log 2>&1 &
        FRONTEND_PID=$!
    fi
    
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
            if [ "$VERBOSE" = false ]; then
                print_error "Check frontend.log for details"
            fi
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
    if [ "$VERBOSE" = true ]; then
        echo "🔍 Verbose logging enabled - all component logs are shown above"
        echo "📝 Logs are prefixed with [BACKEND] and [FRONTEND] for easy identification"
    else
        echo "📝 Logs are saved to backend.log and frontend.log"
        echo "💡 Use --verbose flag to see real-time logs: ./run-dev.sh --verbose"
    fi
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Keep script running
    wait
}

# Run main function
main "$@"
