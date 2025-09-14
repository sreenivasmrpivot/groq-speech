#!/bin/bash

# Start the Next.js frontend with debugging enabled
# This script is used with the "Attach to Frontend" debug configuration

echo "🚀 Starting Next.js frontend with debugging enabled..."

cd examples/groq-speech-ui

# Set Node.js debugging options
export NODE_OPTIONS="--inspect"

# Start the development server
npm run dev
