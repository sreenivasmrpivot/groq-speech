#!/bin/bash

# Script to start Chrome with debugging enabled for Next.js development
# This helps with the "Unable to attach to browser" error

echo "🚀 Starting Chrome with debugging enabled..."
echo "📱 Make sure Next.js dev server is running on http://localhost:3000"
echo "🔧 Chrome will open with debugging port 9222"
echo ""

# Create debug profile directory if it doesn't exist
mkdir -p .vscode/chrome-debug-profile

# Start Chrome with debugging enabled
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=.vscode/chrome-debug-profile \
  --disable-web-security \
  --disable-features=VizDisplayCompositor \
  --no-first-run \
  --no-default-browser-check \
  http://localhost:3000 &

echo "✅ Chrome started with debugging enabled"
echo "🔍 You can now use 'Debug Frontend (Manual Browser)' in VS Code"
echo "📝 Or attach to the browser manually using port 9222"
