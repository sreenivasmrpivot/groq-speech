#!/bin/bash
# Safe Chrome launch script for debugging

echo "🚀 Starting Chrome for debugging (safe mode)..."

# Kill any existing Chrome processes
pkill -f "chrome.*remote-debugging-port" || true

# Wait a moment
sleep 2

# Launch Chrome with safe settings
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --user-data-dir="$(pwd)/.vscode/chrome-debug-safe-$(date +%s)" \
  --no-first-run \
  --no-default-browser-check \
  --disable-web-security \
  --disable-features=VizDisplayCompositor \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  --disable-extensions \
  --disable-plugins \
  --disable-default-apps \
  --remote-debugging-port=9222 \
  http://localhost:3000 &

echo "✅ Chrome launched safely!"
echo "You can now use 'Debug Frontend (Manual Browser)' to attach."
