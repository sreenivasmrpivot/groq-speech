#!/bin/bash

# Safe Chrome Debug Starter
# This script starts Chrome with debugging enabled, with additional safety checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEBUG_PROFILE_DIR="$HOME/.chrome-debug-profile"
CHROME_DEBUG_PORT=9222
CHROME_DEBUG_URL="http://localhost:3000"
CHROME_BINARY="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

echo -e "${BLUE}🚀 Starting Chrome with debugging enabled (Safe Mode)...${NC}"

# Check if Chrome binary exists
if [ ! -f "$CHROME_BINARY" ]; then
    echo -e "${RED}❌ Chrome binary not found at: $CHROME_BINARY${NC}"
    exit 1
fi

# Check if debug profile exists
if [ ! -d "$DEBUG_PROFILE_DIR" ]; then
    echo -e "${RED}❌ Debug profile not found. Please run prepare-chrome-debug.sh first.${NC}"
    exit 1
fi

# Check if port is available
if lsof -Pi :$CHROME_DEBUG_PORT -sTCP:LISTEN -t >/dev/null; then
    echo -e "${YELLOW}⚠️ Port $CHROME_DEBUG_PORT is already in use${NC}"
    echo -e "${YELLOW}🔄 Attempting to free the port...${NC}"
    lsof -ti:$CHROME_DEBUG_PORT | xargs kill -9 || true
    sleep 2
fi

# Kill any existing Chrome processes
echo -e "${YELLOW}🔄 Closing existing Chrome processes...${NC}"
pkill -f "Google Chrome" || true
sleep 3

# Verify Chrome is closed
if pgrep -f "Google Chrome" > /dev/null; then
    echo -e "${YELLOW}⚠️ Chrome processes still running, force killing...${NC}"
    pkill -9 -f "Google Chrome" || true
    sleep 2
fi

# Start Chrome with debugging
echo -e "${GREEN}🔧 Starting Chrome with debug profile...${NC}"
"$CHROME_BINARY" \
    --user-data-dir="$DEBUG_PROFILE_DIR" \
    --remote-debugging-port=$CHROME_DEBUG_PORT \
    --disable-web-security \
    --disable-features=VizDisplayCompositor \
    --enable-logging \
    --v=1 \
    --no-first-run \
    --no-default-browser-check \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-features=TranslateUI \
    --disable-ipc-flooding-protection \
    --enable-automation \
    --disable-extensions \
    --disable-plugins \
    --disable-images \
    --disable-javascript-harmony-shipping \
    --disable-background-networking \
    --disable-sync \
    --metrics-recording-only \
    --no-report-upload \
    --disable-default-apps \
    --mute-audio \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --remote-allow-origins=* \
    --disable-background-mode \
    --disable-component-extensions-with-background-pages \
    --disable-default-apps \
    --disable-extensions \
    --disable-sync \
    --disable-translate \
    --hide-scrollbars \
    --mute-audio \
    --no-first-run \
    --no-pings \
    --no-zygote \
    --single-process \
    "$CHROME_DEBUG_URL" &

# Wait for Chrome to start
echo -e "${YELLOW}⏳ Waiting for Chrome to start...${NC}"
sleep 5

# Check if Chrome is running
if pgrep -f "Google Chrome" > /dev/null; then
    echo -e "${GREEN}✅ Chrome started successfully with debugging enabled${NC}"
    echo -e "${BLUE}🌐 Debug URL: http://localhost:$CHROME_DEBUG_PORT${NC}"
    echo -e "${BLUE}🎯 Target URL: $CHROME_DEBUG_URL${NC}"
    echo -e "${YELLOW}💡 You can now attach VS Code debugger to this Chrome instance${NC}"
    
    # Test debug port
    if curl -s "http://localhost:$CHROME_DEBUG_PORT/json" > /dev/null; then
        echo -e "${GREEN}✅ Debug port is responding${NC}"
    else
        echo -e "${YELLOW}⚠️ Debug port may not be ready yet${NC}"
    fi
else
    echo -e "${RED}❌ Failed to start Chrome${NC}"
    exit 1
fi
