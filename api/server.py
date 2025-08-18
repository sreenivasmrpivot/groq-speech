"""
FastAPI server for Groq Speech SDK.
Provides REST API and WebSocket endpoints for speech recognition.
Uses EXACTLY the same pattern as CLI: recognize_once_async() and start_continuous_recognition()
"""

import os
import sys
import json
import asyncio
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the parent directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import groq_speech after adding to path
from groq_speech import (
    SpeechConfig,
    SpeechRecognizer,
    ResultReason,
    Config,
)


# Pydantic models for API requests/responses
class RecognitionRequest(BaseModel):
    """Request model for speech recognition."""

    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    model: Optional[str] = Field(None, description="Groq model to use")
    target_language: Optional[str] = Field(
        "en", description="Target language for translation"
    )
    enable_timestamps: bool = Field(False, description="Enable word-level timestamps")
    enable_language_detection: bool = Field(True, description="Enable automatic language detection")


class RecognitionResponse(BaseModel):
    """Response model for speech recognition."""

    success: bool
    text: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    api_key_configured: bool


# Global state
active_connections: List[WebSocket] = []
recognition_sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Starting Groq Speech API Server...")
    try:
        Config.get_api_key()
        print("✅ API key validated")
    except ValueError as e:
        print(f"❌ API key error: {e}")
    yield
    print("🛑 Shutting down Groq Speech API Server...")


# Create FastAPI app
app = FastAPI(
    title="Groq Speech API",
    description="REST API and WebSocket endpoints for Groq Speech SDK",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_speech_config(
    model: Optional[str] = None,
    is_translation: bool = False,
    target_language: str = "en",
) -> SpeechConfig:
    """Create speech configuration - EXACTLY like CLI."""
    config = SpeechConfig()

    # Language auto-detection is enabled by default in SpeechConfig
    # This ensures Groq API automatically detects the language correctly

    # Enable translation if requested - EXACTLY like CLI
    if is_translation:
        config.enable_translation = True
        config.set_translation_target_language(target_language)

    # For transcription mode, ensure language auto-detection is enabled
    # This is the key difference that was causing language misidentification

    return config


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Groq Speech API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        api_key_configured = bool(Config.get_api_key())
    except ValueError:
        api_key_configured = False

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        api_key_configured=api_key_configured,
    )


@app.post("/api/v1/recognize", response_model=RecognitionResponse)
async def recognize_speech(request: RecognitionRequest):
    """Recognize speech from audio data - EXACTLY like CLI file recognition."""
    try:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        print(f"🎤 REST API: Received audio data request")
        print(f"📊 Audio data length: {len(request.audio_data)} characters")
        print(f"🔧 Enable timestamps: {request.enable_timestamps}")
        print(f"🌍 Target language: {request.target_language}")
        print(f"🔍 Enable language detection: {request.enable_language_detection}")

        # Setup speech configuration - EXACTLY like CLI
        speech_config = get_speech_config(
            model=request.model,
            is_translation=False,
            target_language=request.target_language,
        )

        # Create recognizer - EXACTLY like CLI
        recognizer = SpeechRecognizer(speech_config)

        # For REST API, we use recognize_audio_data() like CLI does for microphone input
        # This ensures EXACTLY the same processing as CLI
        import base64
        import numpy as np

        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        print(f"📊 Decoded audio: {len(audio_bytes)} bytes, {len(audio_array)} samples")
        print(
            f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}"
        )

        # Use recognize_audio_data for base64 data - EXACTLY like CLI microphone mode
        print("🎤 Calling groq_speech.recognize_audio_data()...")
        print("🔍 This is the EXACT same call as CLI microphone mode")
        print("🌍 Language detection is enabled by default in SpeechConfig")
        result = recognizer.recognize_audio_data(audio_array_float)
        print(f"🎯 groq_speech result: {result.reason}")

        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognition successful: '{result.text}'")
            print(f"🎯 Confidence: {result.confidence}")
            print(f"🌍 Language: {result.language}")

            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=(result.timestamps if request.enable_timestamps else None),
            )
        elif result.reason == ResultReason.NoMatch:
            print("❌ No speech detected in audio")
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            print(f"❌ Recognition failed: {result.reason}")
            if hasattr(result, "cancellation_details") and result.cancellation_details:
                print(
                    f"🔍 Cancellation details: {result.cancellation_details.error_details}"
                )
            return RecognitionResponse(success=False, error="Recognition failed")

    except Exception as e:
        print(f"💥 Error in REST API recognition: {e}")
        import traceback

        traceback.print_exc()
        return RecognitionResponse(success=False, error=str(e))


@app.post("/api/v1/translate", response_model=RecognitionResponse)
async def translate_speech(request: RecognitionRequest):
    """Translate speech from audio data - EXACTLY like CLI file translation."""
    try:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        # Setup speech configuration for translation - EXACTLY like CLI
        speech_config = get_speech_config(
            model=request.model,
            is_translation=True,
            target_language=request.target_language,
        )

        # Create recognizer - EXACTLY like CLI
        recognizer = SpeechRecognizer(speech_config)

        # Convert base64 audio data
        import base64
        import numpy as np

        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        # Perform translation - EXACTLY like CLI
        result = recognizer.recognize_audio_data(audio_array_float, is_translation=True)

        if result.reason == ResultReason.RecognizedSpeech:
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=(result.timestamps if request.enable_timestamps else None),
            )
        elif result.reason == ResultReason.NoMatch:
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            return RecognitionResponse(success=False, error="Translation failed")

    except Exception as e:
        return RecognitionResponse(success=False, error=str(e))


@app.get("/api/v1/models")
async def get_available_models():
    """Get available Groq models."""
    models = [
        {
            "id": "whisper-large-v3",
            "name": "Whisper Large V3",
            "description": "High accuracy, slower processing",
        },
        {
            "id": "whisper-large-v3-turbo",
            "name": "Whisper Large V3 Turbo",
            "description": "Fast processing, good accuracy",
        },
    ]
    return {"models": models}


@app.websocket("/ws/recognize")
async def websocket_recognition(websocket: WebSocket):
    """WebSocket endpoint for real-time speech recognition - EXACTLY like CLI continuous mode."""
    print("🔌 New WebSocket connection request...")
    await websocket.accept()
    print("✅ WebSocket connection accepted")

    active_connections.append(websocket)
    session_id = str(id(websocket))

    recognition_sessions[session_id] = {
        "websocket": websocket,
        "start_time": datetime.now(),
        "is_recording": False,
        "recognizer": None,
        "mode": "continuous",
        "is_translation": False,
        "target_language": "en",
        "model": "whisper-large-v3-turbo",
    }

    try:
        # Send connection confirmation
        await websocket.send_text(
            json.dumps({"type": "connected", "data": {"session_id": session_id}})
        )

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "start_recognition":
                await handle_start_recognition(session_id, data.get("data", {}))
            elif data["type"] == "stop_recognition":
                await handle_stop_recognition(session_id)
            elif data["type"] == "audio_data":
                await handle_audio_data(session_id, data.get("data", {}))
            elif data["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"💥 WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup
        if websocket in active_connections:
            active_connections.remove(websocket)
        if session_id in recognition_sessions:
            del recognition_sessions[session_id]


# Remove unused queue processing function and callback handlers


async def handle_start_recognition(session_id: str, config: Dict[str, Any]):
    """Handle start recognition request - EXACTLY like CLI single mode but for chunks."""
    try:
        session = recognition_sessions[session_id]
        session["is_recording"] = True
        session["mode"] = config.get("mode", "continuous")
        session["is_translation"] = config.get("is_translation", False)
        session["target_language"] = config.get("target_language", "en")
        session["model"] = config.get("model", "whisper-large-v3-turbo")

        # Setup speech configuration - EXACTLY like CLI
        speech_config = get_speech_config(
            model=session["model"],
            is_translation=session["is_translation"],
            target_language=session["target_language"],
        )

        # Create recognizer - EXACTLY like CLI
        session["recognizer"] = SpeechRecognizer(speech_config)

        # Send confirmation
        await session["websocket"].send_text(
            json.dumps({"type": "recognition_started", "data": {"status": "listening"}})
        )

    except Exception as e:
        await send_recognition_error(session_id, str(e))


async def handle_stop_recognition(session_id: str):
    """Handle stop recognition request - EXACTLY like CLI."""
    if session_id in recognition_sessions:
        session = recognition_sessions[session_id]
        session["is_recording"] = False

        # Stop continuous recognition - EXACTLY like CLI
        if session["recognizer"]:
            session["recognizer"].stop_continuous_recognition()
            session["recognizer"] = None

        await session["websocket"].send_text(
            json.dumps({"type": "recognition_stopped", "data": {"status": "stopped"}})
        )


async def handle_audio_data(session_id: str, data: Dict[str, Any]):
    """Handle incoming audio data for continuous recognition."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]
    if not session.get("is_recording", False):
        return

    try:
        # Get audio data from message
        audio_data = data.get("audio_data")
        if not audio_data:
            print("❌ No audio data received")
            return

        # Convert base64 audio data to numpy array
        import base64
        import numpy as np

        audio_bytes = base64.b64decode(audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        print(
            f"🎤 Received audio chunk: {len(audio_bytes)} bytes, {len(audio_array)} samples"
        )

        # Process audio with groq_speech recognizer using recognize_audio_data
        if session["recognizer"]:
            try:
                print("🎤 Processing audio chunk with groq_speech...")
                result = session["recognizer"].recognize_audio_data(audio_array_float)

                if result.reason == ResultReason.RecognizedSpeech:
                    print(f"✅ Recognition successful: '{result.text}'")

                    # Send result directly to WebSocket
                    result_data = {
                        "type": "recognition_result",
                        "data": {
                            "text": result.text,
                            "confidence": result.confidence or 0.95,
                            "language": result.language or "auto-detected",
                            "timestamps": result.timestamps or [],
                            "timing_metrics": {
                                "api_call": 0,
                                "response_processing": 0,
                                "total_time": 0,
                            },
                        },
                    }

                    await session["websocket"].send_text(json.dumps(result_data))
                    print(f"📤 Result sent to frontend: '{result.text}'")

                elif result.reason == ResultReason.NoMatch:
                    print("❌ No speech detected in audio chunk")
                else:
                    print(f"❌ Recognition failed: {result.reason}")

            except Exception as e:
                print(f"💥 Error in audio recognition: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("❌ No recognizer available for audio processing")

    except Exception as e:
        print(f"💥 Error processing audio data: {e}")
        import traceback

        traceback.print_exc()


# Thread-safe callback handlers for groq_speech events
def handle_recognized_callback(session_id: str, result):
    """Handle recognized speech events - thread-safe callback."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]
    if not session.get("is_recording", False):
        return

    if result.reason == ResultReason.RecognizedSpeech:
        print(f"✅ Recognition successful: '{result.text}'")

        # Put result in queue for async processing
        result_data = {
            "type": "recognition_result",
            "data": {
                "text": result.text,
                "confidence": result.confidence or 0.95,
                "language": result.language or "auto-detected",
                "timestamps": result.timestamps or [],
                "timing_metrics": {
                    "api_call": 0,
                    "response_processing": 0,
                    "total_time": 0,
                },
            },
        }

        try:
            session["result_queue"].put(result_data)
        except Exception as e:
            print(f"Error putting result in queue: {e}")


def handle_canceled_callback(session_id: str, result):
    """Handle canceled recognition events - thread-safe callback."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]
    result_data = {
        "type": "recognition_error",
        "data": {"error": "Recognition canceled"},
    }

    try:
        session["result_queue"].put(result_data)
    except Exception as e:
        print(f"Error putting canceled result in queue: {e}")


def handle_session_started_callback(session_id: str, event):
    """Handle session started events - thread-safe callback."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]
    print(f"🎬 Session started for {session_id}")


def handle_session_stopped_callback(session_id: str, event):
    """Handle session stopped events - thread-safe callback."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]
    print(f"🏁 Session stopped for {session_id}")


async def send_recognition_error(session_id: str, error_message: str):
    """Send error message to WebSocket client."""
    if session_id in recognition_sessions:
        session = recognition_sessions[session_id]
        try:
            await session["websocket"].send_text(
                json.dumps({"type": "error", "data": {"error": error_message}})
            )
        except Exception as e:
            print(f"Error sending error message: {e}")


if __name__ == "__main__":
    print("🚀 Starting Groq Speech API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🌐 WebSocket: ws://localhost:8000/ws/recognize")

    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
