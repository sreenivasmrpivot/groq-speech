"""
FastAPI server for Groq Speech SDK.
Provides REST API and WebSocket endpoints for speech recognition.
"""

import os
import sys
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import websockets

# Add the parent directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import (
    SpeechConfig,
    SpeechRecognizer,
    AudioConfig,
    ResultReason,
    Config,
)


# Pydantic models for API requests/responses
class RecognitionRequest(BaseModel):
    """Request model for speech recognition."""

    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    language: Optional[str] = Field("en-US", description="Recognition language")
    model: Optional[str] = Field(None, description="Groq model to use")
    enable_timestamps: bool = Field(False, description="Enable word-level timestamps")
    enable_language_detection: bool = Field(
        False, description="Enable language detection"
    )


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


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str
    data: Dict[str, Any]


# Global state
active_connections: List[WebSocket] = []
recognition_sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Groq Speech API Server...")
    try:
        Config.get_api_key()
        print("✅ API key validated")
    except ValueError as e:
        print(f"❌ API key error: {e}")

    yield

    # Shutdown
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
    language: str = "en-US", model: Optional[str] = None
) -> SpeechConfig:
    """Create speech configuration."""
    config = SpeechConfig()
    if language:
        config.speech_recognition_language = language
    if model:
        config.set_property("Speech_Recognition_GroqModelId", model)
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
    """REST API endpoint for speech recognition."""
    try:
        # Validate API key
        Config.get_api_key()

        # Create speech configuration
        speech_config = get_speech_config(
            language=request.language, model=request.model
        )

        # Enable additional features if requested
        if request.enable_timestamps:
            speech_config.set_property(
                "Speech_Recognition_EnableWordLevelTimestamps", "true"
            )

        if request.enable_language_detection:
            speech_config.set_property(
                "Speech_Recognition_EnableLanguageIdentification", "true"
            )

        # Create recognizer
        recognizer = SpeechRecognizer(speech_config=speech_config)

        # Perform recognition
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=result.timestamps,
            )
        elif result.reason == ResultReason.NoMatch:
            return RecognitionResponse(success=False, error="No speech detected")
        elif result.reason == ResultReason.Canceled:
            error_msg = "Recognition canceled"
            if result.cancellation_details:
                error_msg += f": {result.cancellation_details.error_details}"
            return RecognitionResponse(success=False, error=error_msg)
        else:
            return RecognitionResponse(
                success=False, error="Unknown recognition result"
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")


@app.post("/api/v1/recognize-file")
async def recognize_audio_file(file_path: str, language: str = "en-US"):
    """Recognize speech from audio file."""
    try:
        # Validate API key
        Config.get_api_key()

        # Create configurations
        speech_config = get_speech_config(language=language)
        audio_config = AudioConfig(filename=file_path)

        # Create recognizer
        recognizer = SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Perform recognition
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=result.timestamps,
            )
        else:
            return RecognitionResponse(
                success=False, error="No speech detected in file"
            )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File recognition error: {str(e)}")


@app.get("/api/v1/models")
async def get_available_models():
    """Get available Groq models."""
    models = [
        {
            "id": "whisper-large-v3",
            "name": "Whisper Large V3",
            "description": "High accuracy, slower processing",
            "languages": [
                "en-US",
                "de-DE",
                "fr-FR",
                "es-ES",
                "it-IT",
                "pt-BR",
                "ja-JP",
                "ko-KR",
                "zh-CN",
                "ru-RU",
            ],
        },
        {
            "id": "whisper-large-v3-turbo",
            "name": "Whisper Large V3 Turbo",
            "description": "Fast processing, good accuracy",
            "languages": [
                "en-US",
                "de-DE",
                "fr-FR",
                "es-ES",
                "it-IT",
                "pt-BR",
                "ja-JP",
                "ko-KR",
                "zh-CN",
                "ru-RU",
            ],
        },
    ]
    return {"models": models}


@app.get("/api/v1/languages")
async def get_supported_languages():
    """Get supported languages."""
    languages = [
        {"code": "en-US", "name": "English (US)"},
        {"code": "de-DE", "name": "German"},
        {"code": "fr-FR", "name": "French"},
        {"code": "es-ES", "name": "Spanish"},
        {"code": "it-IT", "name": "Italian"},
        {"code": "pt-BR", "name": "Portuguese (Brazil)"},
        {"code": "ja-JP", "name": "Japanese"},
        {"code": "ko-KR", "name": "Korean"},
        {"code": "zh-CN", "name": "Chinese (Simplified)"},
        {"code": "ru-RU", "name": "Russian"},
    ]
    return {"languages": languages}


@app.websocket("/ws/recognize")
async def websocket_recognition(websocket: WebSocket):
    """WebSocket endpoint for real-time speech recognition."""
    await websocket.accept()
    active_connections.append(websocket)

    session_id = str(id(websocket))
    recognition_sessions[session_id] = {
        "websocket": websocket,
        "start_time": datetime.now(),
        "transcripts": [],
        "is_recording": False,
    }

    try:
        # Send connection confirmation
        await websocket.send_text(
            json.dumps({"type": "connected", "data": {"session_id": session_id}})
        )

        while True:
            # Receive message from client
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "start_recognition":
                await handle_start_recognition(session_id, data.get("data", {}))
            elif data["type"] == "stop_recognition":
                await handle_stop_recognition(session_id)
            elif data["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "data": {"message": "Unknown message type"}}
                    )
                )

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_text(
            json.dumps({"type": "error", "data": {"message": str(e)}})
        )
    finally:
        # Cleanup
        if websocket in active_connections:
            active_connections.remove(websocket)
        if session_id in recognition_sessions:
            del recognition_sessions[session_id]


async def handle_start_recognition(session_id: str, config: Dict[str, Any]):
    """Handle start recognition request."""
    try:
        session = recognition_sessions[session_id]
        session["is_recording"] = True

        # Send confirmation
        await session["websocket"].send_text(
            json.dumps({"type": "recognition_started", "data": {"status": "listening"}})
        )

        # Start recognition in background thread
        def recognize():
            try:
                # Create speech configuration
                speech_config = get_speech_config(
                    language=config.get("language", "en-US"), model=config.get("model")
                )

                # Create recognizer
                recognizer = SpeechRecognizer(speech_config=speech_config)

                # Perform recognition
                result = recognizer.recognize_once_async()

                # Send result via asyncio
                asyncio.create_task(send_recognition_result(session_id, result))

            except Exception as e:
                asyncio.create_task(send_recognition_error(session_id, str(e)))

        # Run in thread
        thread = threading.Thread(target=recognize)
        thread.daemon = True
        thread.start()

    except Exception as e:
        await send_recognition_error(session_id, str(e))


async def handle_stop_recognition(session_id: str):
    """Handle stop recognition request."""
    if session_id in recognition_sessions:
        recognition_sessions[session_id]["is_recording"] = False
        await recognition_sessions[session_id]["websocket"].send_text(
            json.dumps({"type": "recognition_stopped", "data": {"status": "stopped"}})
        )


async def send_recognition_result(session_id: str, result):
    """Send recognition result to client."""
    if session_id not in recognition_sessions:
        return

    session = recognition_sessions[session_id]

    if result.reason == ResultReason.RecognizedSpeech:
        transcript = {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "timestamp": datetime.now().isoformat(),
            "timestamps": result.timestamps,
        }

        session["transcripts"].append(transcript)

        await session["websocket"].send_text(
            json.dumps({"type": "recognition_result", "data": transcript})
        )

        # Continue recognition if still active
        if session["is_recording"]:
            await handle_start_recognition(session_id, {})

    elif result.reason == ResultReason.NoMatch:
        await session["websocket"].send_text(
            json.dumps(
                {"type": "recognition_error", "data": {"error": "No speech detected"}}
            )
        )

        # Continue recognition if still active
        if session["is_recording"]:
            await handle_start_recognition(session_id, {})

    elif result.reason == ResultReason.Canceled:
        await session["websocket"].send_text(
            json.dumps(
                {"type": "recognition_error", "data": {"error": "Recognition canceled"}}
            )
        )
        session["is_recording"] = False


async def send_recognition_error(session_id: str, error: str):
    """Send recognition error to client."""
    if session_id in recognition_sessions:
        await recognition_sessions[session_id]["websocket"].send_text(
            json.dumps({"type": "recognition_error", "data": {"error": error}})
        )


def create_app() -> FastAPI:
    """Create and configure FastAPI app."""
    return app


def get_app() -> FastAPI:
    """Get the FastAPI app instance."""
    return app


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Groq Speech API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🌐 WebSocket: ws://localhost:8000/ws/recognize")

    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
