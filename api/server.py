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
import time

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


def get_speech_config(model: Optional[str] = None) -> SpeechConfig:
    """Create speech configuration."""
    config = SpeechConfig()
    # Language auto-detected by Groq API
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
    """Recognize speech from audio data."""
    try:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        # Get speech configuration
        speech_config = get_speech_config(request.model)
        if request.target_language:
            speech_config.set_translation_target_language(request.target_language)

        # Create recognizer
        recognizer = SpeechRecognizer(speech_config)

        # Decode audio data
        import base64

        audio_data = base64.b64decode(request.audio_data)

        # Perform recognition
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=result.timestamps if request.enable_timestamps else None,
            )
        elif result.reason == ResultReason.NoMatch:
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            return RecognitionResponse(success=False, error="Recognition failed")

    except Exception as e:
        return RecognitionResponse(success=False, error=str(e))


@app.post("/api/v1/translate", response_model=RecognitionResponse)
async def translate_speech(request: RecognitionRequest):
    """Translate speech from audio data."""
    try:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        # Get speech configuration for translation
        speech_config = get_speech_config(request.model)
        speech_config.enable_translation = True  # Enable translation mode
        if request.target_language:
            speech_config.set_translation_target_language(request.target_language)

        # Create recognizer
        recognizer = SpeechRecognizer(speech_config)

        # Decode audio data
        import base64

        audio_data = base64.b64decode(request.audio_data)

        # Perform translation
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=result.timestamps if request.enable_timestamps else None,
            )
        elif result.reason == ResultReason.NoMatch:
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            return RecognitionResponse(success=False, error="Translation failed")

    except Exception as e:
        return RecognitionResponse(success=False, error=str(e))


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
            print(f"Received WebSocket message: {message}")
            data = json.loads(message)

            if data["type"] == "start_recognition":
                print(f"Starting recognition for session {session_id}")
                await handle_start_recognition(session_id, data.get("data", {}))
            elif data["type"] == "stop_recognition":
                print(f"Stopping recognition for session {session_id}")
                await handle_stop_recognition(session_id)
            elif data["type"] == "audio_data":
                print(f"Processing audio data for session {session_id}")
                await handle_audio_data(session_id, data.get("data", {}))
            elif data["type"] == "ping":
                print(f"Ping received for session {session_id}")
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                print(f"Unknown message type: {data['type']}")
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

        # Store configuration in session for audio processing
        session["current_model"] = config.get("model")
        session["is_translation"] = config.get("is_translation", False)
        session["target_language"] = config.get("target_language")

        # Send confirmation
        await session["websocket"].send_text(
            json.dumps({"type": "recognition_started", "data": {"status": "listening"}})
        )

        # Start recognition in background thread
        def recognize():
            try:
                # Create speech configuration
                speech_config = get_speech_config(model=config.get("model"))

                # Enable translation if requested
                if config.get("is_translation", False):
                    speech_config.enable_translation = True
                    # Set target language if provided
                    if config.get("target_language"):
                        speech_config.set_translation_target_language(
                            config.get("target_language")
                        )

                # Create recognizer
                recognizer = SpeechRecognizer(speech_config=speech_config)

                # For WebSocket, we need to wait for audio data from the client
                # The actual recognition will happen when audio data is received
                # For now, just send a "ready" message
                try:
                    # Try to get the current event loop (works in main thread)
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no event loop in current thread, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Send ready message
                asyncio.run_coroutine_threadsafe(
                    session["websocket"].send_text(
                        json.dumps(
                            {"type": "ready", "data": {"status": "ready_for_audio"}}
                        )
                    ),
                    loop,
                )

            except Exception as e:
                # Get the main event loop from the main thread
                try:
                    # Try to get the current event loop (works in main thread)
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no event loop in current thread, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Use asyncio.run_coroutine_threadsafe to send error from thread
                asyncio.run_coroutine_threadsafe(
                    send_recognition_error(session_id, str(e)), loop
                )

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


async def handle_audio_data(session_id: str, audio_data: Dict[str, Any]):
    """Handle incoming audio data for recognition using groq_speech SDK - EXACTLY like CLI."""
    import base64
    import tempfile
    import os

    print(f"Received audio data for session {session_id}")

    if session_id not in recognition_sessions:
        print(f"Session {session_id} not found")
        return

    session = recognition_sessions[session_id]
    print(f"Processing audio data for session {session_id}")

    # Decode base64 audio data
    audio_bytes = base64.b64decode(audio_data["audio_data"])
    print(f"Decoded audio data: {len(audio_bytes)} bytes")

    # Get the MIME type from the frontend
    mime_type = audio_data.get("mime_type", "audio/webm")
    print(f"Audio MIME type: {mime_type}")

    # Initialize audio chunks list if it doesn't exist
    if "audio_chunks" not in session:
        session["audio_chunks"] = []
        session["mime_type"] = mime_type
        session["chunk_count"] = 0

    # Add current chunk to the session
    session["audio_chunks"].append(audio_bytes)
    session["chunk_count"] += 1

    print(
        f"Accumulated {session['chunk_count']} audio chunks, total size: {sum(len(chunk) for chunk in session['audio_chunks'])} bytes"
    )

    # Only process when we have enough audio data
    min_chunks = 3
    min_total_size = 50000  # 50KB minimum for a valid audio file

    if (
        session["chunk_count"] >= min_chunks
        and sum(len(chunk) for chunk in session["audio_chunks"]) >= min_total_size
    ):

        try:
            # Combine all accumulated chunks
            combined_audio = b"".join(session["audio_chunks"])
            print(
                f"Processing combined audio: {len(combined_audio)} bytes from {session['chunk_count']} chunks"
            )

            # Use groq_speech SDK EXACTLY like CLI example
            from groq_speech import (
                SpeechConfig,
                SpeechRecognizer,
                ResultReason,
            )

            # Setup speech configuration - EXACTLY like CLI
            config = SpeechConfig()
            model = session.get("current_model", "whisper-large-v3-turbo")
            config.model = model

            # Enable translation if requested - EXACTLY like CLI
            is_translation = session.get("is_translation", False)
            if is_translation:
                config.enable_translation = True
                target_language = session.get("target_language", "en")
                config.set_translation_target_language(target_language)
                print(f"🔀 Translation mode enabled (target: {target_language})")

            print(
                f"Using groq_speech SDK with model: {model}, translation: {is_translation}"
            )

            # Create temporary file for the combined audio data
            audio_file_path = None

            try:
                # Save the combined audio as a temporary file
                # Use the original format extension to maintain compatibility
                if mime_type.startswith("audio/webm"):
                    file_extension = ".webm"
                elif mime_type.startswith("audio/ogg"):
                    file_extension = ".ogg"
                elif mime_type.startswith("audio/mp4"):
                    file_extension = ".m4a"
                else:
                    file_extension = ".webm"  # Default

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                ) as f:
                    f.write(combined_audio)
                    audio_file_path = f.name
                    print(f"Created temporary audio file: {audio_file_path}")

                # Create recognizer - EXACTLY like CLI
                print("🎤 Starting recognition...")
                start_time = time.time()

                # Create recognizer - EXACTLY like CLI (no AudioConfig needed)
                recognizer = SpeechRecognizer(config)

                print(
                    "Created SpeechRecognizer using groq_speech SDK - EXACTLY like CLI"
                )

                # Perform recognition - EXACTLY like CLI
                result = recognizer.recognize_once_async()
                print(f"Recognition result: {result.reason}")

                processing_time = time.time() - start_time

                # Process result - EXACTLY like CLI
                if result.reason == ResultReason.RecognizedSpeech:
                    print(f"✅ Recognition successful!")
                    print(f"📝 Text: {result.text}")
                    print(f"🎯 Confidence: {result.confidence:.2f}")
                    print(f"🌍 Language: {result.language}")
                    print(f"⏱️  Processing time: {processing_time:.2f}s")

                    # Send the result back to the frontend
                    await session["websocket"].send_text(
                        json.dumps(
                            {
                                "type": "recognition_result",
                                "data": {
                                    "text": result.text,
                                    "confidence": result.confidence or 0.95,
                                    "language": result.language or "auto-detected",
                                    "timestamps": result.timestamps or [],
                                    "timing_metrics": {
                                        "api_call": 0,
                                        "response_processing": processing_time,
                                        "total_time": processing_time,
                                    },
                                },
                            }
                        )
                    )

                elif result.reason == ResultReason.NoMatch:
                    print("❌ No speech detected in the audio file")
                    await session["websocket"].send_text(
                        json.dumps(
                            {
                                "type": "recognition_error",
                                "data": {"error": "No speech detected"},
                            }
                        )
                    )

                elif result.reason == ResultReason.Canceled:
                    print(
                        f"❌ Recognition canceled: {result.cancellation_details.error_details}"
                    )
                    await session["websocket"].send_text(
                        json.dumps(
                            {
                                "type": "recognition_error",
                                "data": {"error": "Recognition canceled"},
                            }
                        )
                    )

                # Clear accumulated chunks after successful processing
                session["audio_chunks"] = []
                session["chunk_count"] = 0
                print("Cleared audio chunks after successful processing")

            except Exception as e:
                print(f"❌ Error during recognition: {e}")
                await send_recognition_error(session_id, str(e))

            finally:
                # Clean up the temporary file if it was created
                if audio_file_path and os.path.exists(audio_file_path):
                    try:
                        os.unlink(audio_file_path)
                        print("Cleaned up temporary file")
                    except Exception as cleanup_error:
                        print(f"Error cleaning up temporary file: {cleanup_error}")

        except Exception as e:
            print(f"Error combining audio chunks: {e}")
            await send_recognition_error(session_id, str(e))

    else:
        # Not enough chunks yet, just acknowledge receipt
        print(
            f"Accumulating audio chunks: {session['chunk_count']}/{min_chunks} chunks, {sum(len(chunk) for chunk in session['audio_chunks'])}/{min_total_size} bytes"
        )
        # Send acknowledgment that chunk was received
        await session["websocket"].send_text(
            json.dumps(
                {
                    "type": "chunk_received",
                    "data": {
                        "chunk_count": session["chunk_count"],
                        "total_size": sum(
                            len(chunk) for chunk in session["audio_chunks"]
                        ),
                    },
                }
            )
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
