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
import uuid
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
        True, description="Enable automatic language detection"
    )
    enable_diarization: bool = Field(False, description="Enable speaker diarization")


class RecognitionResponse(BaseModel):
    """Response model for speech recognition."""

    success: bool
    text: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    num_speakers: Optional[int] = None
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
        SpeechConfig.get_api_key()
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
        api_key_configured = bool(SpeechConfig.get_api_key())
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
        print(f"🎭 Enable diarization: {request.enable_diarization}")

        # Setup speech configuration - EXACTLY like CLI
        speech_config = get_speech_config(
            model=request.model,
            is_translation=False,
            target_language=request.target_language,
        )

        # Create recognizer - EXACTLY like CLI
        recognizer = SpeechRecognizer(speech_config)

        # Decode audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf

        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        print(f"📊 Decoded audio: {len(audio_bytes)} bytes, {len(audio_array)} samples")
        print(f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}")

        # Process based on diarization requirement
        if request.enable_diarization:
            # For diarization, save to temp file and use process_file
            print("🎭 Processing with diarization...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array_float, 16000)
            
            try:
                result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=False)
            finally:
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use direct audio data processing
            print("🎤 Processing without diarization...")
            result = recognizer.recognize_audio_data(audio_array_float, is_translation=False)

        # Handle diarization result
        if hasattr(result, "segments") and result.segments:
            print(f"✅ Diarization successful: {len(result.segments)} segments, {result.num_speakers} speakers")
            
            # Convert segments to API format
            segments_data = []
            for segment in result.segments:
                segment_data = {
                    "speaker_id": segment.speaker_id,
                    "text": getattr(segment, 'text', '') or getattr(segment, 'transcription', '') or '[No text]',
                    "start_time": getattr(segment, 'start_time', 0),
                    "end_time": getattr(segment, 'end_time', 0)
                }
                segments_data.append(segment_data)

            return RecognitionResponse(
                success=True,
                text=None,  # No single text for diarization
                confidence=None,
                language=result.language if hasattr(result, 'language') else None,
                timestamps=None,
                segments=segments_data,
                num_speakers=result.num_speakers
            )

        # Handle regular recognition result
        elif result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognition successful: '{result.text}'")
            print(f"🎯 Confidence: {result.confidence}")
            print(f"🌍 Language: {result.language}")

            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=(result.timestamps if request.enable_timestamps else None),
                segments=None,
                num_speakers=None
            )
        elif result.reason == ResultReason.NoMatch:
            print("❌ No speech detected in audio")
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            print(f"❌ Recognition failed: {result.reason}")
            if hasattr(result, "cancellation_details") and result.cancellation_details:
                print(f"🔍 Cancellation details: {result.cancellation_details.error_details}")
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

        print(f"🔀 Translation API: Received audio data request")
        print(f"📊 Audio data length: {len(request.audio_data)} characters")
        print(f"🎭 Enable diarization: {request.enable_diarization}")

        # Setup speech configuration for translation - EXACTLY like CLI
        speech_config = get_speech_config(
            model=request.model,
            is_translation=True,
            target_language=request.target_language,
        )

        # Enable translation - EXACTLY like CLI
        speech_config.enable_translation = True
        print(f"🔀 Translation mode enabled (target: {request.target_language})")

        # Create recognizer - EXACTLY like CLI (AFTER enabling translation)
        recognizer = SpeechRecognizer(speech_config)

        # Decode audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf

        audio_bytes = base64.b64decode(request.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array_float = audio_array.astype(np.float32) / 32768.0

        print(f"📊 Decoded audio: {len(audio_bytes)} bytes, {len(audio_array)} samples")

        # Process based on diarization requirement
        if request.enable_diarization:
            # For diarization, save to temp file and use process_file
            print("🎭 Processing translation with diarization...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array_float, 16000)
            
            try:
                result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=True)
            finally:
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use direct audio data processing
            print("🔀 Processing translation without diarization...")
            result = recognizer.recognize_audio_data(audio_array_float, is_translation=True)

        # Handle diarization result
        if hasattr(result, "segments") and result.segments:
            print(f"✅ Translation diarization successful: {len(result.segments)} segments, {result.num_speakers} speakers")
            
            # Convert segments to API format
            segments_data = []
            for segment in result.segments:
                segment_data = {
                    "speaker_id": segment.speaker_id,
                    "text": getattr(segment, 'text', '') or getattr(segment, 'transcription', '') or '[No text]',
                    "start_time": getattr(segment, 'start_time', 0),
                    "end_time": getattr(segment, 'end_time', 0)
                }
                segments_data.append(segment_data)

            return RecognitionResponse(
                success=True,
                text=None,  # No single text for diarization
                confidence=None,
                language=result.language if hasattr(result, 'language') else None,
                timestamps=None,
                segments=segments_data,
                num_speakers=result.num_speakers
            )

        # Handle regular translation result
        elif result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Translation successful: '{result.text}'")
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                timestamps=(result.timestamps if request.enable_timestamps else None),
                segments=None,
                num_speakers=None
            )
        elif result.reason == ResultReason.NoMatch:
            return RecognitionResponse(success=False, error="No speech detected")
        else:
            return RecognitionResponse(success=False, error="Translation failed")

    except Exception as e:
        print(f"💥 Error in REST API translation: {e}")
        import traceback
        traceback.print_exc()
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
    """WebSocket endpoint for continuous speech recognition - EXACTLY like CLI."""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        print(f"🔌 WebSocket connected: {session_id}")

        # Initialize session
        recognition_sessions[session_id] = {
            "websocket": websocket,
            "recognizer": None,
            "is_recording": False,
        }

        # Wait for start message
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "start_recognition":
                    await handle_start_recognition(session_id, data)
                elif message_type == "stop_recognition":
                    await handle_stop_recognition(session_id)
                elif message_type == "single_recognition":
                    await handle_single_recognition(session_id, data)
                elif message_type == "file_recognition":
                    await handle_file_recognition(session_id, data)
                elif message_type == "ping":
                    try:
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    except Exception as e:
                        print(f"❌ Error sending pong: {e}")
                else:
                    print(f"❌ Unknown message type: {message_type}")

            except json.JSONDecodeError:
                print("❌ Invalid JSON message")
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                try:
                    await websocket.send_text(
                        json.dumps({"type": "error", "error": str(e)})
                    )
                except Exception as send_error:
                    print(f"❌ Error sending error message: {send_error}")

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        # Cleanup session
        if session_id in recognition_sessions:
            await cleanup_session(session_id)


async def process_result_queue(session_id: str):
    """Process results from the thread-safe queue and send to frontend."""
    try:
        session = recognition_sessions.get(session_id)
        if not session:
            return

        websocket = session["websocket"]
        result_queue = session.get("result_queue")

        if not result_queue:
            return

        print(f"🔄 Starting result queue processor for session: {session_id}")

        while session.get("is_recording", False):
            try:
                # Check for results in the queue (non-blocking)
                try:
                    event_type, data = result_queue.get_nowait()
                except queue.Empty:
                    # No results yet, wait a bit
                    await asyncio.sleep(0.1)
                    continue

                print(f"📨 Processing {event_type} event from queue")

                # Process the event based on type
                if event_type == "recognized":
                    if data.reason == ResultReason.RecognizedSpeech:
                        print(f"📝 Recognition result: {data.text}")

                        # Check if WebSocket is still open before sending
                        try:
                            if websocket.client_state.value == 1:  # WebSocket.OPEN
                                # Send result to frontend
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "recognition_result",
                                            "data": {
                                                "text": data.text,
                                                "confidence": data.confidence,
                                                "language": data.language,
                                                "timestamps": (
                                                    data.timestamps
                                                    if hasattr(data, "timestamps")
                                                    else None
                                                ),
                                                "is_translation": False,
                                                "enable_diarization": False
                                            }
                                        }
                                    )
                                )
                        except Exception as e:
                            print(f"❌ Error sending recognition result: {e}")
                            break
                    elif data.reason == ResultReason.NoMatch:
                        print("❌ No speech detected")
                        try:
                            if websocket.client_state.value == 1:  # WebSocket.OPEN
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "no_speech",
                                            "data": {
                                                "message": "No speech detected",
                                            }
                                        }
                                    )
                                )
                        except Exception as e:
                            print(f"❌ Error sending no speech message: {e}")
                            break

                elif event_type == "canceled":
                    if (
                        hasattr(data, "cancellation_details")
                        and data.cancellation_details
                    ):
                        error_msg = data.cancellation_details.error_details
                    else:
                        error_msg = "Recognition canceled"

                    print(f"❌ Recognition canceled: {error_msg}")
                    try:
                        if websocket.client_state.value == 1:  # WebSocket.OPEN
                            await websocket.send_text(
                                json.dumps(
                                    {"type": "recognition_canceled", "data": {"error": error_msg}}
                                )
                            )
                    except Exception as e:
                        print(f"❌ Error sending cancellation message: {e}")
                        break

                elif event_type == "session_started":
                    print(f"🎬 Recognition session started for: {session_id}")
                    try:
                        if websocket.client_state.value == 1:  # WebSocket.OPEN
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_started",
                                        "data": {
                                            "message": "Recognition session started",
                                        }
                                    }
                                )
                            )
                    except Exception as e:
                        print(f"❌ Error sending session started message: {e}")
                        break

                elif event_type == "session_stopped":
                    print(f"🏁 Recognition session stopped for: {session_id}")
                    try:
                        if websocket.client_state.value == 1:  # WebSocket.OPEN
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_stopped",
                                        "data": {
                                            "message": "Recognition session stopped",
                                        }
                                    }
                                )
                            )
                    except Exception as e:
                        print(f"❌ Error sending session stopped message: {e}")
                        break

            except Exception as e:
                print(f"❌ Error processing result from queue: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        print(f"❌ Error in result queue processor: {e}")
    finally:
        print(f"🔄 Result queue processor stopped for session: {session_id}")


async def handle_start_recognition(session_id: str, data: dict):
    """Start continuous recognition - EXACTLY like CLI."""
    try:
        session = recognition_sessions[session_id]
        websocket = session["websocket"]

        print(f"🎤 Starting continuous recognition for session: {session_id}")

        # Get recognition parameters
        is_translation = data.get("is_translation", False)
        enable_diarization = data.get("enable_diarization", False)
        target_language = data.get("target_language", "en")

        # Setup speech configuration - EXACTLY like CLI
        speech_config = get_speech_config(
            model=None,
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer - EXACTLY like CLI
        recognizer = SpeechRecognizer(speech_config)

        # Configure translation if needed - EXACTLY like CLI
        if is_translation:
            speech_config.enable_translation = True
            print(f"🔀 Translation mode enabled (target: {target_language})")

        session["recognizer"] = recognizer

        # Set up event handlers - EXACTLY like CLI
        # Use thread-safe queue for communication between groq_speech callbacks and main event loop
        result_queue = queue.Queue()

        def on_recognized(result):
            """Thread-safe callback for recognized speech."""
            try:
                result_queue.put(("recognized", result))
            except Exception as e:
                print(f"❌ Error in recognized callback: {e}")

        def on_canceled(result):
            """Thread-safe callback for canceled recognition."""
            try:
                result_queue.put(("canceled", result))
            except Exception as e:
                print(f"❌ Error in canceled callback: {e}")

        def on_session_started(event):
            """Thread-safe callback for session started."""
            try:
                result_queue.put(("session_started", event))
            except Exception as e:
                print(f"❌ Error in session started callback: {e}")

        def on_session_stopped(event):
            """Thread-safe callback for session stopped."""
            try:
                result_queue.put(("session_stopped", event))
            except Exception as e:
                print(f"❌ Error in session stopped callback: {e}")

        # Connect the thread-safe callbacks
        recognizer.connect("recognized", on_recognized)
        recognizer.connect("canceled", on_canceled)
        recognizer.connect("session_started", on_session_started)
        recognizer.connect("session_stopped", on_session_stopped)

        # Store the queue in the session for processing
        session["result_queue"] = result_queue

        # Start continuous recognition - EXACTLY like CLI
        print("🎤 Calling recognizer.start_continuous_recognition()...")
        recognizer.start_continuous_recognition()

        session["is_recording"] = True

        # Start background task to process results from the queue
        asyncio.create_task(process_result_queue(session_id))

        # Send success message
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "recognition_started",
                        "data": {
                            "message": "Continuous recognition started",
                        }
                    }
                )
            )
        except Exception as e:
            print(f"❌ Error sending recognition started message: {e}")

        print(f"✅ Continuous recognition started for session: {session_id}")

    except Exception as e:
        print(f"❌ Error starting recognition: {e}")
        try:
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "data": {"error": f"Failed to start recognition: {str(e)}"}}
                )
            )
        except Exception as send_error:
            print(f"❌ Error sending error message: {send_error}")


async def handle_stop_recognition(session_id: str):
    """Stop continuous recognition - EXACTLY like CLI."""
    try:
        session = recognition_sessions.get(session_id)
        websocket = session["websocket"]

        print(f"🛑 Stopping continuous recognition for session: {session_id}")

        if session["recognizer"] and session.get("is_recording", False):
            # Stop continuous recognition - EXACTLY like CLI
            session["recognizer"].stop_continuous_recognition()
            session["is_recording"] = False

            # Check if WebSocket is still open before sending
            try:
                if websocket.client_state.value == 1:  # WebSocket.OPEN
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "recognition_stopped",
                                "data": {
                                    "message": "Continuous recognition stopped",
                                }
                            }
                        )
                    )
            except Exception as e:
                print(f"❌ Error sending recognition stopped message: {e}")

            print(f"✅ Continuous recognition stopped for session: {session_id}")
        else:
            try:
                if websocket.client_state.value == 1:  # WebSocket.OPEN
                    await websocket.send_text(
                        json.dumps(
                            {"type": "error", "data": {"error": "No active recognition session"}}
                        )
                    )
            except Exception as e:
                print(f"❌ Error sending no active session message: {e}")

    except Exception as e:
        print(f"❌ Error stopping recognition: {e}")
        # Don't try to send error message if WebSocket is already closed
        try:
            if websocket.client_state.value == 1:  # WebSocket.OPEN
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"error": f"Failed to stop recognition: {str(e)}"},
                        }
                    )
                )
        except Exception as send_error:
            print(f"❌ Error sending stop error message: {send_error}")


async def handle_single_recognition(session_id: str, data: dict):
    """Handle single microphone recognition - record then process."""
    try:
        session = recognition_sessions[session_id]
        websocket = session["websocket"]

        print(f"🎤 Starting single recognition for session: {session_id}")

        # Get recognition parameters
        is_translation = data.get("is_translation", False)
        enable_diarization = data.get("enable_diarization", False)
        target_language = data.get("target_language", "en")

        # Setup speech configuration
        speech_config = get_speech_config(
            model=None,
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer
        recognizer = SpeechRecognizer(speech_config)

        # Configure translation if needed
        if is_translation:
            speech_config.enable_translation = True
            print(f"🔀 Translation mode enabled (target: {target_language})")

        session["recognizer"] = recognizer

        # Send acknowledgment
        try:
            await websocket.send_text(
                json.dumps({
                    "type": "single_recognition_started",
                    "data": {
                        "message": "Single recognition started - speak now",
                        "enable_diarization": enable_diarization,
                        "is_translation": is_translation
                    }
                })
            )
        except Exception as e:
            print(f"❌ Error sending single recognition started message: {e}")

        print(f"✅ Single recognition started for session: {session_id}")

    except Exception as e:
        print(f"❌ Error starting single recognition: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": {"error": str(e)}})
            )
        except Exception as send_error:
            print(f"❌ Error sending single recognition error message: {send_error}")


async def handle_file_recognition(session_id: str, data: dict):
    """Handle file-based recognition via WebSocket."""
    try:
        session = recognition_sessions[session_id]
        websocket = session["websocket"]

        print(f"📁 Processing file recognition for session: {session_id}")

        # Get recognition parameters
        audio_data = data.get("audio_data")
        is_translation = data.get("is_translation", False)
        enable_diarization = data.get("enable_diarization", False)
        target_language = data.get("target_language", "en")

        if not audio_data:
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": {"error": "No audio data provided"}})
                )
            except Exception as e:
                print(f"❌ Error sending no audio data error: {e}")
            return

        # Setup speech configuration
        speech_config = get_speech_config(
            model=None,
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer
        recognizer = SpeechRecognizer(speech_config)

        # Configure translation if needed
        if is_translation:
            speech_config.enable_translation = True
            print(f"🔀 Translation mode enabled (target: {target_language})")

        # Decode and process audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Process audio
        if enable_diarization:
            # For diarization, save to temp file and use process_file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array, 16000)
            
            try:
                result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=is_translation)
            finally:
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use direct audio data processing
            result = recognizer.recognize_audio_data(audio_array, 16000, is_translation=is_translation)

        # Send result
        try:
            if result:
                if hasattr(result, "text") and result.text:
                    await websocket.send_text(
                        json.dumps({
                            "type": "recognition_result",
                            "data": {
                                "text": result.text,
                                "confidence": getattr(result, "confidence", None),
                                "language": getattr(result, "language", None),
                                "is_translation": is_translation,
                                "enable_diarization": False
                            }
                        })
                    )
                elif hasattr(result, "segments") and result.segments:
                    # Diarization result
                    segments_data = []
                    for segment in result.segments:
                        segments_data.append({
                            "speaker_id": segment.speaker_id,
                            "text": getattr(segment, "text", "") or getattr(segment, "transcription", "") or "[No text]",
                            "start_time": getattr(segment, "start_time", 0),
                            "end_time": getattr(segment, "end_time", 0)
                        })
                    
                    await websocket.send_text(
                        json.dumps({
                            "type": "diarization_result",
                            "data": {
                                "segments": segments_data,
                                "num_speakers": result.num_speakers,
                                "is_translation": is_translation,
                                "enable_diarization": True
                            }
                        })
                    )
            else:
                await websocket.send_text(
                    json.dumps({
                        "type": "recognition_result",
                        "data": {
                            "text": "",
                            "error": "No result generated"
                        }
                    })
                )
        except Exception as e:
            print(f"❌ Error sending file recognition result: {e}")

        print(f"✅ File recognition completed for session: {session_id}")

    except Exception as e:
        print(f"❌ Error processing file recognition: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": {"error": str(e)}})
            )
        except Exception as send_error:
            print(f"❌ Error sending file recognition error message: {send_error}")


async def cleanup_session(session_id: str):
    """Clean up recognition session."""
    try:
        session = recognition_sessions.get(session_id)
        if session:
            if session["recognizer"] and session.get("is_recording", False):
                session["recognizer"].stop_continuous_recognition()
            del recognition_sessions[session_id]
            print(f"🧹 Cleaned up session: {session_id}")
    except Exception as e:
        print(f"❌ Error cleaning up session: {e}")


if __name__ == "__main__":
    print("🚀 Starting Groq Speech API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🌐 WebSocket: ws://localhost:8000/ws/recognize")

    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
