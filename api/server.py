"""
FastAPI server for Groq Speech SDK.
Provides REST API and WebSocket endpoints for speech recognition.
"""

import os
import sys
import json
import asyncio
import threading
import time
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

        # Decode base64 audio data and convert to numpy array
        import base64
        import numpy as np

        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(request.audio_data)

            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to float32
            audio_array_float = audio_array.astype(np.float32) / 32768.0

            # Perform recognition on the audio data
            result = recognizer.recognize_audio_data(audio_array_float)

        except Exception as audio_error:
            print(f"Error processing audio data: {audio_error}")
            return RecognitionResponse(
                success=False, error=f"Failed to process audio data: {str(audio_error)}"
            )

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

        # Decode base64 audio data and convert to numpy array
        import base64
        import numpy as np

        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(request.audio_data)

            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to float32
            audio_array_float = audio_array.astype(np.float32) / 32768.0

            # Perform translation on the audio data
            result = recognizer.recognize_audio_data(
                audio_array_float, is_translation=True
            )

        except Exception as audio_error:
            print(f"Error processing audio data: {audio_error}")
            return RecognitionResponse(
                success=False, error=f"Failed to process audio data: {str(audio_error)}"
            )

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


@app.post("/api/v1/recognize-file")
async def recognize_audio_file(file_path: str, language: str = "en-US"):
    """Recognize speech from audio file."""
    try:
        # Validate API key
        Config.get_api_key()

        # Create configurations
        speech_config = get_speech_config(language=language)
        # audio_config = AudioConfig(filename=file_path) # This line was causing a runtime error

        # Create recognizer
        recognizer = SpeechRecognizer(
            speech_config=speech_config,  # audio_config=audio_config # This line was causing a runtime error
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


@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple WebSocket test endpoint to verify connectivity."""
    print("🧪 WebSocket test connection request...")
    await websocket.accept()
    print("✅ WebSocket test connection accepted")

    try:
        # Send immediate test message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "test",
                    "data": {"message": "WebSocket connection successful!"},
                }
            )
        )
        print("✅ Test message sent")

        # Wait for client message
        message = await websocket.receive_text()
        print(f"📨 Test message received: {message}")

        # Send response
        await websocket.send_text(
            json.dumps(
                {"type": "test_response", "data": {"message": "Echo: " + message}}
            )
        )
        print("✅ Test response sent")

    except WebSocketDisconnect:
        print("🔌 WebSocket test disconnected")
    except Exception as e:
        print(f"💥 WebSocket test error: {e}")
    finally:
        print("🧹 WebSocket test cleanup completed")


@app.websocket("/ws/recognize")
async def websocket_recognition(websocket: WebSocket):
    """WebSocket endpoint for real-time speech recognition."""
    print("🔌 New WebSocket connection request...")
    await websocket.accept()
    print("✅ WebSocket connection accepted")

    active_connections.append(websocket)
    print(f"📊 Active connections: {len(active_connections)}")

    session_id = str(id(websocket))
    recognition_sessions[session_id] = {
        "websocket": websocket,
        "start_time": datetime.now(),
        "transcripts": [],
        "is_recording": False,
    }
    print(f"🆔 Created session {session_id}")

    try:
        # Send connection confirmation
        print(f"📤 Sending connection confirmation to session {session_id}")
        await websocket.send_text(
            json.dumps({"type": "connected", "data": {"session_id": session_id}})
        )
        print(f"✅ Connection confirmation sent to session {session_id}")

        while True:
            # Receive message from client
            print(f"👂 Waiting for message from session {session_id}...")
            message = await websocket.receive_text()
            print(
                f"📨 Received WebSocket message from session {session_id}: "
                f"{message}"
            )
            data = json.loads(message)

            if data["type"] == "start_recognition":
                print(f"🚀 Starting recognition for session {session_id}")
                await handle_start_recognition(session_id, data.get("data", {}))
            elif data["type"] == "stop_recognition":
                print(f"🛑 Stopping recognition for session {session_id}")
                await handle_stop_recognition(session_id)
            elif data["type"] == "audio_data":
                print(f"� Processing audio data for session {session_id}")
                # Process audio data asynchronously without blocking
                asyncio.create_task(handle_audio_data(session_id, data.get("data", {})))
            elif data["type"] == "ping":
                print(f"🏓 Ping received for session {session_id}")
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                print(
                    f"❓ Unknown message type from session {session_id}: "
                    f"{data['type']}"
                )
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "data": {"message": "Unknown message type"}}
                    )
                )

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"💥 WebSocket error for session {session_id}: {e}")
        await websocket.send_text(
            json.dumps({"type": "error", "data": {"message": str(e)}})
        )
    finally:
        # Cleanup
        print(f"🧹 Cleaning up session {session_id}")
        if websocket in active_connections:
            active_connections.remove(websocket)
            print(
                f"📊 Removed from active connections. Total: {len(active_connections)}"
            )
        if session_id in recognition_sessions:
            del recognition_sessions[session_id]
            print(f"🗑️ Deleted session {session_id}")
        print(f"✅ Cleanup completed for session {session_id}")


async def handle_websocket_close(websocket: WebSocket):
    """Handle WebSocket close and clean up the session."""
    try:
        # Find and remove the session
        session_id_to_remove = None
        for session_id, session in recognition_sessions.items():
            if session.get("websocket") == websocket:
                session_id_to_remove = session_id
                print(f"🔌 WebSocket closed for session {session_id}, cleaning up...")
                break

        if session_id_to_remove:
            # Remove the session
            del recognition_sessions[session_id_to_remove]
            print(f"🧹 Cleaned up session {session_id_to_remove}")

    except Exception as e:
        print(f"Error handling WebSocket close: {e}")


async def handle_start_recognition(session_id: str, config: Dict[str, Any]):
    """Handle start recognition request."""
    try:
        session = recognition_sessions[session_id]
        session["is_recording"] = True

        # Store configuration in session for audio processing
        session["current_model"] = config.get("model")
        session["is_translation"] = config.get("is_translation", False)
        session["target_language"] = config.get("target_language")
        session["mode"] = config.get("mode", "continuous")  # "continuous" or "single"
        session["buffered_results"] = []  # For single shot mode

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
                # recognizer = SpeechRecognizer(speech_config=speech_config)  # Unused variable removed

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
    print(f"🛑 Handling stop recognition for session {session_id}")

    if session_id in recognition_sessions:
        session = recognition_sessions[session_id]

        # Mark session as stopped
        session["is_recording"] = False
        print(f"✅ Marked session {session_id} as stopped")

        # Handle single shot mode - send all buffered results
        mode = session.get("mode", "continuous")
        if mode == "single" and session.get("buffered_results"):
            print(
                f"📤 Sending {len(session['buffered_results'])} buffered results for single shot mode"
            )
            try:
                # Send all buffered results
                for i, result in enumerate(session["buffered_results"]):
                    await session["websocket"].send_text(
                        json.dumps({"type": "recognition_result", "data": result})
                    )
                    print(f"✅ Sent buffered result {i+1}: '{result['text']}'")

                # Clear buffered results
                session["buffered_results"] = []

            except Exception as e:
                print(f"❌ Error sending buffered results: {e}")

        # Send confirmation to frontend
        try:
            await session["websocket"].send_text(
                json.dumps(
                    {"type": "recognition_stopped", "data": {"status": "stopped"}}
                )
            )
            print(f"✅ Stop confirmation sent to frontend for session {session_id}")
        except Exception as e:
            print(f"❌ Error sending stop confirmation: {e}")

        # Clean up the session
        try:
            if session_id in recognition_sessions:
                del recognition_sessions[session_id]
                print(f"🧹 Cleaned up session {session_id}")
        except Exception as e:
            print(f"❌ Error cleaning up session: {e}")
    else:
        print(f"⚠️ Session {session_id} not found for stop recognition")


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

    # Check if session has been stopped
    if not session.get("is_recording", False):
        print(f"⚠️ Session {session_id} has been stopped, ignoring audio data")
        return

    # Decode base64 audio data
    audio_bytes = base64.b64decode(audio_data["audio_data"])
    print(f"Decoded audio data: {len(audio_bytes)} bytes")

    # Get the MIME type from the frontend
    mime_type = audio_data.get("mime_type", "audio/webm")
    print(f"Audio MIME type: {mime_type}")

    # Accumulate audio chunks for better recognition
    if "audio_chunks" not in session:
        session["audio_chunks"] = []

    session["audio_chunks"].append(audio_bytes)
    total_audio_size = sum(len(chunk) for chunk in session["audio_chunks"])

    print(
        f"Accumulated {len(session['audio_chunks'])} chunks, total size: {total_audio_size} bytes"
    )

    # Only process if we have enough audio data or if it's the last chunk
    min_audio_size = 1000  # Reduced to 1KB minimum for better recognition
    is_last_chunk = audio_data.get("is_last_chunk", False)

    if total_audio_size < min_audio_size and not is_last_chunk:
        print(
            f"⏳ Waiting for more audio data (current: {total_audio_size}, need: {min_audio_size})"
        )
        return

    # Validate that we have some audio data
    if total_audio_size == 0:
        print("⚠️ No audio data to process")
        return

    # Process accumulated audio chunks
    try:
        # Combine all accumulated chunks
        combined_audio = b"".join(session["audio_chunks"])
        print(f"Processing combined audio: {len(combined_audio)} bytes")

        # Create temporary file for the combined audio
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

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as f:
                f.write(combined_audio)
                audio_file_path = f.name
                print(f"Created temporary audio file: {audio_file_path}")

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

            # Create recognizer - EXACTLY like CLI
            print("🎤 Starting recognition...")
            start_time = time.time()

            # Create recognizer - EXACTLY like CLI (no AudioConfig needed)
            recognizer = SpeechRecognizer(config)

            print("Created SpeechRecognizer using groq_speech SDK - EXACTLY like CLI")

            # For WebSocket mode, we need to process the audio data directly
            # Convert the temporary file to numpy array and process it
            import numpy as np
            import soundfile as sf

            try:
                # Try to read the audio file directly
                try:
                    audio_data, sample_rate = sf.read(audio_file_path)
                    print(
                        f"Read audio file: {len(audio_data)} samples, {sample_rate} Hz"
                    )
                except Exception as read_error:
                    print(f"Could not read audio file directly: {read_error}")
                    print("Converting audio format using pydub...")

                    # Fallback: convert audio to WAV using pydub
                    from pydub import AudioSegment

                    audio = AudioSegment.from_file(audio_file_path)
                    # Convert to WAV format that soundfile can read
                    wav_path = audio_file_path + ".wav"
                    audio.export(wav_path, format="wav")

                    # Read the converted WAV file
                    audio_data, sample_rate = sf.read(wav_path)
                    print(
                        f"Read converted WAV file: {len(audio_data)} samples, {sample_rate} Hz"
                    )

                    # Clean up converted file
                    os.unlink(wav_path)

                # Perform recognition on the audio data
                result = recognizer.recognize_audio_data(audio_data, is_translation)
                print(f"Recognition result: {result.reason}")

                # Clean up temporary file
                os.unlink(audio_file_path)
                print(f"Cleaned up temporary file: {audio_file_path}")

            except Exception as audio_error:
                print(f"Error processing audio data: {audio_error}")
                # Clean up temporary file on error
                if audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                raise audio_error

            processing_time = time.time() - start_time

            # Process result - EXACTLY like CLI
            if result.reason == ResultReason.RecognizedSpeech:
                print(f"✅ Recognition successful!")
                print(f"📝 Text: {result.text}")
                print(f"🎯 Confidence: {result.confidence:.2f}")
                print(f"🌍 Language: {result.language}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")

                # IMPORTANT: Log the transcribed text clearly for debugging
                print(
                    f"🎤 TRANSCRIBED TEXT: '{result.text}' (Total audio size: {len(combined_audio)} bytes)"
                )

                # Handle result based on mode
                mode = session.get("mode", "continuous")

                if mode == "single":
                    # For single shot mode, buffer the result until stop_recognition
                    print(f"📥 Buffering result for single shot mode: '{result.text}'")
                    session["buffered_results"].append(
                        {
                            "text": result.text,
                            "confidence": result.confidence or 0.95,
                            "language": result.language or "auto-detected",
                            "timestamps": result.timestamps or [],
                            "timing_metrics": {
                                "api_call": 0,
                                "response_processing": processing_time,
                                "total_time": processing_time,
                            },
                        }
                    )
                    print(
                        f"✅ Result buffered. Total buffered: {len(session['buffered_results'])}"
                    )
                else:
                    # For continuous mode, send result immediately
                    print(f"📤 Sending result to frontend: '{result.text}'")
                    try:
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
                        print(f"✅ Result sent successfully to frontend")

                    except Exception as send_error:
                        print(f"❌ Failed to send result to frontend: {send_error}")
                        # Try to close the WebSocket if it's broken
                        try:
                            await session["websocket"].close()
                            print(
                                f"🔌 Closed broken WebSocket for session {session_id}"
                            )
                        except Exception as close_error:
                            print(f"Error closing WebSocket: {close_error}")

                # Clear accumulated chunks after successful processing
                session["audio_chunks"] = []

            elif result.reason == ResultReason.NoMatch:
                print("❌ No speech detected in the audio file")
                print(
                    f"🎤 NO SPEECH DETECTED (Total audio size: {len(combined_audio)} bytes)"
                )
                # Clear accumulated chunks even for no match
                session["audio_chunks"] = []

            elif result.reason == ResultReason.Canceled:
                print(
                    f"❌ Recognition canceled: {result.cancellation_details.error_details}"
                )
                print(
                    f"🎤 RECOGNITION CANCELED (Total audio size: {len(combined_audio)} bytes)"
                )
                await session["websocket"].send_text(
                    json.dumps(
                        {
                            "type": "recognition_error",
                            "data": {"error": "Recognition canceled"},
                        }
                    )
                )
                # Clear accumulated chunks on error
                session["audio_chunks"] = []

        except Exception as e:
            print(f"❌ Error during recognition: {e}")
            # Only send error for actual errors, not for no speech detected
            if "no speech" not in str(e).lower():
                await send_recognition_error(session_id, str(e))
            # Clear accumulated chunks on error
            session["audio_chunks"] = []

        finally:
            # Clean up the temporary file if it was created
            if audio_file_path and os.path.exists(audio_file_path):
                try:
                    os.unlink(audio_file_path)
                    print("Cleaned up temporary file")
                except Exception as cleanup_error:
                    print(f"Error cleaning up temporary file: {cleanup_error}")

    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        await send_recognition_error(session_id, str(e))


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
    print("🚀 Starting Groq Speech API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🌐 WebSocket: ws://localhost:8000/ws/recognize")

    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
