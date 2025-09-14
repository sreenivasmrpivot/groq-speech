"""
FastAPI server for Groq Speech SDK.

Provides REST API and WebSocket endpoints for speech recognition and translation.
The API server acts as a bridge between the web interface and the core SDK,
using the same SpeechRecognizer class as the CLI interface.

Key Features:
- REST API endpoints for file processing
- WebSocket endpoints for real-time processing
- Support for both transcription and translation
- Speaker diarization support
- Health check and monitoring endpoints

Architecture:
- Layer 2b: API Client (parallel to CLI Client)
- Uses same SpeechRecognizer as CLI
- Provides HTTP/WebSocket interface to Layer 3 (UI)
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

# Load environment variables from groq_speech/.env
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "groq_speech", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
    print(f"🔍 GROQ_API_KEY: {'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
    print(f"🔍 HF_TOKEN: {'SET' if os.getenv('HF_TOKEN') else 'NOT SET'}")
else:
    print(f"⚠️  Environment file not found: {env_path}")

# Import groq_speech after adding to path
from groq_speech import (
    SpeechConfig,
    SpeechRecognizer,
    ResultReason,
)
from groq_speech.logging_utils import api_logger


# Pydantic models for API requests/responses
class LogRequest(BaseModel):
    component: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

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
    api_logger.info("🚀 Starting Groq Speech API Server...")
    try:
        SpeechConfig.get_api_key()
        api_logger.success("API key validated")
    except ValueError as e:
        api_logger.error(f"API key error: {e}")
    yield
    api_logger.info("🛑 Shutting down Groq Speech API Server...")


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


@app.post("/api/v1/log")
async def log_frontend_message(request: LogRequest):
    """Receive logs from frontend and display them in terminal."""
    try:
        # Format the log message for terminal display
        timestamp = request.timestamp
        component = request.component
        level = request.level
        message = request.message
        data = request.data or {}
        
        # Create a formatted log message
        data_str = ""
        if data and len(data) > 0:
            data_str = " | " + " ".join([f"{k}={v}" for k, v in data.items()])
        
        formatted_message = f"[{timestamp}] [{component}] [{level}] {message}{data_str}"
        
        # Print to terminal with color coding
        if level == "ERROR":
            print(f"\033[0;31m[FRONTEND] {formatted_message}\033[0m")  # Red
        elif level == "WARNING":
            print(f"\033[1;33m[FRONTEND] {formatted_message}\033[0m")  # Yellow
        elif level == "INFO":
            print(f"\033[0;34m[FRONTEND] {formatted_message}\033[0m")  # Blue
        else:
            print(f"[FRONTEND] {formatted_message}")
        
        return {"status": "logged"}
    except Exception as e:
        print(f"[FRONTEND] Error processing log: {e}")
        return {"status": "error", "message": str(e)}

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

@app.post("/api/log")
async def frontend_log(request: dict):
    """Receive frontend logs and forward to terminal."""
    try:
        component = request.get("component", "UNKNOWN")
        level = request.get("level", "INFO")
        message = request.get("message", "")
        context = request.get("context", {})
        prefix = request.get("prefix", "")
        timestamp = request.get("timestamp", datetime.now().isoformat())
        
        # Format the log message for terminal output
        time_str = timestamp[11:23] if len(timestamp) > 23 else timestamp
        context_str = ""
        if context:
            context_str = " | " + " ".join([f"{k}={json.dumps(v)}" for k, v in context.items()])
        
        prefix_str = f"[{prefix}] " if prefix else ""
        formatted_message = f"[{time_str}] [FRONTEND-{component}] [{level}] {prefix_str}{message}{context_str}"
        
        # Print to terminal with color coding
        if level == "ERROR":
            print(f"\033[0;31m{formatted_message}\033[0m")  # Red
        elif level == "WARN":
            print(f"\033[1;33m{formatted_message}\033[0m")  # Yellow
        elif level == "DEBUG":
            print(f"\033[0;36m{formatted_message}\033[0m")  # Cyan
        else:
            print(f"\033[0;34m{formatted_message}\033[0m")  # Blue
            
        return {"status": "logged"}
    except Exception as e:
        print(f"Error processing frontend log: {e}")
        return {"status": "error", "message": str(e)}



@app.post("/api/v1/recognize", response_model=RecognitionResponse)
async def recognize_speech(request: RecognitionRequest):
    """Recognize speech from audio data - EXACTLY like CLI file recognition."""
    try:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        api_logger.info("🎤 REST API: Received audio data request", {
            "audio_length": len(request.audio_data),
            "enable_timestamps": request.enable_timestamps,
            "target_language": request.target_language,
            "enable_language_detection": request.enable_language_detection,
            "enable_diarization": request.enable_diarization,
            "model": request.model,
            "timestamp": datetime.now().isoformat()
        })
        
        api_logger.dataFlow("Frontend", "API", {
            "request_type": "recognition",
            "audio_data_length": len(request.audio_data),
            "enable_diarization": request.enable_diarization,
            "target_language": request.target_language
        }, "Audio data received from frontend")

        # Setup speech configuration - EXACTLY like CLI
        speech_config = get_speech_config(
            model=request.model,
            is_translation=False,
            target_language=request.target_language,
        )

        # Create recognizer - EXACTLY like CLI
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=request.target_language
        )

        # Decode audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf
        import io

        audio_bytes = base64.b64decode(request.audio_data)
        
        # Try to decode as WAV file first (proper audio format)
        try:
            audio_array_float, sample_rate = sf.read(io.BytesIO(audio_bytes))
            print(f"📊 Decoded as WAV: {len(audio_array_float)} samples, {sample_rate}Hz")
            print(f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}")
        except Exception as e:
            print(f"⚠️  Failed to decode as WAV, falling back to raw PCM: {e}")
            # Fallback to raw PCM int16 (old behavior)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array_float = audio_array.astype(np.float32) / 32768.0
            sample_rate = 16000
            print(f"📊 Decoded as raw PCM: {len(audio_array)} samples, {sample_rate}Hz")
            print(f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}")

        # Process based on diarization requirement
        if request.enable_diarization:
            # For diarization, save to temp file and use process_file
            print("🎭 Processing with diarization...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array_float, sample_rate)
            
            try:
                api_logger.dataFlow("API", "Groq API", {
                    "method": "process_file",
                    "file_path": temp_path,
                    "enable_diarization": True,
                    "is_translation": False,
                    "audio_samples": len(audio_array_float),
                    "sample_rate": 16000
                }, "Sending audio file to Groq API for diarization")
                
                result = await recognizer.process_file(temp_path, enable_diarization=True, is_translation=False)
                
                api_logger.dataFlow("Groq API", "API", {
                    "result_type": "diarization",
                    "has_segments": hasattr(result, 'segments') and result.segments is not None,
                    "segments_count": len(result.segments) if hasattr(result, 'segments') and result.segments else 0,
                    "has_text": hasattr(result, 'text') and result.text is not None,
                    "text_length": len(result.text) if hasattr(result, 'text') and result.text else 0
                }, "Received diarization result from Groq API")
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
            
            api_logger.dataFlow("API", "Groq API", {
                "method": "recognize_audio_data",
                "audio_samples": len(audio_array_float),
                "sample_rate": sample_rate,
                "enable_diarization": False,
                "is_translation": False
            }, "Sending audio data to Groq API for recognition")
            
            result = recognizer.recognize_audio_data(audio_array_float, sample_rate, is_translation=False)
            
            api_logger.dataFlow("Groq API", "API", {
                "result_type": "recognition",
                "has_text": hasattr(result, 'text') and result.text is not None,
                "text_length": len(result.text) if hasattr(result, 'text') and result.text else 0,
                "confidence": getattr(result, 'confidence', None),
                "reason": getattr(result, 'reason', None)
            }, "Received recognition result from Groq API")

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

            api_logger.dataFlow("API", "Frontend", {
                "response_type": "recognition",
                "text_length": len(result.text) if result.text else 0,
                "confidence": result.confidence,
                "language": result.language,
                "has_timestamps": hasattr(result, 'timestamps') and result.timestamps is not None
            }, "Sending recognition response to frontend")

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
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=request.target_language
        )

        # Decode audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf
        import io

        audio_bytes = base64.b64decode(request.audio_data)
        
        # Try to decode as WAV file first (proper audio format)
        try:
            audio_array_float, sample_rate = sf.read(io.BytesIO(audio_bytes))
            print(f"📊 Decoded as WAV: {len(audio_array_float)} samples, {sample_rate}Hz")
            print(f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}")
        except Exception as e:
            print(f"⚠️  Failed to decode as WAV, falling back to raw PCM: {e}")
            # Fallback to raw PCM int16 (old behavior)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array_float = audio_array.astype(np.float32) / 32768.0
            sample_rate = 16000
            print(f"📊 Decoded as raw PCM: {len(audio_array)} samples, {sample_rate}Hz")
            print(f"🎵 Audio range: {audio_array_float.min():.4f} to {audio_array_float.max():.4f}")

        # Process based on diarization requirement
        if request.enable_diarization:
            # For diarization, save to temp file and use process_file
            print("🎭 Processing translation with diarization...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array_float, sample_rate)
            
            try:
                result = await recognizer.process_file(temp_path, enable_diarization=True, is_translation=True)
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
            result = recognizer.recognize_audio_data(audio_array_float, sample_rate, is_translation=True)

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


@app.post("/api/v1/recognize-microphone", response_model=RecognitionResponse)
async def recognize_microphone_single(request: dict):
    """Single microphone recognition - EXACTLY like speech_demo.py process_microphone_single."""
    try:
        # Extract audio data and parameters - EXACTLY like speech_demo.py
        audio_data = request.get("audio_data")  # Raw float32 array as list
        sample_rate = request.get("sample_rate", 16000)
        enable_diarization = request.get("enable_diarization", False)
        is_translation = request.get("is_translation", False)
        target_language = request.get("target_language", "en")
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        api_logger.info("🎤 SINGLE MIC: Processing microphone audio", {
            "audio_samples": len(audio_data),
            "sample_rate": sample_rate,
            "enable_diarization": enable_diarization,
            "is_translation": is_translation,
            "duration": len(audio_data) / sample_rate,
            "timestamp": datetime.now().isoformat()
        })

        # Convert list to numpy array - EXACTLY like speech_demo.py
        import numpy as np
        audio_array = np.array(audio_data, dtype=np.float32)
        
        print(f"📊 Microphone audio: {len(audio_array)} samples, {sample_rate}Hz")
        print(f"📊 Duration: {len(audio_array) / sample_rate:.2f} seconds")

        # Setup speech configuration - EXACTLY like speech_demo.py
        speech_config = get_speech_config(
            model="whisper-large-v3-turbo",
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer - EXACTLY like speech_demo.py
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=target_language
        )

        # Process with groq_speech - EXACTLY like speech_demo.py process_microphone_single
        if enable_diarization:
            # For diarization, save to temp file and use process_file - EXACTLY like speech_demo.py
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array, sample_rate)
            
            try:
                result = await recognizer.process_file(temp_path, enable_diarization=True, is_translation=is_translation)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use chunked audio data processing - EXACTLY like speech_demo.py
            result = recognizer.recognize_audio_data_chunked(audio_array, sample_rate, is_translation=is_translation)

        # Return result - EXACTLY like speech_demo.py
        if hasattr(result, "text") and result.text:
            print(f"✅ Single microphone recognition successful: '{result.text}'")
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=getattr(result, 'confidence', 0.0),
                language=getattr(result, 'language', 'Unknown')
            )
        elif hasattr(result, "segments") and result.segments:
            print(f"✅ Single microphone diarization successful: {result.num_speakers} speakers, {len(result.segments)} segments")
            
            # Convert segments to the format expected by frontend
            segments_data = []
            full_text_parts = []
            
            for segment in result.segments:
                segment_text = getattr(segment, 'text', '') or getattr(segment, 'transcription', '') or '[No text]'
                if segment_text and segment_text.strip() and segment_text != '[No text]':
                    full_text_parts.append(segment_text)
                
                segments_data.append({
                    "speaker_id": segment.speaker_id,
                    "text": segment_text,
                    "start_time": getattr(segment, 'start_time', 0.0),
                    "end_time": getattr(segment, 'end_time', 0.0)
                })
            
            # Combine all segment texts into main text
            combined_text = " ".join(full_text_parts) if full_text_parts else "[No speech detected]"
            
            return RecognitionResponse(
                success=True,
                text=combined_text,
                confidence=getattr(result, 'confidence', 0.95),
                language=getattr(result, 'language', 'Unknown'),
                segments=segments_data,
                num_speakers=result.num_speakers
            )
        else:
            return RecognitionResponse(success=False, error="No speech detected")

    except Exception as e:
        api_logger.error("Error processing single microphone recognition", {
            "error": str(e)
        })
        import traceback
        traceback.print_exc()
        return RecognitionResponse(success=False, error=str(e))


@app.post("/api/v1/recognize-microphone-continuous", response_model=RecognitionResponse)
async def recognize_microphone_continuous(request: dict):
    """Continuous microphone recognition - EXACTLY like speech_demo.py process_microphone_continuous."""
    try:
        # Extract audio data and parameters - EXACTLY like speech_demo.py
        audio_data = request.get("audio_data")  # Raw float32 array as list
        sample_rate = request.get("sample_rate", 16000)
        enable_diarization = request.get("enable_diarization", False)
        is_translation = request.get("is_translation", False)
        target_language = request.get("target_language", "en")
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")

        api_logger.info("🎤 CONTINUOUS MIC: Processing microphone audio chunk", {
            "audio_samples": len(audio_data),
            "sample_rate": sample_rate,
            "enable_diarization": enable_diarization,
            "is_translation": is_translation,
            "duration": len(audio_data) / sample_rate,
            "timestamp": datetime.now().isoformat()
        })

        # Convert list to numpy array - EXACTLY like speech_demo.py
        import numpy as np
        audio_array = np.array(audio_data, dtype=np.float32)
        
        print(f"📊 Continuous microphone chunk: {len(audio_array)} samples, {sample_rate}Hz")
        print(f"📊 Duration: {len(audio_array) / sample_rate:.2f} seconds")

        # Setup speech configuration - EXACTLY like speech_demo.py
        speech_config = get_speech_config(
            model="whisper-large-v3-turbo",
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer - EXACTLY like speech_demo.py
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=target_language
        )

        # Process with groq_speech - EXACTLY like speech_demo.py process_microphone_continuous
        if enable_diarization:
            # For diarization, save to temp file and use process_file - EXACTLY like speech_demo.py
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array, sample_rate)
            
            try:
                result = await recognizer.process_file(temp_path, enable_diarization=True, is_translation=is_translation)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use direct audio data processing - EXACTLY like speech_demo.py
            result = recognizer.recognize_audio_data(audio_array, sample_rate, is_translation=is_translation)

        # Return result - EXACTLY like speech_demo.py
        if hasattr(result, "text") and result.text:
            print(f"✅ Continuous microphone recognition successful: '{result.text}'")
            return RecognitionResponse(
                success=True,
                text=result.text,
                confidence=getattr(result, 'confidence', 0.0),
                language=getattr(result, 'language', 'Unknown')
            )
        elif hasattr(result, "segments") and result.segments:
            print(f"✅ Continuous microphone diarization successful: {result.num_speakers} speakers, {len(result.segments)} segments")
            
            # Convert segments to the format expected by frontend
            segments_data = []
            full_text_parts = []
            
            for segment in result.segments:
                segment_text = getattr(segment, 'text', '') or getattr(segment, 'transcription', '') or '[No text]'
                if segment_text and segment_text.strip() and segment_text != '[No text]':
                    full_text_parts.append(segment_text)
                
                segments_data.append({
                    "speaker_id": segment.speaker_id,
                    "text": segment_text,
                    "start_time": getattr(segment, 'start_time', 0.0),
                    "end_time": getattr(segment, 'end_time', 0.0)
                })
            
            # Combine all segment texts into main text
            combined_text = " ".join(full_text_parts) if full_text_parts else "[No speech detected]"
            
            return RecognitionResponse(
                success=True,
                text=combined_text,
                confidence=getattr(result, 'confidence', 0.95),
                language=getattr(result, 'language', 'Unknown'),
                segments=segments_data,
                num_speakers=result.num_speakers
            )
        else:
            return RecognitionResponse(success=False, error="No speech detected")

    except Exception as e:
        api_logger.error("Error processing continuous microphone recognition", {
            "error": str(e)
        })
        import traceback
        traceback.print_exc()
        return RecognitionResponse(success=False, error=str(e))


@app.websocket("/ws/recognize")
async def websocket_recognition(websocket: WebSocket):
    """WebSocket endpoint for continuous speech recognition - EXACTLY like CLI."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    websocket_closed = False

    try:
        api_logger.info("🔌 WebSocket connected", {
            "session_id": session_id,
            "client_ip": websocket.client.host if websocket.client else "unknown",
            "timestamp": datetime.now().isoformat()
        })

        # Initialize session
        recognition_sessions[session_id] = {
            "websocket": websocket,
            "recognizer": None,
            "is_recording": False,
        }

        # Wait for start message
        while not websocket_closed:
            try:
                # Check if WebSocket is still open
                if websocket.client_state.value != 1:  # WebSocket.OPEN
                    print(f"🔌 WebSocket is not open, state: {websocket.client_state.value}")
                    print(f"🔍 WebSocket client_state details: {websocket.client_state}")
                    print(f"🔍 WebSocket application_state: {websocket.application_state.value}")
                    websocket_closed = True
                    break
                    
                # Use asyncio.wait_for to add a timeout to prevent hanging
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)  # Increased timeout for long audio processing
                except asyncio.TimeoutError:
                    print(f"⏰ WebSocket receive timeout for session {session_id}")
                    # Send a ping to check if connection is still alive
                    success = await safe_send_message(websocket, {"type": "ping"})
                    if not success:
                        print(f"❌ WebSocket ping failed, connection may be closed")
                        websocket_closed = True
                        break
                    continue
                except RuntimeError as e:
                    if "WebSocket is not connected" in str(e):
                        print(f"🔌 WebSocket connection closed: {e}")
                        websocket_closed = True
                        break
                    else:
                        print(f"❌ WebSocket runtime error: {e}")
                        websocket_closed = True
                        break
                except Exception as e:
                    print(f"❌ WebSocket receive error: {e}")
                    websocket_closed = True
                    break
                    
                data = json.loads(message)
                message_type = data.get("type")

                api_logger.debug("WebSocket message received", {
                    "session_id": session_id,
                    "message_type": message_type,
                    "message_size": len(message),
                    "timestamp": datetime.now().isoformat()
                })

                if message_type == "start_recognition":
                    api_logger.info("Handling start_recognition", {
                        "session_id": session_id,
                        "data": data.get("data", {})
                    })
                    await handle_start_recognition(session_id, data)
                elif message_type == "stop_recognition":
                    api_logger.info("Handling stop_recognition", {"session_id": session_id})
                    await handle_stop_recognition(session_id)
                elif message_type == "single_recognition":
                    api_logger.info("Handling single_recognition", {
                        "session_id": session_id,
                        "data": data.get("data", {})
                    })
                    await handle_single_recognition(session_id, data)
                elif message_type == "file_recognition":
                    api_logger.info("Handling file_recognition", {
                        "session_id": session_id,
                        "data": data.get("data", {})
                    })
                    # handle_file_recognition removed - use raw audio endpoint instead
                elif message_type == "audio_data":
                    # Audio data via WebSocket is deprecated - use raw audio endpoint instead
                    await safe_send_message(websocket, {
                        "type": "error", 
                        "data": {"error": "Audio data via WebSocket is deprecated. Please use the raw audio endpoint instead."}
                    })
                elif message_type == "audio_chunk":
                    # Audio chunking is no longer supported - use raw audio endpoint instead
                    await safe_send_message(websocket, {
                        "type": "error", 
                        "data": {"error": "Audio chunking is deprecated. Please use the raw audio endpoint instead."}
                    })
                elif message_type == "ping":
                    api_logger.debug("Handling ping", {"session_id": session_id})
                    success = await safe_send_message(websocket, {"type": "pong"})
                    if not success:
                        print(f"❌ Failed to send pong, WebSocket may be closed")
                        websocket_closed = True
                        break
                else:
                    api_logger.warning("Unknown message type received", {
                        "session_id": session_id,
                        "message_type": message_type,
                        "data": data
                    })

            except json.JSONDecodeError as e:
                api_logger.error("Invalid JSON message", {
                    "session_id": session_id,
                    "error": str(e),
                    "message_preview": message[:200] + "..." if len(message) > 200 else message
                })
            except WebSocketDisconnect:
                api_logger.info("WebSocket disconnected", {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                websocket_closed = True
                break
            except Exception as e:
                api_logger.error("Error processing WebSocket message", {
                    "session_id": session_id,
                    "error": str(e),
                    "message_type": data.get("type") if 'data' in locals() else "unknown"
                })
                print(f"❌ WebSocket message processing error: {e}")
                import traceback
                traceback.print_exc()
                
                # Check if this is a WebSocket connection error
                if "WebSocket is not connected" in str(e) or "Cannot call" in str(e):
                    print(f"🔌 WebSocket connection lost, breaking out of loop")
                    websocket_closed = True
                    break
                
                # Don't try to send error message if WebSocket is closed
                if not websocket_closed:
                    try:
                        await safe_send_message(websocket, {"type": "error", "data": {"error": str(e)}})
                    except Exception as send_error:
                        print(f"❌ Failed to send error message: {send_error}")
                        websocket_closed = True
                        break
                else:
                    # If WebSocket is already closed, break out of the loop
                    break

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
        websocket_closed = True
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        websocket_closed = True
    finally:
        # Cleanup session
        print(f"🧹 Cleaning up WebSocket session: {session_id}")
        if session_id in recognition_sessions:
            await cleanup_session(session_id)


async def safe_send_message(websocket: WebSocket, message: dict) -> bool:
    """Safely send a message to WebSocket, return True if successful."""
    try:
        if websocket.client_state.value == 1:  # WebSocket.OPEN
            await websocket.send_text(json.dumps(message))
            return True
    except Exception as e:
        print(f"❌ Error sending message: {e}")
    return False


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
                        success = await safe_send_message(websocket, {
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
                        })
                        if not success:
                            print("❌ WebSocket closed, stopping result processor")
                            break
                    elif data.reason == ResultReason.NoMatch:
                        print("❌ No speech detected")
                        success = await safe_send_message(websocket, {
                            "type": "no_speech",
                            "data": {
                                "message": "No speech detected",
                            }
                        })
                        if not success:
                            print("❌ WebSocket closed, stopping result processor")
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
                    success = await safe_send_message(websocket, {
                        "type": "recognition_canceled", 
                        "data": {"error": error_msg}
                    })
                    if not success:
                        print("❌ WebSocket closed, stopping result processor")
                        break

                elif event_type == "session_started":
                    print(f"🎬 Recognition session started for: {session_id}")
                    success = await safe_send_message(websocket, {
                        "type": "session_started",
                        "data": {
                            "message": "Recognition session started",
                        }
                    })
                    if not success:
                        print("❌ WebSocket closed, stopping result processor")
                        break

                elif event_type == "session_stopped":
                    print(f"🏁 Recognition session stopped for: {session_id}")
                    success = await safe_send_message(websocket, {
                        "type": "session_stopped",
                        "data": {
                            "message": "Recognition session stopped",
                        }
                    })
                    if not success:
                        print("❌ WebSocket closed, stopping result processor")
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
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=target_language
        )

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
        await safe_send_message(websocket, {
            "type": "recognition_started",
            "data": {
                "message": "Continuous recognition started",
            }
        })

        print(f"✅ Continuous recognition started for session: {session_id}")

    except Exception as e:
        print(f"❌ Error starting recognition: {e}")
        await safe_send_message(websocket, {
            "type": "error", 
            "data": {"error": f"Failed to start recognition: {str(e)}"}
        })


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

            # Send success message
            await safe_send_message(websocket, {
                "type": "recognition_stopped",
                "data": {
                    "message": "Continuous recognition stopped",
                }
            })

            print(f"✅ Continuous recognition stopped for session: {session_id}")
        else:
            await safe_send_message(websocket, {
                "type": "error", 
                "data": {"error": "No active recognition session"}
            })

    except Exception as e:
        print(f"❌ Error stopping recognition: {e}")
        await safe_send_message(websocket, {
            "type": "error",
            "data": {"error": f"Failed to stop recognition: {str(e)}"},
        })


async def handle_single_recognition(session_id: str, data: dict):
    """Handle single microphone recognition - record then process."""
    try:
        session = recognition_sessions[session_id]
        websocket = session["websocket"]

        print(f"🎤 Starting single recognition for session: {session_id}")
        
        api_logger.dataFlow("Frontend", "API", {
            "session_id": session_id,
            "message_type": "single_recognition",
            "data": data.get("data", {})
        }, "Single recognition request received from frontend")

        # Get recognition parameters
        is_translation = data.get("is_translation", False)
        enable_diarization = data.get("enable_diarization", False)
        target_language = data.get("target_language", "en")
        
        api_logger.info("Single recognition parameters", {
            "session_id": session_id,
            "is_translation": is_translation,
            "enable_diarization": enable_diarization,
            "target_language": target_language
        })

        # Setup speech configuration
        speech_config = get_speech_config(
            model=None,
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=target_language
        )

        # Configure translation if needed
        if is_translation:
            speech_config.enable_translation = True
            print(f"🔀 Translation mode enabled (target: {target_language})")

        session["recognizer"] = recognizer

        # Send acknowledgment
        await safe_send_message(websocket, {
            "type": "single_recognition_started",
            "data": {
                "message": "Single recognition started - speak now",
                "enable_diarization": enable_diarization,
                "is_translation": is_translation
            }
        })

        print(f"✅ Single recognition started for session: {session_id}")

    except Exception as e:
        print(f"❌ Error starting single recognition: {e}")
        await safe_send_message(websocket, {
            "type": "error", 
            "data": {"error": str(e)}
        })


# Removed handle_file_recognition - using raw audio endpoint instead
    """Handle file-based recognition via WebSocket."""
    try:
        session = recognition_sessions[session_id]
        websocket = session["websocket"]

        api_logger.info("Processing file recognition", {
            "session_id": session_id,
            "data_structure": "nested" if "data" in data else "flat",
            "raw_data_keys": list(data.keys())
        })

        # Get recognition parameters - handle nested structure from UI
        audio_data = data.get("audio_data")
        is_translation = data.get("is_translation", False)
        enable_diarization = data.get("enable_diarization", False)
        target_language = data.get("target_language", "en")
        
        # If audio_data is not in top level, check in nested data structure
        if not audio_data and "data" in data:
            nested_data = data["data"]
            audio_data = nested_data.get("audio_data")
            is_translation = nested_data.get("is_translation", is_translation)
            enable_diarization = nested_data.get("enable_diarization", enable_diarization)
            target_language = nested_data.get("target_language", target_language)

        api_logger.info("Audio data parameters extracted", {
            "session_id": session_id,
            "audio_data_length": len(audio_data) if audio_data else 0,
            "audio_data_type": type(audio_data).__name__,
            "is_translation": is_translation,
            "enable_diarization": enable_diarization,
            "target_language": target_language
        })

        # Log detailed audio data analysis
        if audio_data:
            api_logger.info("🔍 Frontend Audio Data Analysis", {
                "session_id": session_id,
                "action": "frontend_audio_analysis",
                "audio_data_length": len(audio_data),
                "audio_data_length_mb": f"{len(audio_data) / (1024 * 1024):.2f} MB",
                "audio_data_preview": audio_data[:100] + "..." if len(audio_data) > 100 else audio_data,
                "expected_duration_60min": "60 minutes",
                "expected_base64_length_60min": f"{(16000 * 60 * 60 * 2 * 4 / 3) / (1024 * 1024):.2f} MB",  # 16kHz * 60s * 60min * 2 bytes * 4/3 base64
                "timestamp": datetime.now().isoformat()
            })

        if not audio_data:
            api_logger.error("No audio data provided in WebSocket message", {
                "session_id": session_id,
                "data_keys": list(data.keys()),
                "nested_data_keys": list(data.get("data", {}).keys()) if "data" in data else []
            })
            await safe_send_message(websocket, {
                "type": "error", 
                "data": {"error": "No audio data provided"}
            })
            return

        # Setup speech configuration
        speech_config = get_speech_config(
            model=None,
            is_translation=is_translation,
            target_language=target_language,
        )

        # Create recognizer
        recognizer = SpeechRecognizer(
            speech_config=speech_config,
            translation_target_language=target_language
        )

        # Configure translation if needed
        if is_translation:
            speech_config.enable_translation = True
            api_logger.info("Translation mode enabled", {
                "session_id": session_id,
                "target_language": target_language
            })

        # Decode and process audio data
        import base64
        import numpy as np
        import tempfile
        import soundfile as sf
        import io
        from pydub import AudioSegment

        api_logger.info("Starting audio decoding", {
            "session_id": session_id,
            "base64_length": len(audio_data),
            "is_translation": is_translation,
            "enable_diarization": enable_diarization
        })
        
        # Log base64 sample for debugging
        api_logger.debug("Base64 sample (first 100 chars)", {
            "session_id": session_id,
            "sample": audio_data[:100] if len(audio_data) > 100 else audio_data
        })

        try:
            # Log base64 data characteristics before decoding
            api_logger.debug("Base64 data characteristics", {
                "session_id": session_id,
                "base64_length": len(audio_data),
                "base64_sample_start": audio_data[:50] if len(audio_data) > 50 else audio_data,
                "base64_sample_end": audio_data[-50:] if len(audio_data) > 50 else audio_data,
                "base64_valid_chars": all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in audio_data[:100])
            })
            
            # Log base64 data characteristics before decoding
            api_logger.info("🔍 Base64 Data Analysis", {
                "session_id": session_id,
                "action": "base64_analysis",
                "base64_length": len(audio_data),
                "base64_length_mb": f"{len(audio_data) / (1024 * 1024):.2f} MB",
                "base64_start": audio_data[:50] if len(audio_data) > 50 else audio_data,
                "base64_end": audio_data[-50:] if len(audio_data) > 50 else audio_data,
                "base64_valid_chars": all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in audio_data[:1000]),
                "expected_decoded_bytes": int(len(audio_data) * 3 / 4),  # Base64 to bytes conversion
                "timestamp": datetime.now().isoformat()
            })
            
            audio_bytes = base64.b64decode(audio_data)
            
            # Log decoded data characteristics
            api_logger.info("🔍 Decoded Bytes Analysis", {
                "session_id": session_id,
                "action": "decoded_bytes_analysis",
                "decoded_bytes_length": len(audio_bytes),
                "decoded_bytes_length_mb": f"{len(audio_bytes) / (1024 * 1024):.2f} MB",
                "expected_length": int(len(audio_data) * 3 / 4),
                "length_match": "✅ MATCH" if len(audio_bytes) == int(len(audio_data) * 3 / 4) else "❌ MISMATCH",
                "bytes_preview": audio_bytes[:20].hex() if len(audio_bytes) > 20 else audio_bytes.hex(),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            api_logger.error("Base64 decoding failed", {
                "session_id": session_id,
                "base64_length": len(audio_data),
                "error": str(e),
                "base64_sample": audio_data[:100] if len(audio_data) > 100 else audio_data
            })
            raise ValueError(f"Invalid base64 audio data: {e}")
        
        # Validate decoded data
        compression_ratio = len(audio_bytes) / len(audio_data) if len(audio_data) > 0 else 0
        expected_ratio = 0.75  # Base64 should compress to ~75% of original size
        is_valid_ratio = 0.6 <= compression_ratio <= 0.8  # Allow some variance
        
        api_logger.info("Audio data decoded from base64", {
            "session_id": session_id,
            "base64_length": len(audio_data),
            "base64_length_mb": f"{len(audio_data) / (1024 * 1024):.2f} MB",
            "decoded_bytes": len(audio_bytes),
            "decoded_bytes_mb": f"{len(audio_bytes) / (1024 * 1024):.2f} MB",
            "compression_ratio": compression_ratio,
            "expected_ratio": expected_ratio,
            "ratio_valid": "✅ VALID" if is_valid_ratio else "❌ INVALID"
        })

        # Log detailed data analysis
        api_logger.info("📊 Backend Data Analysis", {
            "session_id": session_id,
            "action": "backend_data_analysis",
            "base64_length": len(audio_data),
            "base64_length_mb": f"{len(audio_data) / (1024 * 1024):.2f} MB",
            "decoded_bytes": len(audio_bytes),
            "decoded_bytes_mb": f"{len(audio_bytes) / (1024 * 1024):.2f} MB",
            "compression_ratio": compression_ratio,
            "expected_ratio": expected_ratio,
            "ratio_valid": "✅ VALID" if is_valid_ratio else "❌ INVALID",
            "timestamp": datetime.now().isoformat()
        })
        
        if not is_valid_ratio:
            api_logger.warning("Suspicious base64 compression ratio", {
                "session_id": session_id,
                "compression_ratio": compression_ratio,
                "expected_ratio": expected_ratio,
                "base64_length": len(audio_data),
                "decoded_bytes": len(audio_bytes)
            })
        
        # Try to decode as raw PCM data first (like REST API)
        try:
            # Log detailed information about the decoded bytes
            api_logger.info("🔍 Detailed Audio Bytes Analysis", {
                "session_id": session_id,
                "action": "detailed_bytes_analysis",
                "audio_bytes_length": len(audio_bytes),
                "audio_bytes_length_mb": f"{len(audio_bytes) / (1024 * 1024):.2f} MB",
                "expected_samples_for_60min": 16000 * 60 * 60,  # 16kHz * 60s * 60min
                "expected_bytes_for_60min": 16000 * 60 * 60 * 2,  # 16kHz * 60s * 60min * 2 bytes per sample
                "bytes_per_sample": 2,  # int16 = 2 bytes
                "expected_samples": len(audio_bytes) // 2,
                "timestamp": datetime.now().isoformat()
            })
            
            # Decode as raw PCM int16 data (same as REST API)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array_float = audio_array.astype(np.float32) / 32768.0
            sample_rate = 16000  # Fixed sample rate for microphone recording
            
            # Log the actual decoded array details
            api_logger.info("🔍 Decoded Audio Array Analysis", {
                "session_id": session_id,
                "action": "decoded_array_analysis",
                "audio_array_length": len(audio_array),
                "audio_array_float_length": len(audio_array_float),
                "expected_length": len(audio_bytes) // 2,
                "length_match": "✅ MATCH" if len(audio_array) == len(audio_bytes) // 2 else "❌ MISMATCH",
                "duration_seconds": len(audio_array) / sample_rate,
                "duration_minutes": (len(audio_array) / sample_rate) / 60,
                "timestamp": datetime.now().isoformat()
            })
            
            api_logger.success("Audio decoded as raw PCM", {
                "session_id": session_id,
                "samples": len(audio_array),
                "sample_rate": sample_rate,
                "duration": len(audio_array) / sample_rate,
                "format": "raw_pcm_int16",
                "audio_range": f"{audio_array_float.min():.4f} to {audio_array_float.max():.4f}",
                "non_zero_samples": np.count_nonzero(audio_array),
                "silence_ratio": (len(audio_array) - np.count_nonzero(audio_array)) / len(audio_array) if len(audio_array) > 0 else 0
            })
        except Exception as e:
            api_logger.warning("Failed to decode as raw PCM, trying WebM", {
                "session_id": session_id,
                "error": str(e)
            })
            try:
                # Fallback to WebM/Opus format
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_array_float = audio_array / (2**15)  # 16-bit audio normalization
                sample_rate = audio_segment.frame_rate
                api_logger.success("Audio decoded via pydub", {
                    "session_id": session_id,
                    "samples": len(audio_array),
                    "sample_rate": sample_rate,
                    "duration": len(audio_array) / sample_rate,
                    "format": "webm"
                })
            except Exception as e2:
                api_logger.warning("Failed to decode as WebM, trying soundfile", {
                    "session_id": session_id,
                    "error": str(e2)
                })
                try:
                    # Fallback to soundfile for other formats
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    audio_array_float = audio_array.astype(np.float32)
                    api_logger.success("Audio decoded via soundfile", {
                        "session_id": session_id,
                        "samples": len(audio_array_float),
                        "sample_rate": sample_rate,
                        "duration": len(audio_array_float) / sample_rate,
                        "format": "soundfile"
                    })
                except Exception as e3:
                    api_logger.warning("Failed to decode as audio file, trying raw PCM", {
                        "session_id": session_id,
                        "error": str(e3)
                    })
                    # Last resort: raw PCM interpretation (Int16Array as Uint8Array)
                    print(f"🔍 Raw PCM data size: {len(audio_bytes)} bytes")
                    
                    # The frontend sends Int16Array as little-endian Uint8Array
                    audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    sample_rate = 16000  # Default sample rate
                    audio_array_float = audio_array_int16.astype(np.float32) / 32768.0
                    
                    print(f"📊 Decoded as raw PCM: {len(audio_array_int16)} samples, {sample_rate}Hz")
                    print(f"📊 Expected samples for {len(audio_bytes)} bytes: {len(audio_bytes) // 2}")
                    print(f"📊 Duration: {len(audio_array_float) / sample_rate:.2f} seconds")
                    
                    # Validate the data size
                    expected_samples = len(audio_bytes) // 2  # 2 bytes per Int16 sample
                    if len(audio_array_int16) != expected_samples:
                        print(f"⚠️ Sample count mismatch: got {len(audio_array_int16)}, expected {expected_samples}")
                        # Try to fix by padding or truncating
                        if len(audio_array_int16) < expected_samples:
                            # Pad with zeros
                            padding = np.zeros(expected_samples - len(audio_array_int16), dtype=np.int16)
                            audio_array_int16 = np.concatenate([audio_array_int16, padding])
                            audio_array_float = audio_array_int16.astype(np.float32) / 32768.0
                            print(f"📊 Padded to {len(audio_array_int16)} samples")
                        else:
                            # Truncate
                            audio_array_int16 = audio_array_int16[:expected_samples]
                            audio_array_float = audio_array_int16.astype(np.float32) / 32768.0
                            print(f"📊 Truncated to {len(audio_array_int16)} samples")
                    
                    api_logger.info("Audio decoded as raw PCM", {
                        "session_id": session_id,
                        "samples": len(audio_array_float),
                        "sample_rate": sample_rate,
                        "duration": len(audio_array_float) / sample_rate,
                        "format": "raw_pcm_int16"
                    })

        # Process audio
        if enable_diarization:
            # For diarization, save to temp file and use process_file
            print(f"🔍 DEBUG: Processing with diarization for session {session_id}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array_float, sample_rate)
            
            try:
                print(f"🔍 DEBUG: About to call recognizer.process_file for session {session_id}")
                result = await recognizer.process_file(temp_path, enable_diarization=True, is_translation=is_translation)
                print(f"🔍 DEBUG: Completed recognizer.process_file for session {session_id}")
            finally:
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            # For non-diarization, use direct audio data processing
            print(f"🔍 DEBUG: Processing without diarization for session {session_id}")
            api_logger.info("Starting audio recognition", {
                "session_id": session_id,
                "audio_samples": len(audio_array_float),
                "sample_rate": sample_rate,
                "duration": len(audio_array_float) / sample_rate,
                "duration_minutes": f"{(len(audio_array_float) / sample_rate) / 60:.2f} min",
                "is_translation": is_translation,
                "audio_range": f"{audio_array_float.min():.4f} to {audio_array_float.max():.4f}",
                "non_zero_samples": np.count_nonzero(audio_array_float),
                "silence_ratio": f"{(len(audio_array_float) - np.count_nonzero(audio_array_float)) / len(audio_array_float):.2%}",
                "expected_samples": int(sample_rate * 60),  # Expected samples for 1 minute
                "duration_match": "✅ MATCH" if len(audio_array_float) == int(sample_rate * 60) else "❌ MISMATCH"
            })

            # Log data being sent to Groq SDK
            api_logger.info("📤 Backend sending data to Groq SDK", {
                "session_id": session_id,
                "action": "backend_to_sdk_transmission",
                "audio_samples": len(audio_array_float),
                "sample_rate": sample_rate,
                "duration": len(audio_array_float) / sample_rate,
                "duration_minutes": f"{(len(audio_array_float) / sample_rate) / 60:.2f} min",
                "is_translation": is_translation,
                "audio_range": f"{audio_array_float.min():.4f} to {audio_array_float.max():.4f}",
                "non_zero_samples": np.count_nonzero(audio_array_float),
                "silence_ratio": f"{(len(audio_array_float) - np.count_nonzero(audio_array_float)) / len(audio_array_float):.2%}",
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"🔍 DEBUG: About to call recognizer.recognize_audio_data for session {session_id}")
            result = recognizer.recognize_audio_data(audio_array_float, sample_rate, is_translation=is_translation)
            print(f"🔍 DEBUG: Completed recognizer.recognize_audio_data for session {session_id}")

            # Log result returned from Groq SDK
            api_logger.info("📥 Backend received result from Groq SDK", {
                "session_id": session_id,
                "action": "sdk_to_backend_result",
                "result_type": type(result).__name__,
                "has_text": hasattr(result, 'text'),
                "text_length": len(result.text) if hasattr(result, 'text') and result.text else 0,
                "text_preview": result.text[:100] + "..." if hasattr(result, 'text') and result.text and len(result.text) > 100 else (result.text if hasattr(result, 'text') and result.text else "No text"),
                "confidence": getattr(result, 'confidence', 'N/A'),
                "language": getattr(result, 'language', 'N/A'),
                "has_segments": hasattr(result, 'segments'),
                "segments_count": len(result.segments) if hasattr(result, 'segments') else 0,
                "timestamp": datetime.now().isoformat()
            })
            
            api_logger.info("Audio recognition completed", {
                "session_id": session_id,
                "result_text": result.text if result and hasattr(result, "text") else "None",
                "result_confidence": getattr(result, "confidence", None) if result else None,
                "result_language": getattr(result, "language", None) if result else None
            })

        # Send result
        if result:
            if hasattr(result, "text") and result.text:
                success = await safe_send_message(websocket, {
                    "type": "recognition_result",
                    "data": {
                        "text": result.text,
                        "confidence": getattr(result, "confidence", None),
                        "language": getattr(result, "language", None),
                        "is_translation": is_translation,
                        "enable_diarization": False
                    }
                })
                if not success:
                    print(f"❌ Failed to send recognition result, WebSocket may be closed")
                    return
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
                
                success = await safe_send_message(websocket, {
                    "type": "diarization_result",
                    "data": {
                        "segments": segments_data,
                        "num_speakers": result.num_speakers,
                        "is_translation": is_translation,
                        "enable_diarization": True
                    }
                })
                if not success:
                    print(f"❌ Failed to send diarization result, WebSocket may be closed")
                    return
        else:
            success = await safe_send_message(websocket, {
                "type": "recognition_result",
                "data": {
                    "text": "",
                    "error": "No result generated"
                }
            })
            if not success:
                print(f"❌ Failed to send error result, WebSocket may be closed")
                return

        print(f"✅ File recognition completed for session: {session_id}")
        
        # Add a small delay to ensure the message is sent before the function returns
        await asyncio.sleep(0.1)
        
        # Check if WebSocket is still open after sending result
        if websocket.client_state.value != 1:  # WebSocket.OPEN
            print(f"🔌 WebSocket closed after sending result for session {session_id}")
            return
            
        # Send a keep-alive ping to maintain connection
        success = await safe_send_message(websocket, {"type": "ping"})
        if not success:
            print(f"❌ Failed to send keep-alive ping, WebSocket may be closed")
            return
            
        # Wait a bit more to ensure the ping is processed
        await asyncio.sleep(0.1)
        
        # Final check if WebSocket is still open
        if websocket.client_state.value != 1:  # WebSocket.OPEN
            print(f"🔌 WebSocket closed after keep-alive ping for session {session_id}")
            return
            
        print(f"✅ WebSocket connection maintained for session {session_id}")
        
        # Log the WebSocket state for debugging
        print(f"🔍 WebSocket state after processing: {websocket.client_state.value}")
        
        # Check if the WebSocket is still in the active connections list
        if session_id in recognition_sessions:
            print(f"✅ Session {session_id} still active in recognition_sessions")
        else:
            print(f"❌ Session {session_id} not found in recognition_sessions")
            
        # Log the total number of active sessions
        print(f"🔍 Total active sessions: {len(recognition_sessions)}")
        
        # Log the session details for debugging
        if session_id in recognition_sessions:
            session = recognition_sessions[session_id]
            print(f"🔍 Session details: {list(session.keys())}")
            if "websocket" in session:
                print(f"🔍 WebSocket in session: {session['websocket'].client_state.value}")
                
        # Final status check
        print(f"🔍 Final WebSocket state check: {websocket.client_state.value}")
        if websocket.client_state.value == 1:  # WebSocket.OPEN
            print(f"✅ WebSocket is still open and ready for more messages")
        else:
            print(f"❌ WebSocket is closed or in error state")
            
        # Log the WebSocket connection details
        if websocket.client:
            print(f"🔍 WebSocket client: {websocket.client.host}:{websocket.client.port}")
        else:
            print(f"❌ WebSocket client is None")
            
        # Log the WebSocket application state
        print(f"🔍 WebSocket application state: {websocket.application_state.value}")
        
        # Log the WebSocket close code and reason if available
        if hasattr(websocket, 'close_code'):
            print(f"🔍 WebSocket close code: {websocket.close_code}")
        if hasattr(websocket, 'close_reason'):
            print(f"🔍 WebSocket close reason: {websocket.close_reason}")
            
        # Log the WebSocket headers if available
        if hasattr(websocket, 'headers'):
            print(f"🔍 WebSocket headers: {dict(websocket.headers)}")
            
        # Log the WebSocket URL if available
        if hasattr(websocket, 'url'):
            print(f"🔍 WebSocket URL: {websocket.url}")
            
        # Log the WebSocket scope if available
        if hasattr(websocket, 'scope'):
            print(f"🔍 WebSocket scope keys: {list(websocket.scope.keys()) if websocket.scope else 'None'}")

    except Exception as e:
        print(f"❌ Error processing file recognition: {e}")
        import traceback
        traceback.print_exc()
        try:
            await safe_send_message(websocket, {
                "type": "error", 
                "data": {"error": str(e)}
            })
        except Exception as send_error:
            print(f"❌ Failed to send error message to WebSocket: {send_error}")
            # Don't re-raise the error, just log it




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
