"""
Voice Activity Detection Service using Silero VAD.

This module provides internal VAD functionality for the Groq Speech SDK.
Consumers of the SDK don't need to know about this implementation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import threading
from dataclasses import dataclass

# Lazy import to avoid loading VAD libraries unless needed
_silero_vad_model = None
_silero_vad_utils = None
_webrtc_vad = None
_vad_available = None
_vad_type = None

def _import_silero_vad():
    """Try to import Silero VAD."""
    try:
        from silero_vad import load_silero_vad
        model, utils = load_silero_vad()
        
        # Test the model to ensure it works
        import torch
        test_audio = torch.randn(16000)
        get_speech_timestamps = utils[0]
        timestamps = get_speech_timestamps(test_audio, model)
        
        return model, utils, "silero"
    except Exception as e:
        return None, None, None

def _import_webrtc_vad():
    """Try to import WebRTC VAD."""
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        return vad, "webrtc"
    except Exception as e:
        return None, None

def _initialize_vad():
    """Initialize the best available VAD."""
    global _silero_vad_model, _silero_vad_utils, _webrtc_vad, _vad_available, _vad_type
    
    if _vad_available is not None:
        return _vad_available
    
    # Try Silero VAD first
    model, utils, vtype = _import_silero_vad()
    if vtype == "silero":
        _silero_vad_model = model
        _silero_vad_utils = utils
        _vad_type = "silero"
        _vad_available = True
        print("✅ Silero VAD loaded and tested successfully")
        return True
    
    # Fallback to WebRTC VAD
    vad, vtype = _import_webrtc_vad()
    if vtype == "webrtc":
        _webrtc_vad = vad
        _vad_type = "webrtc"
        _vad_available = True
        print("✅ WebRTC VAD loaded successfully")
        return True
    
    # No VAD available
    print("⚠️  No VAD libraries available")
    print("💡 Using fallback RMS-based voice activity detection")
    _vad_available = False
    _vad_type = "fallback"
    return False

@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    threshold: float = 0.5  # Speech detection threshold (0.0-1.0)
    min_speech_duration_ms: int = 250  # Minimum speech duration in milliseconds
    min_silence_duration_ms: int = 10000  # Minimum silence duration in milliseconds (10 seconds)
    max_silence_duration_ms: int = 10000  # Maximum silence before forcing chunk (10 seconds)
    sample_rate: int = 16000  # Audio sample rate

@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration: float
    confidence: float

class VADService:
    """
    Voice Activity Detection Service using Silero VAD.
    
    This service provides intelligent speech detection and chunking
    for continuous audio streams.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize VAD service.
        
        Args:
            config: VAD configuration, uses defaults if None
        """
        self.config = config or VADConfig()
        self._model = None
        self._get_speech_timestamps = None
        self._is_initialized = False
        self._lock = threading.Lock()
        self._vad_type = None
        
    def _ensure_initialized(self) -> bool:
        """Ensure VAD model is loaded."""
        with self._lock:
            if self._is_initialized:
                return True
                
            if not _initialize_vad():
                self._vad_type = "fallback"
                return False
                
            try:
                # Access the global _vad_type
                global _vad_type
                if _vad_type == "silero":
                    self._model, utils = _silero_vad_model, _silero_vad_utils
                    self._get_speech_timestamps = utils[0]
                    self._vad_type = "silero"
                elif _vad_type == "webrtc":
                    self._model = _webrtc_vad
                    self._get_speech_timestamps = None  # WebRTC VAD doesn't use this
                    self._vad_type = "webrtc"
                else:
                    self._vad_type = "fallback"
                    return False
                    
                self._is_initialized = True
                return True
            except Exception as e:
                print(f"⚠️  Failed to initialize VAD: {e}")
                self._vad_type = "fallback"
                return False
    
    def is_available(self) -> bool:
        """Check if VAD is available."""
        return self._ensure_initialized()
    
    def detect_speech_segments(self, audio_data: np.ndarray, 
                             sample_rate: Optional[int] = None) -> List[SpeechSegment]:
        """
        Detect speech segments in audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate, uses config default if None
            
        Returns:
            List of detected speech segments
        """
        if not self._ensure_initialized():
            return []
            
        sample_rate = sample_rate or self.config.sample_rate
        
        try:
            if _vad_type == "silero":
                # Use Silero VAD
                speech_timestamps = self._get_speech_timestamps(
                    audio_data,
                    self._model,
                    threshold=self.config.threshold,
                    min_speech_duration_ms=self.config.min_speech_duration_ms,
                    min_silence_duration_ms=self.config.min_silence_duration_ms
                )
                
                # Convert to SpeechSegment objects
                segments = []
                for timestamp in speech_timestamps:
                    start_sample = timestamp['start']
                    end_sample = timestamp['end']
                    start_time = start_sample / sample_rate
                    end_time = end_sample / sample_rate
                    duration = end_time - start_time
                    confidence = timestamp.get('confidence', 0.5)
                    
                    segments.append(SpeechSegment(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        confidence=confidence
                    ))
                
                return segments
                
            elif _vad_type == "webrtc":
                # Use WebRTC VAD
                import webrtcvad
                
                # WebRTC VAD requires 16-bit PCM audio
                if audio_data.dtype != np.int16:
                    # Convert float32 to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data
                
                # WebRTC VAD works on 10ms, 20ms, or 30ms frames
                frame_duration_ms = 20
                frame_size = int(sample_rate * frame_duration_ms / 1000)
                
                segments = []
                current_segment_start = None
                
                for i in range(0, len(audio_int16) - frame_size, frame_size):
                    frame = audio_int16[i:i + frame_size]
                    
                    # WebRTC VAD requires exactly the right frame size
                    if len(frame) != frame_size:
                        continue
                    
                    try:
                        is_speech = self._model.is_speech(frame.tobytes(), sample_rate)
                        
                        if is_speech and current_segment_start is None:
                            # Start of speech segment
                            current_segment_start = i
                        elif not is_speech and current_segment_start is not None:
                            # End of speech segment
                            end_sample = i + frame_size
                            start_time = current_segment_start / sample_rate
                            end_time = end_sample / sample_rate
                            duration = end_time - start_time
                            
                            # Only include segments longer than minimum duration
                            if duration >= self.config.min_speech_duration_ms / 1000:
                                segments.append(SpeechSegment(
                                    start_sample=current_segment_start,
                                    end_sample=end_sample,
                                    start_time=start_time,
                                    end_time=end_time,
                                    duration=duration,
                                    confidence=0.8  # WebRTC VAD doesn't provide confidence
                                ))
                            
                            current_segment_start = None
                    except Exception:
                        # Skip frames that cause errors
                        continue
                
                # Handle case where speech continues to end of audio
                if current_segment_start is not None:
                    end_sample = len(audio_int16)
                    start_time = current_segment_start / sample_rate
                    end_time = end_sample / sample_rate
                    duration = end_time - start_time
                    
                    if duration >= self.config.min_speech_duration_ms / 1000:
                        segments.append(SpeechSegment(
                            start_sample=current_segment_start,
                            end_sample=end_sample,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            confidence=0.8
                        ))
                
                return segments
            
            else:
                return []
            
        except Exception as e:
            print(f"⚠️  VAD detection failed: {e}")
            return []
    
    def has_speech(self, audio_data: np.ndarray, 
                   sample_rate: Optional[int] = None) -> bool:
        """
        Check if audio contains speech.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate, uses config default if None
            
        Returns:
            True if speech is detected, False otherwise
        """
        segments = self.detect_speech_segments(audio_data, sample_rate)
        return len(segments) > 0
    
    def get_speech_ratio(self, audio_data: np.ndarray, 
                        sample_rate: Optional[int] = None) -> float:
        """
        Get the ratio of speech to total audio duration.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate, uses config default if None
            
        Returns:
            Ratio of speech duration to total duration (0.0-1.0)
        """
        segments = self.detect_speech_segments(audio_data, sample_rate)
        if not segments:
            return 0.0
            
        total_duration = len(audio_data) / (sample_rate or self.config.sample_rate)
        speech_duration = sum(segment.duration for segment in segments)
        
        return speech_duration / total_duration if total_duration > 0 else 0.0
    
    def should_create_chunk(self, audio_data: np.ndarray, 
                          sample_rate: Optional[int] = None,
                          max_duration_seconds: Optional[float] = None) -> Tuple[bool, str]:
        """
        Determine if a chunk should be created based on VAD analysis.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate, uses config default if None
            max_duration_seconds: Maximum duration before forcing chunk
            
        Returns:
            Tuple of (should_create, reason)
        """
        sample_rate = sample_rate or self.config.sample_rate
        duration_seconds = len(audio_data) / sample_rate
        
        # Calculate approximate size in MB (16kHz, 32-bit float = 4 bytes per sample)
        # 1 second = 16,000 samples = 64,000 bytes = 0.064 MB
        estimated_size_mb = (len(audio_data) * 4) / (1024 * 1024)
        
        # PRIORITY 1: Check if we're approaching 24MB limit (conservative threshold)
        if estimated_size_mb >= 20.0:  # 20MB threshold to stay under 24MB
            return True, f"Approaching 24MB limit ({estimated_size_mb:.1f}MB)"
        
        # PRIORITY 2: Check if we've hit maximum duration
        if max_duration_seconds and duration_seconds >= max_duration_seconds:
            return True, f"Maximum duration reached ({duration_seconds:.1f}s)"
        
        # If VAD is not available, use enhanced fallback
        if not self._ensure_initialized():
            # Enhanced fallback using multiple audio features
            audio_rms = np.sqrt(np.mean(audio_data**2))
            audio_max = np.max(np.abs(audio_data))
            audio_std = np.std(audio_data)
            
            # More sophisticated silence detection
            is_silence = (audio_rms < 0.001 and audio_max < 0.01 and audio_std < 0.005)
            
            if is_silence:
                if duration_seconds >= self.config.max_silence_duration_ms / 1000:
                    return True, f"Silence duration exceeded (fallback, {duration_seconds:.1f}s)"
                return False, "No speech detected (fallback), continuing..."
            
            # Check for very short audio bursts that might be noise
            if duration_seconds < 0.5 and audio_rms < 0.01:
                return False, f"Short audio burst detected (fallback), continuing... (RMS: {audio_rms:.4f})"
            
            return False, f"Audio detected (fallback), continuing... (RMS: {audio_rms:.4f}, Max: {audio_max:.4f})"
        
        # PRIORITY 3: Check for silence-based chunking using VAD (only if not approaching size limit)
        segments = self.detect_speech_segments(audio_data, sample_rate)
        
        if not segments:
            # No speech detected - check if we have enough silence
            if duration_seconds >= self.config.max_silence_duration_ms / 1000:
                return True, f"Silence duration exceeded ({duration_seconds:.1f}s)"
            return False, "No speech detected, continuing..."
        
        # Check if there's been silence at the end (only if we're not close to size limit)
        last_segment = segments[-1]
        silence_at_end = duration_seconds - last_segment.end_time
        
        # Only create chunk based on silence if we're not approaching the size limit
        if silence_at_end >= self.config.min_silence_duration_ms / 1000 and estimated_size_mb < 15.0:
            return True, f"Silence detected at end ({silence_at_end:.1f}s)"
        
        return False, f"Active speech detected, continuing... (last speech: {last_segment.end_time:.1f}s ago, size: {estimated_size_mb:.1f}MB)"
    
    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """
        Get audio level (RMS) for visual feedback.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Audio level (0.0-1.0)
        """
        if len(audio_data) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(audio_data**2))
        # Normalize to 0-1 range (adjust based on typical audio levels)
        return min(rms * 10, 1.0)  # Scale factor may need adjustment

# VAD service is now initialized per SpeechRecognizer instance
# No global VAD service needed - each SpeechRecognizer creates its own VADService
