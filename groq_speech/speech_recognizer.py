"""
Speech recognizer for Groq Speech services.
"""

import asyncio
import json
import time
import threading
import base64
import io
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import soundfile as sf
from .speech_config import SpeechConfig
from .audio_config import AudioConfig
from .result_reason import ResultReason, CancellationReason
from .config import Config


class SpeechRecognitionResult:
    """
    Result of a speech recognition operation.
    """
    
    def __init__(self, text: str = "", reason: ResultReason = ResultReason.NoMatch, 
                 confidence: float = 0.0, language: str = "", 
                 cancellation_details: Optional['CancellationDetails'] = None,
                 no_match_details: Optional['NoMatchDetails'] = None):
        """
        Initialize speech recognition result.
        
        Args:
            text: Recognized text
            reason: Result reason
            confidence: Confidence score (0.0 to 1.0)
            language: Detected language
            cancellation_details: Details if recognition was canceled
            no_match_details: Details if no speech was recognized
        """
        self.text = text
        self.reason = reason
        self.confidence = confidence
        self.language = language
        self.cancellation_details = cancellation_details
        self.no_match_details = no_match_details
    
    def __str__(self):
        return f"SpeechRecognitionResult(text='{self.text}', reason={self.reason}, confidence={self.confidence})"


class CancellationDetails:
    """
    Details about why speech recognition was canceled.
    """
    
    def __init__(self, reason: CancellationReason, error_details: str = ""):
        """
        Initialize cancellation details.
        
        Args:
            reason: Reason for cancellation
            error_details: Detailed error message
        """
        self.reason = reason
        self.error_details = error_details


class NoMatchDetails:
    """
    Details about why no speech was recognized.
    """
    
    def __init__(self, reason: str = "NoMatch", error_details: str = ""):
        """
        Initialize no match details.
        
        Args:
            reason: Reason for no match
            error_details: Detailed error message
        """
        self.reason = reason
        self.error_details = error_details


class SpeechRecognizer:
    """
    Main class for speech recognition using Groq services.
    Supports single-shot and continuous recognition.
    """
    
    def __init__(self, speech_config: SpeechConfig, audio_config: Optional[AudioConfig] = None):
        """
        Initialize speech recognizer.
        
        Args:
            speech_config: Speech configuration
            audio_config: Audio configuration (optional for microphone input)
        """
        self.speech_config = speech_config
        self.audio_config = audio_config
        
        # Note: Groq client initialization is commented out due to compatibility issues
        # In a production environment, you would initialize the Groq client here
        # self.groq_client = groq.Groq(api_key=speech_config.api_key)
        
        # Event handlers
        self.recognizing_handlers: List[Callable] = []
        self.recognized_handlers: List[Callable] = []
        self.session_started_handlers: List[Callable] = []
        self.session_stopped_handlers: List[Callable] = []
        self.canceled_handlers: List[Callable] = []
        
        # Continuous recognition state
        self._is_recognizing = False
        self._recognition_thread = None
        self._stop_recognition = False
        
        # Validate configuration
        self.speech_config.validate()
    
    def connect(self, event_type: str, handler: Callable):
        """
        Connect an event handler.
        
        Args:
            event_type: Type of event ('recognizing', 'recognized', 'session_started', 
                                     'session_stopped', 'canceled')
            handler: Event handler function
        """
        if event_type == 'recognizing':
            self.recognizing_handlers.append(handler)
        elif event_type == 'recognized':
            self.recognized_handlers.append(handler)
        elif event_type == 'session_started':
            self.session_started_handlers.append(handler)
        elif event_type == 'session_stopped':
            self.session_stopped_handlers.append(handler)
        elif event_type == 'canceled':
            self.canceled_handlers.append(handler)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, event_data: Any):
        """Trigger event handlers for a specific event type."""
        handlers = []
        if event_type == 'recognizing':
            handlers = self.recognizing_handlers
        elif event_type == 'recognized':
            handlers = self.recognized_handlers
        elif event_type == 'session_started':
            handlers = self.session_started_handlers
        elif event_type == 'session_stopped':
            handlers = self.session_stopped_handlers
        elif event_type == 'canceled':
            handlers = self.canceled_handlers
        
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                print(f"Error in event handler: {e}")
    
    def recognize_once_async(self) -> 'SpeechRecognitionResult':
        """
        Perform single-shot speech recognition.
        
        Returns:
            SpeechRecognitionResult object
        """
        try:
            # Trigger session started event
            self._trigger_event('session_started', {'session_id': f'session_{int(time.time())}'})
            
            # Get audio data
            if self.audio_config and self.audio_config.filename:
                # File-based recognition
                audio_data = self.audio_config.get_file_audio_data()
                result = self._recognize_audio_data(audio_data)
            else:
                # Microphone-based recognition
                result = self._recognize_from_microphone()
            
            # Trigger session stopped event
            self._trigger_event('session_stopped', {'session_id': f'session_{int(time.time())}'})
            
            return result
            
        except Exception as e:
            cancellation_details = CancellationDetails(
                CancellationReason.Error,
                f"Recognition failed: {str(e)}"
            )
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=cancellation_details
            )
    
    def _recognize_audio_data(self, audio_data: np.ndarray) -> SpeechRecognitionResult:
        """
        Recognize speech from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            SpeechRecognitionResult
        """
        try:
            # Save audio to temporary buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.audio_config.sample_rate if self.audio_config else 16000, format='WAV')
            buffer.seek(0)
            
            # For demonstration purposes, we'll simulate recognition
            # In a real implementation, you would call the appropriate Groq API endpoint
            # This simulates successful recognition for demo purposes
            
            # Simulate processing time
            time.sleep(0.5)
            
            # Simulate different results based on language
            if self.speech_config.speech_recognition_language == "de-DE":
                simulated_text = "Hallo, das ist ein simuliertes Spracherkennungsergebnis."
            else:
                simulated_text = "Hello, this is a simulated speech recognition result."
            
            return SpeechRecognitionResult(
                text=simulated_text,
                reason=ResultReason.RecognizedSpeech,
                confidence=0.95,
                language=self.speech_config.speech_recognition_language
            )
                
        except Exception as e:
            cancellation_details = CancellationDetails(
                CancellationReason.Error,
                f"Recognition failed: {str(e)}"
            )
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=cancellation_details
            )
    
    def _recognize_from_microphone(self) -> SpeechRecognitionResult:
        """
        Recognize speech from microphone input.
        
        Returns:
            SpeechRecognitionResult
        """
        if not self.audio_config:
            self.audio_config = AudioConfig()
        
        try:
            print("Speak into your microphone...")
            print("(Press Ctrl+C to stop)")
            
            # Collect audio for a few seconds
            audio_chunks = []
            start_time = time.time()
            max_duration = 5  # Maximum 5 seconds for demo
            
            # Use the audio config to read audio
            with self.audio_config as audio:
                print("Recording audio...")
                while time.time() - start_time < max_duration:
                    try:
                        chunk = audio.read_audio_chunk(1024)
                        if chunk and len(chunk) > 0:
                            audio_chunks.append(chunk)
                            print(".", end="", flush=True)  # Show progress
                    except KeyboardInterrupt:
                        print("\nStopped by user")
                        break
                    except Exception as e:
                        print(f"\nError reading audio: {e}")
                        break
                
                print()  # New line after progress dots
                
                if not audio_chunks:
                    print("No audio captured. Please try speaking louder or check your microphone.")
                    return SpeechRecognitionResult(
                        reason=ResultReason.NoMatch,
                        no_match_details=NoMatchDetails("No audio captured")
                    )
                
                print(f"Captured {len(audio_chunks)} audio chunks")
                
                # Combine audio chunks
                audio_data = b''.join(audio_chunks)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize
                
                return self._recognize_audio_data(audio_array)
                
        except Exception as e:
            cancellation_details = CancellationDetails(
                CancellationReason.Error,
                f"Microphone recognition failed: {str(e)}"
            )
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=cancellation_details
            )
    
    def start_continuous_recognition(self):
        """Start continuous speech recognition."""
        if self._is_recognizing:
            return
        
        self._is_recognizing = True
        self._stop_recognition = False
        self._recognition_thread = threading.Thread(target=self._continuous_recognition_worker)
        self._recognition_thread.start()
    
    def stop_continuous_recognition(self):
        """Stop continuous speech recognition."""
        self._stop_recognition = True
        self._is_recognizing = False
        if self._recognition_thread:
            self._recognition_thread.join()
    
    def _continuous_recognition_worker(self):
        """Worker thread for continuous recognition."""
        while not self._stop_recognition:
            try:
                result = self.recognize_once_async()
                if result.reason == ResultReason.RecognizedSpeech:
                    self._trigger_event('recognized', result)
                elif result.reason == ResultReason.Canceled:
                    self._trigger_event('canceled', result)
                    break
            except Exception as e:
                print(f"Error in continuous recognition: {e}")
                break
    
    def recognize_once(self) -> SpeechRecognitionResult:
        """
        Perform single-shot speech recognition (synchronous).
        
        Returns:
            SpeechRecognitionResult object
        """
        return self.recognize_once_async()
    
    def is_recognizing(self) -> bool:
        """Check if recognition is currently active."""
        return self._is_recognizing 