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
import groq
from .speech_config import SpeechConfig
from .audio_config import AudioConfig
from .audio_processor import OptimizedAudioProcessor, AudioChunker
from .result_reason import ResultReason, CancellationReason
from .config import Config
from .property_id import PropertyId


class TimingMetrics:
    """Timing metrics for transcription pipeline."""

    def __init__(self):
        self.microphone_start = None
        self.microphone_end = None
        self.api_call_start = None
        self.api_call_end = None
        self.processing_start = None
        self.processing_end = None
        self.total_start = None
        self.total_end = None

    def start_microphone(self):
        """Start microphone timing."""
        self.microphone_start = time.time()
        if not self.total_start:
            self.total_start = self.microphone_start

    def end_microphone(self):
        """End microphone timing."""
        self.microphone_end = time.time()

    def start_api_call(self):
        """Start API call timing."""
        self.api_call_start = time.time()

    def end_api_call(self):
        """End API call timing."""
        self.api_call_end = time.time()

    def start_processing(self):
        """Start response processing timing."""
        self.processing_start = time.time()

    def end_processing(self):
        """End response processing timing."""
        self.processing_end = time.time()
        self.total_end = time.time()

    def get_metrics(self) -> Dict[str, float]:
        """Get all timing metrics."""
        metrics = {}

        if self.microphone_start and self.microphone_end:
            metrics["microphone_capture"] = self.microphone_end - self.microphone_start

        if self.api_call_start and self.api_call_end:
            metrics["api_call"] = self.api_call_end - self.api_call_start

        if self.processing_start and self.processing_end:
            metrics["response_processing"] = self.processing_end - self.processing_start

        if self.total_start and self.total_end:
            metrics["total_time"] = self.total_end - self.total_start

        return metrics


class SpeechRecognitionResult:
    """
    Result of a speech recognition operation.
    """

    def __init__(
        self,
        text: str = "",
        reason: ResultReason = ResultReason.NoMatch,
        confidence: float = 0.0,
        language: str = "",
        cancellation_details: Optional["CancellationDetails"] = None,
        no_match_details: Optional["NoMatchDetails"] = None,
        timestamps: Optional[List[Dict[str, Any]]] = None,
        timing_metrics: Optional[TimingMetrics] = None,
    ):
        """
        Initialize speech recognition result.

        Args:
            text: Recognized text
            reason: Result reason
            confidence: Confidence score (0.0 to 1.0)
            language: Detected language
            cancellation_details: Details if recognition was canceled
            no_match_details: Details if no speech was recognized
            timestamps: Word-level or segment-level timestamps
            timing_metrics: Timing metrics for the transcription pipeline
        """
        self.text = text
        self.reason = reason
        self.confidence = confidence
        self.language = language
        self.cancellation_details = cancellation_details
        self.no_match_details = no_match_details
        self.timestamps = timestamps or []
        self.timing_metrics = timing_metrics

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

    def __init__(
        self, speech_config: SpeechConfig, audio_config: Optional[AudioConfig] = None
    ):
        """
        Initialize speech recognizer.

        Args:
            speech_config: Speech configuration
            audio_config: Audio configuration (optional for microphone input)
        """
        self.speech_config = speech_config
        self.audio_config = audio_config

        # Initialize Groq client with proper error handling
        try:
            # For newer versions of groq library, only pass api_key
            self.groq_client = groq.Groq(api_key=speech_config.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Groq client: {e}")

        # Initialize optimized audio processor
        audio_config_dict = Config.get_audio_config()
        self.audio_processor = OptimizedAudioProcessor(
            sample_rate=audio_config_dict["sample_rate"],
            channels=audio_config_dict["channels"],
            chunk_duration=audio_config_dict["chunk_duration"],
            buffer_size=audio_config_dict["buffer_size"],
            enable_vad=audio_config_dict["vad_enabled"],
            enable_compression=audio_config_dict["enable_compression"],
        )

        # Initialize audio chunker for large files
        self.audio_chunker = AudioChunker(
            chunk_duration=30.0,  # 30 seconds per chunk
            overlap_duration=2.0,  # 2 seconds overlap
            sample_rate=audio_config_dict["sample_rate"],
        )

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

        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "total_processing_time": 0.0,
            "avg_response_time": 0.0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
        }

        # Validate configuration
        self.speech_config.validate()

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for Groq API requirements.

        Args:
            audio_data: Input audio data

        Returns:
            Preprocessed audio data
        """
        # Ensure audio is mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed (Groq requirement)
        target_sample_rate = 16000
        current_sample_rate = (
            self.audio_config.sample_rate if self.audio_config else 16000
        )

        if current_sample_rate != target_sample_rate:
            # Simple resampling (in production, use librosa or scipy)
            ratio = target_sample_rate / current_sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data,
            )

        return audio_data

    def _call_groq_transcription_api(
        self, audio_buffer: io.BytesIO, is_translation: bool = False
    ) -> Dict[str, Any]:
        """
        Call Groq API for transcription or translation with optimized parameters.

        Args:
            audio_buffer: Audio data buffer
            is_translation: Whether to use translation endpoint

        Returns:
            API response as dictionary
        """
        try:
            # Get model configuration from environment
            model_config = Config.get_model_config()

            # Get configuration parameters with environment overrides
            model = model_config["model_id"]
            language = self.speech_config.speech_recognition_language
            response_format = model_config["response_format"]
            temperature = model_config["temperature"]
            prompt = (
                self.speech_config.get_property(PropertyId.Speech_Recognition_Prompt)
                or None
            )

            # Prepare timestamp granularities based on config
            timestamp_granularities = []
            if model_config["enable_word_timestamps"]:
                timestamp_granularities.append("word")
            if model_config["enable_segment_timestamps"]:
                timestamp_granularities.append("segment")

            # Default to segment if no granularities specified
            if not timestamp_granularities:
                timestamp_granularities = ["segment"]

            # Prepare API parameters
            api_params = {
                "file": ("audio.wav", audio_buffer.getvalue(), "audio/wav"),
                "model": model,
                "response_format": response_format,
                "timestamp_granularities": timestamp_granularities,
                "temperature": temperature,
            }

            # Add language parameter (only for transcription, not translation)
            if not is_translation and language:
                # Convert language code format (e.g., "en-US" -> "en")
                lang_code = language.split("-")[0] if "-" in language else language
                api_params["language"] = lang_code

            # Add prompt if specified
            if prompt:
                api_params["prompt"] = prompt

            # Call appropriate API endpoint
            if is_translation:
                # Translation endpoint only supports 'en' language
                api_params["language"] = "en"
                response = self.groq_client.audio.translations.create(**api_params)
            else:
                response = self.groq_client.audio.transcriptions.create(**api_params)

            return response

        except Exception as e:
            raise Exception(f"Groq API call failed: {str(e)}")

    def _parse_groq_response(
        self, response: Any, is_translation: bool = False
    ) -> SpeechRecognitionResult:
        """
        Parse Groq API response into SpeechRecognitionResult.

        Args:
            response: Groq API response
            is_translation: Whether this was a translation response

        Returns:
            Parsed recognition result
        """
        try:
            # Extract text
            text = getattr(response, "text", "")

            # Extract confidence (if available)
            confidence = getattr(response, "confidence", 0.95)  # Default confidence

            # Extract language
            language = getattr(
                response, "language", self.speech_config.speech_recognition_language
            )

            # Extract timestamps if verbose_json format
            timestamps = []
            if hasattr(response, "segments"):
                for segment in response.segments:
                    timestamp_info = {
                        "start": getattr(segment, "start", 0),
                        "end": getattr(segment, "end", 0),
                        "text": getattr(segment, "text", ""),
                        "avg_logprob": getattr(segment, "avg_logprob", 0),
                        "compression_ratio": getattr(segment, "compression_ratio", 0),
                        "no_speech_prob": getattr(segment, "no_speech_prob", 0),
                    }
                    timestamps.append(timestamp_info)

            return SpeechRecognitionResult(
                text=text,
                reason=ResultReason.RecognizedSpeech,
                confidence=confidence,
                language=language,
                timestamps=timestamps,
            )

        except Exception as e:
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=CancellationDetails(
                    CancellationReason.Error, f"Failed to parse API response: {str(e)}"
                ),
            )

    def connect(self, event_type: str, handler: Callable):
        """
        Connect an event handler.

        Args:
            event_type: Type of event ('recognizing', 'recognized', 'session_started',
                                     'session_stopped', 'canceled')
            handler: Event handler function
        """
        if event_type == "recognizing":
            self.recognizing_handlers.append(handler)
        elif event_type == "recognized":
            self.recognized_handlers.append(handler)
        elif event_type == "session_started":
            self.session_started_handlers.append(handler)
        elif event_type == "session_stopped":
            self.session_stopped_handlers.append(handler)
        elif event_type == "canceled":
            self.canceled_handlers.append(handler)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def _trigger_event(self, event_type: str, event_data: Any):
        """Trigger event handlers for a specific event type."""
        handlers = []
        if event_type == "recognizing":
            handlers = self.recognizing_handlers
        elif event_type == "recognized":
            handlers = self.recognized_handlers
        elif event_type == "session_started":
            handlers = self.session_started_handlers
        elif event_type == "session_stopped":
            handlers = self.session_stopped_handlers
        elif event_type == "canceled":
            handlers = self.canceled_handlers

        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                print(f"Error in event handler: {e}")

    def recognize_once_async(self) -> "SpeechRecognitionResult":
        """
        Perform single-shot speech recognition.

        Returns:
            SpeechRecognitionResult object
        """
        try:
            # Trigger session started event
            self._trigger_event(
                "session_started", {"session_id": f"session_{int(time.time())}"}
            )

            # Get audio data
            if self.audio_config and self.audio_config.filename:
                # File-based recognition
                audio_data = self.audio_config.get_file_audio_data()
                result = self._recognize_audio_data(audio_data)
            else:
                # Microphone-based recognition
                result = self._recognize_from_microphone()

            # Trigger session stopped event
            self._trigger_event(
                "session_stopped", {"session_id": f"session_{int(time.time())}"}
            )

            return result

        except Exception as e:
            cancellation_details = CancellationDetails(
                CancellationReason.Error, f"Recognition failed: {str(e)}"
            )
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled, cancellation_details=cancellation_details
            )

    def _recognize_audio_data(
        self, audio_data: np.ndarray, is_translation: bool = False
    ) -> SpeechRecognitionResult:
        """
        Recognize speech from audio data using Groq API.

        Args:
            audio_data: Audio data as numpy array
            is_translation: Whether to use translation endpoint

        Returns:
            SpeechRecognitionResult
        """
        timing_metrics = TimingMetrics()

        try:
            # Start API call timing
            timing_metrics.start_api_call()

            # Preprocess audio
            audio_data = self._preprocess_audio(audio_data)

            # Save audio to temporary buffer
            buffer = io.BytesIO()
            sf.write(
                buffer, audio_data, 16000, format="WAV"
            )  # Use 16kHz as per Groq requirements
            buffer.seek(0)

            # Call Groq API
            response = self._call_groq_transcription_api(buffer, is_translation)

            # End API call timing
            timing_metrics.end_api_call()

            # Start processing timing
            timing_metrics.start_processing()

            # Parse response
            result = self._parse_groq_response(response, is_translation)

            # End processing timing
            timing_metrics.end_processing()

            # Add timing metrics to result
            result.timing_metrics = timing_metrics

            return result

        except Exception as e:
            timing_metrics.end_processing()
            cancellation_details = CancellationDetails(
                CancellationReason.Error, f"Recognition failed: {str(e)}"
            )
            result = SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=cancellation_details,
                timing_metrics=timing_metrics,
            )
            return result

    def translate_audio_data(self, audio_data: np.ndarray) -> SpeechRecognitionResult:
        """
        Translate audio to English text using Groq API.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            SpeechRecognitionResult with English translation
        """
        return self._recognize_audio_data(audio_data, is_translation=True)

    def _recognize_from_microphone(self) -> SpeechRecognitionResult:
        """
        Recognize speech from microphone input.

        Returns:
            SpeechRecognitionResult
        """
        if not self.audio_config:
            self.audio_config = AudioConfig()

        timing_metrics = TimingMetrics()
        timing_metrics.start_microphone()

        try:
            print("Speak into your microphone...")
            print("(Press Ctrl+C to stop)")

            # Collect audio with improved parameters
            audio_chunks = []
            start_time = time.time()
            max_duration = 20  # Increased to 20 seconds for better capture
            silence_threshold = 0.5  # Seconds of silence to stop recording

            # Use the audio config to read audio
            with self.audio_config as audio:
                print("Recording audio...")
                last_audio_time = time.time()

                while time.time() - start_time < max_duration:
                    try:
                        chunk = audio.read_audio_chunk(4096)  # Increased chunk size
                        if chunk and len(chunk) > 0:
                            # Check if chunk contains audio (not just silence)
                            audio_array = np.frombuffer(chunk, dtype=np.int16)
                            audio_energy = np.mean(audio_array**2)

                            if audio_energy > 100:  # Threshold for audio activity
                                last_audio_time = time.time()
                                audio_chunks.append(chunk)
                                print(".", end="", flush=True)  # Show progress
                            elif time.time() - last_audio_time > silence_threshold:
                                # Stop if we've had silence for too long
                                print("\nSilence detected, stopping recording")
                                break

                    except KeyboardInterrupt:
                        print("\nStopped by user")
                        break
                    except Exception as e:
                        print(f"\nError reading audio: {e}")
                        break

                print()  # New line after progress dots

                if not audio_chunks:
                    print(
                        "No audio captured. Please try speaking louder or check your microphone."
                    )
                    timing_metrics.end_microphone()
                    return SpeechRecognitionResult(
                        reason=ResultReason.NoMatch,
                        no_match_details=NoMatchDetails("No audio captured"),
                        timing_metrics=timing_metrics,
                    )

                print(f"Captured {len(audio_chunks)} audio chunks")
                timing_metrics.end_microphone()

                # Combine audio chunks
                audio_data = b"".join(audio_chunks)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

                # Update performance stats
                self.performance_stats["total_requests"] += 1

                return self._recognize_audio_data(audio_array)

        except Exception as e:
            timing_metrics.end_microphone()
            self.performance_stats["failed_recognitions"] += 1
            cancellation_details = CancellationDetails(
                CancellationReason.Error, f"Microphone recognition failed: {str(e)}"
            )
            return SpeechRecognitionResult(
                reason=ResultReason.Canceled,
                cancellation_details=cancellation_details,
                timing_metrics=timing_metrics,
            )

    def start_continuous_recognition(self):
        """Start continuous speech recognition."""
        if self._is_recognizing:
            return

        self._is_recognizing = True
        self._stop_recognition = False
        self._recognition_thread = threading.Thread(
            target=self._continuous_recognition_worker
        )
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
                    self._trigger_event("recognized", result)
                elif result.reason == ResultReason.Canceled:
                    self._trigger_event("canceled", result)
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

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        audio_stats = self.audio_processor.get_performance_stats()

        return {
            "total_requests": self.performance_stats["total_requests"],
            "successful_recognitions": self.performance_stats[
                "successful_recognitions"
            ],
            "failed_recognitions": self.performance_stats["failed_recognitions"],
            "avg_response_time": self.performance_stats["avg_response_time"],
            "audio_processing": audio_stats,
            "model_config": Config.get_model_config(),
            "audio_config": Config.get_audio_config(),
        }
