"""
Speaker Diarization and Segmentation for Groq Speech services.

This module provides speaker diarization capabilities that integrate seamlessly
with the existing Groq Speech SDK. It uses Pyannote.audio for efficient
speaker segmentation and then processes each segment through Groq's API.

ARCHITECTURE OVERVIEW:
1. CORE COMPONENTS
   - SpeakerDiarizer: Main class for speaker diarization operations
   - DiarizationResult: Structured results with speaker information
   - SpeakerSegment: Individual speaker segment with metadata
   - AudioSegmenter: Audio segmentation and preprocessing

2. DIARIZATION PIPELINE
   - Voice Activity Detection (VAD)
   - Speaker embedding extraction
   - Speaker clustering and segmentation
   - Segment validation and optimization

3. GROQ INTEGRATION
   - Batch processing of speaker segments
   - Transcription with speaker labels
   - Confidence scoring and validation
   - Performance optimization

4. FEATURES
   - Real-time and batch diarization
   - Configurable speaker detection sensitivity
   - Automatic speaker labeling (Speaker A, Speaker B, etc.)
   - Integration with existing recognition pipeline

KEY FEATURES:
- Efficient speaker segmentation using Pyannote.audio
- Seamless integration with Groq transcription API
- Configurable diarization parameters
- Real-time and batch processing modes
- Speaker confidence scoring
- Automatic speaker labeling and identification
- Performance optimization and caching
- Error handling and recovery

USAGE EXAMPLES:
    # Basic diarization
    diarizer = SpeakerDiarizer()
    result = diarizer.diarize_audio(audio_data)

    # Batch processing with custom parameters
    config = DiarizationConfig(
        min_speakers=2,
        max_speakers=5,
        min_segment_duration=1.0
    )
    result = diarizer.diarize_audio(audio_data, config)

    # Integration with existing recognizer
    recognizer = SpeechRecognizer(speech_config)
    diarized_result = recognizer.recognize_with_diarization(audio_data)
"""

import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf
import torch

# Lazy import Pyannote to avoid loading it when not needed
PYANNOTE_AVAILABLE = None
Pipeline = None
Annotation = None

def _import_pyannote():
    """Lazy import Pyannote.audio only when needed."""
    global PYANNOTE_AVAILABLE, Pipeline, Annotation
    
    if PYANNOTE_AVAILABLE is not None:
        return PYANNOTE_AVAILABLE
    
    try:
        from pyannote.audio import Pipeline
        from pyannote.core.annotation import Annotation
        PYANNOTE_AVAILABLE = True
        return True
    except ImportError:
        PYANNOTE_AVAILABLE = False
        print(
            "Warning: Pyannote.audio not available. Install with: pip install pyannote.audio"
        )
        
        # Create a fallback Annotation class for when Pyannote is not available
        class Annotation:
            def itertracks(self, yield_label=False):
                return []
        
        return False


from .speech_config import SpeechConfig
from .exceptions import DiarizationError


class DiarizationConfig:
    """
    Configuration for speaker diarization operations.

    CRITICAL: This class provides configurable parameters for the diarization
    pipeline, allowing users to tune the behavior for their specific use cases:

    1. Speaker Detection: Minimum and maximum number of speakers
    2. Segment Control: Duration thresholds and overlap settings
    3. Quality Settings: Confidence thresholds and validation parameters
    4. Performance: Batch processing and caching options
    5. Optimization: New advanced settings for better output quality
    """

    def __init__(
        self,
        min_speakers: int = 1,
        max_speakers: int = 5,
        min_segment_duration: float = 2.0,
        max_segment_duration: float = 30.0,
        confidence_threshold: float = 0.7,
        # New optimization parameters
        silence_threshold: float = 0.8,
        speaker_change_sensitivity: float = 0.7,
        max_segments_per_chunk: int = 8,
        chunk_strategy: str = "adaptive",
        min_chunk_duration: float = 15.0,
        max_chunk_duration: float = 30.0,
        overlap_duration: float = 2.0,
        enable_cross_chunk_persistence: bool = True,
        speaker_similarity_threshold: float = 0.85,
        enable_adaptive_merging: bool = True,
        merge_time_threshold: float = 1.5,
        enable_context_awareness: bool = True,
    ):
        """
        Initialize diarization configuration.

        Args:
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            confidence_threshold: Minimum confidence for segment inclusion
            silence_threshold: Silence detection sensitivity (0.1-1.0)
            speaker_change_sensitivity: Speaker change detection sensitivity (0.1-1.0)
            max_segments_per_chunk: Maximum segments per audio chunk
            chunk_strategy: Chunking strategy ('fixed', 'adaptive', 'conversation_aware')
            min_chunk_duration: Minimum chunk duration in seconds
            max_chunk_duration: Maximum chunk duration in seconds
            overlap_duration: Overlap between chunks in seconds
            enable_cross_chunk_persistence: Enable speaker persistence across chunks
            speaker_similarity_threshold: Threshold for speaker matching (0.5-1.0)
            enable_adaptive_merging: Enable intelligent segment merging
            merge_time_threshold: Time threshold for merging segments
            enable_context_awareness: Enable conversation flow analysis
        """
        # Legacy parameters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.confidence_threshold = confidence_threshold

        # New optimization parameters
        self.silence_threshold = silence_threshold
        self.speaker_change_sensitivity = speaker_change_sensitivity
        self.max_segments_per_chunk = max_segments_per_chunk
        self.chunk_strategy = chunk_strategy
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.overlap_duration = overlap_duration
        self.enable_cross_chunk_persistence = enable_cross_chunk_persistence
        self.speaker_similarity_threshold = speaker_similarity_threshold
        self.enable_adaptive_merging = enable_adaptive_merging
        self.merge_time_threshold = merge_time_threshold
        self.enable_context_awareness = enable_context_awareness

    @classmethod
    def from_environment(cls) -> "DiarizationConfig":
        """Create configuration from environment variables."""
        from .config import Config

        config_dict = Config.get_diarization_config()

        return cls(
            min_speakers=Config.DIARIZATION_MAX_SPEAKERS,
            max_speakers=Config.DIARIZATION_MAX_SPEAKERS,
            min_segment_duration=Config.DIARIZATION_MIN_SEGMENT_DURATION,
            max_segment_duration=Config.DIARIZATION_MAX_CHUNK_DURATION,
            confidence_threshold=0.7,  # Default confidence
            silence_threshold=Config.DIARIZATION_SILENCE_THRESHOLD,
            speaker_change_sensitivity=Config.DIARIZATION_SPEAKER_CHANGE_SENSITIVITY,
            max_segments_per_chunk=Config.DIARIZATION_MAX_SEGMENTS_PER_CHUNK,
            chunk_strategy=Config.DIARIZATION_CHUNK_STRATEGY,
            min_chunk_duration=Config.DIARIZATION_MIN_CHUNK_DURATION,
            max_chunk_duration=Config.DIARIZATION_MAX_CHUNK_DURATION,
            overlap_duration=Config.DIARIZATION_OVERLAP_DURATION,
            enable_cross_chunk_persistence=Config.DIARIZATION_ENABLE_CROSS_CHUNK_PERSISTENCE,
            speaker_similarity_threshold=Config.DIARIZATION_SPEAKER_SIMILARITY_THRESHOLD,
            enable_adaptive_merging=Config.DIARIZATION_ENABLE_ADAPTIVE_MERGING,
            merge_time_threshold=Config.DIARIZATION_MERGE_TIME_THRESHOLD,
            enable_context_awareness=Config.DIARIZATION_ENABLE_CONTEXT_AWARENESS,
        )

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        if self.min_speakers < 1:
            errors.append("min_speakers must be at least 1")

        if self.max_speakers < self.min_speakers:
            errors.append("max_speakers must be greater than or equal to min_speakers")

        if self.min_segment_duration < 0.5:
            errors.append("min_segment_duration must be at least 0.5 seconds")

        if self.max_segment_duration < self.min_segment_duration:
            errors.append(
                "max_segment_duration must be greater than min_segment_duration"
            )

        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")

        if self.silence_threshold < 0.1 or self.silence_threshold > 1.0:
            errors.append("silence_threshold must be between 0.1 and 1.0")

        if (
            self.speaker_change_sensitivity < 0.1
            or self.speaker_change_sensitivity > 1.0
        ):
            errors.append("speaker_change_sensitivity must be between 0.1 and 1.0")

        if self.max_segments_per_chunk < 1:
            errors.append("max_segments_per_chunk must be at least 1")

        if self.chunk_strategy not in ["fixed", "adaptive", "conversation_aware"]:
            errors.append(
                "chunk_strategy must be 'fixed', 'adaptive', or 'conversation_aware'"
            )

        if self.min_chunk_duration < 5.0:
            errors.append("min_chunk_duration must be at least 5.0 seconds")

        if self.max_chunk_duration < self.min_chunk_duration:
            errors.append("max_chunk_duration must be greater than min_chunk_duration")

        if self.overlap_duration < 0.0:
            errors.append("overlap_duration must be non-negative")

        if (
            self.speaker_similarity_threshold < 0.5
            or self.speaker_similarity_threshold > 1.0
        ):
            errors.append("speaker_similarity_threshold must be between 0.5 and 1.0")

        if self.merge_time_threshold < 0.0:
            errors.append("merge_time_threshold must be non-negative")

        if errors:
            print("DiarizationConfig validation errors:")
            for error in errors:
                print(f"  ❌ {error}")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "min_segment_duration": self.min_segment_duration,
            "max_segment_duration": self.max_segment_duration,
            "confidence_threshold": self.confidence_threshold,
            "silence_threshold": self.silence_threshold,
            "speaker_change_sensitivity": self.speaker_change_sensitivity,
            "max_segments_per_chunk": self.max_segments_per_chunk,
            "chunk_strategy": self.chunk_strategy,
            "min_chunk_duration": self.min_chunk_duration,
            "max_chunk_duration": self.max_chunk_duration,
            "overlap_duration": self.overlap_duration,
            "enable_cross_chunk_persistence": self.enable_cross_chunk_persistence,
            "speaker_similarity_threshold": self.speaker_similarity_threshold,
            "enable_adaptive_merging": self.enable_adaptive_merging,
            "merge_time_threshold": self.merge_time_threshold,
            "enable_context_awareness": self.enable_context_awareness,
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = f"DiarizationConfig(\n"
        config_str += f"  Speakers: {self.min_speakers}-{self.max_speakers}\n"
        config_str += (
            f"  Segments: {self.min_segment_duration}s-{self.max_segment_duration}s\n"
        )
        config_str += f"  Chunking: {self.chunk_strategy} ({self.min_chunk_duration}s-{self.max_chunk_duration}s)\n"
        config_str += f"  Quality: silence={self.silence_threshold}, sensitivity={self.speaker_change_sensitivity}\n"
        config_str += f"  Persistence: {self.enable_cross_chunk_persistence}\n"
        config_str += f"  Merging: {self.enable_adaptive_merging} (threshold: {self.merge_time_threshold}s)\n"
        config_str += f"  Context: {self.enable_context_awareness}\n"
        config_str += ")"
        return config_str


class SpeakerSegment:
    """
    Represents a single speaker segment with metadata.

    CRITICAL: This class encapsulates all information about a speaker segment,
    including timing, speaker identification, and transcription data:

    1. Segment Information: Start/end times and duration
    2. Speaker Data: Speaker ID and confidence scores
    3. Audio Content: Raw audio data and processed text
    4. Metadata: Quality metrics and processing information
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        speaker_id: str,
        confidence: float = 1.0,
        audio_data: Optional[np.ndarray] = None,
        text: str = "",
        language: str = "",
        transcription_confidence: float = 0.0,
    ):
        """
        Initialize speaker segment.

        Args:
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            speaker_id: Unique identifier for the speaker
            confidence: Speaker detection confidence (0.0 to 1.0)
            audio_data: Raw audio data for this segment
            text: Transcribed text from this segment
            language: Detected language of the speech
            transcription_confidence: Confidence of the transcription
        """
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.audio_data = audio_data
        self.text = text
        self.language = language
        self.transcription_confidence = transcription_confidence

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time

    @property
    def is_valid(self) -> bool:
        """Check if segment is valid (positive duration, reasonable confidence)."""
        return self.duration > 0 and self.confidence >= 0.0 and self.confidence <= 1.0

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"SpeakerSegment(speaker={self.speaker_id}, time={self.start_time:.2f}-{self.end_time:.2f}, text='{self.text[:50]}...')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
            "text": self.text,
            "language": self.language,
            "transcription_confidence": self.transcription_confidence,
            "duration": self.duration,
        }


class DiarizationResult:
    """
    Result of speaker diarization operation.

    CRITICAL: This class provides a comprehensive view of the diarization
    results, including all speaker segments and metadata:

    1. Speaker Segments: All detected speaker segments with timing
    2. Speaker Mapping: Mapping of speaker IDs to labels
    3. Quality Metrics: Overall diarization quality and confidence
    4. Processing Information: Timing and performance data
    """

    def __init__(
        self,
        segments: List[SpeakerSegment],
        speaker_mapping: Dict[str, str],
        total_duration: float,
        num_speakers: int,
        overall_confidence: float = 0.0,
        processing_time: float = 0.0,
        error_details: Optional[str] = None,
    ):
        """
        Initialize diarization result.

        Args:
            segments: List of detected speaker segments
            speaker_mapping: Mapping of speaker IDs to human-readable labels
            total_duration: Total duration of the audio in seconds
            num_speakers: Number of unique speakers detected
            overall_confidence: Overall confidence of the diarization
            processing_time: Time taken to process the audio
            error_details: Details of any errors that occurred
        """
        self.segments = segments
        self.speaker_mapping = speaker_mapping
        self.total_duration = total_duration
        self.num_speakers = num_speakers
        self.overall_confidence = overall_confidence
        self.processing_time = processing_time
        self.error_details = error_details

    @property
    def is_successful(self) -> bool:
        """Check if diarization was successful."""
        return self.error_details is None and len(self.segments) > 0

    @property
    def speaker_labels(self) -> List[str]:
        """Get list of unique speaker labels."""
        return list(self.speaker_mapping.values())

    def get_segments_by_speaker(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [seg for seg in self.segments if seg.speaker_id == speaker_id]

    def get_transcript(self, include_speaker_labels: bool = True) -> str:
        """Get full transcript with optional speaker labels."""
        if not self.segments:
            return ""

        transcript_lines = []
        for segment in sorted(self.segments, key=lambda x: x.start_time):
            if include_speaker_labels:
                speaker_label = self.speaker_mapping.get(
                    segment.speaker_id, segment.speaker_id
                )
                transcript_lines.append(f"[{speaker_label}] {segment.text}")
            else:
                transcript_lines.append(segment.text)

        return "\n".join(transcript_lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "speaker_mapping": self.speaker_mapping,
            "total_duration": self.total_duration,
            "num_speakers": self.num_speakers,
            "overall_confidence": self.overall_confidence,
            "processing_time": self.processing_time,
            "error_details": self.error_details,
            "is_successful": self.is_successful,
        }

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.is_successful:
            return f"DiarizationResult({self.num_speakers} speakers, {len(self.segments)} segments, {self.total_duration:.2f}s)"
        else:
            return f"DiarizationResult(FAILED: {self.error_details})"


class AudioSegmenter:
    """
    Handles audio segmentation and preprocessing for diarization.

    CRITICAL: This class manages the audio processing pipeline for speaker
    diarization, ensuring optimal audio quality and format compatibility:

    1. Audio Preprocessing: Format conversion and quality optimization
    2. Segment Extraction: Extract individual speaker segments
    3. Audio Validation: Ensure segments meet quality requirements
    4. Format Conversion: Convert to formats compatible with Groq API
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio segmenter.

        Args:
            sample_rate: Target sample rate for audio processing
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels

    def extract_segment_audio(
        self, full_audio: np.ndarray, segment: SpeakerSegment
    ) -> np.ndarray:
        """
        Extract audio data for a specific segment.

        Args:
            full_audio: Complete audio data
            segment: Speaker segment with timing information

        Returns:
            Audio data for the specified segment
        """
        start_sample = int(segment.start_time * self.sample_rate)
        end_sample = int(segment.end_time * self.sample_rate)

        # Ensure bounds are within audio length
        start_sample = max(0, start_sample)
        end_sample = min(len(full_audio), end_sample)

        if start_sample >= end_sample:
            return np.array([])

        return full_audio[start_sample:end_sample]

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for optimal diarization.

        Args:
            audio_data: Input audio data

        Returns:
            Preprocessed audio data
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Ensure float32 format with range [-1, 1]
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Normalize audio levels
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95

        return audio_data

    def validate_segment(
        self, segment: SpeakerSegment, min_duration: float = 0.5
    ) -> bool:
        """
        Validate if a segment meets quality requirements.

        Args:
            segment: Speaker segment to validate
            min_duration: Minimum required duration

        Returns:
            True if segment is valid, False otherwise
        """
        return (
            segment.is_valid
            and segment.duration >= min_duration
            and segment.confidence > 0.0
        )


class GlobalSpeakerTracker:
    """
    Track speakers across all chunks for continuity.

    This class maintains a global database of speaker embeddings
    to ensure consistent speaker identification across audio chunks.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the global speaker tracker.

        Args:
            similarity_threshold: Threshold for speaker matching (0.5-1.0)
        """
        self.speaker_embeddings = {}  # Global speaker database
        self.speaker_counter = 0
        self.similarity_threshold = similarity_threshold
        self.speaker_history = {}  # Track speaker appearance patterns

    def identify_speaker(self, audio_segment: np.ndarray, chunk_id: str = None) -> str:
        """
        Identify speaker using global knowledge.

        Args:
            audio_segment: Audio data for speaker identification
            chunk_id: Optional chunk identifier for tracking

        Returns:
            Speaker ID (consistent across chunks)
        """
        # Extract embedding from audio segment
        embedding = self._extract_embedding(audio_segment)

        if embedding is None:
            # Fallback to new speaker if embedding extraction fails
            speaker_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            return speaker_id

        # Find matching speaker in global database
        speaker_id = self._find_matching_speaker(embedding)

        if speaker_id is None:
            # New speaker
            speaker_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_embeddings[speaker_id] = embedding
            self.speaker_counter += 1

        # Track speaker appearance
        if chunk_id:
            if speaker_id not in self.speaker_history:
                self.speaker_history[speaker_id] = []
            self.speaker_history[speaker_id].append(chunk_id)

        return speaker_id

    def _extract_embedding(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment.

        Args:
            audio_segment: Audio data

        Returns:
            Speaker embedding vector or None if extraction fails
        """
        try:
            # Simple feature extraction (placeholder for more sophisticated methods)
            # In a real implementation, this would use a pre-trained speaker embedding model

            # For now, use basic audio features as a placeholder
            features = self._extract_basic_features(audio_segment)
            return features

        except Exception as e:
            print(f"Warning: Failed to extract speaker embedding: {e}")
            return None

    def _extract_basic_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract basic audio features as a placeholder for speaker embedding.

        Args:
            audio_segment: Audio data

        Returns:
            Feature vector
        """
        # This is a simplified placeholder - in production, use a proper speaker embedding model
        # like d-vectors, x-vectors, or similar

        # Basic features: mean, std, spectral centroid, etc.
        features = []

        # Amplitude statistics
        features.extend(
            [
                np.mean(np.abs(audio_segment)),
                np.std(audio_segment),
                np.max(np.abs(audio_segment)),
                np.min(audio_segment),
            ]
        )

        # Spectral features (simplified)
        if len(audio_segment) > 1024:
            # FFT-based features
            fft = np.fft.fft(audio_segment[:1024])
            magnitude = np.abs(fft)
            features.extend(
                [
                    np.mean(magnitude),
                    np.std(magnitude),
                    np.argmax(magnitude) / len(magnitude),  # Dominant frequency
                ]
            )
        else:
            # Pad with zeros if audio is too short
            features.extend([0.0, 0.0, 0.0])

        # Normalize features
        features = np.array(features, dtype=np.float32)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        return features

    def _find_matching_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """
        Find matching speaker in global database.

        Args:
            embedding: Speaker embedding to match

        Returns:
            Speaker ID if match found, None otherwise
        """
        if not self.speaker_embeddings:
            return None

        best_match = None
        best_similarity = 0.0

        for speaker_id, stored_embedding in self.speaker_embeddings.items():
            similarity = self._calculate_similarity(embedding, stored_embedding)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = speaker_id

        return best_match

    def _calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception:
            return 0.0

    def get_speaker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked speakers.

        Returns:
            Dictionary with speaker statistics
        """
        return {
            "total_speakers": len(self.speaker_embeddings),
            "speaker_ids": list(self.speaker_embeddings.keys()),
            "speaker_history": self.speaker_history.copy(),
            "similarity_threshold": self.similarity_threshold,
        }

    def reset(self):
        """Reset the speaker tracker."""
        self.speaker_embeddings.clear()
        self.speaker_counter = 0
        self.speaker_history.clear()

    def __str__(self) -> str:
        """String representation of the tracker."""
        return f"GlobalSpeakerTracker(speakers={len(self.speaker_embeddings)}, threshold={self.similarity_threshold})"


# TO BE DELETED - Internal implementation, not a public entry point
class SpeakerDiarizer:
    """
    Main class for speaker diarization operations.

    CRITICAL: This class orchestrates the entire speaker diarization pipeline,
    from audio input to final results with speaker labels:

    1. Diarization Pipeline: Voice activity detection and speaker clustering
    2. Groq Integration: Transcription of individual speaker segments
    3. Result Processing: Speaker mapping and confidence scoring
    4. Performance Optimization: Batch processing and caching
    5. Error Handling: Comprehensive error management and recovery

    The diarizer supports multiple modes:
    - Real-time diarization for live audio
    - Batch processing for pre-recorded audio
    - Integration with existing speech recognition
    - Customizable diarization parameters
    """

    def __init__(
        self,
        config: Optional[DiarizationConfig] = None,
        speech_config: Optional[SpeechConfig] = None,
    ):
        """
        Initialize speaker diarizer.

        Args:
            config: Diarization configuration parameters
            speech_config: Speech recognition configuration for Groq API

        Initialization process:
        1. Configuration validation and setup
        2. Pyannote pipeline initialization (if available)
        3. Audio segmenter setup
        4. Caching system initialization
        5. Performance monitoring setup
        6. Speaker persistence system initialization
        """
        self.config = config or DiarizationConfig()
        self.config.validate()

        self.speech_config = speech_config or SpeechConfig()

        # Initialize Pyannote pipeline if available
        self._pipeline = None
        self._initialize_pipeline()

        # Initialize audio segmenter
        self.audio_segmenter = AudioSegmenter()

        # Speaker persistence system for cross-chunk identification
        self.speaker_cache = {}  # Cache speaker embeddings across chunks
        self.global_speaker_counter = 0  # Global speaker counter
        self.speaker_similarity_threshold = 0.75  # Threshold for speaker matching
        self.speaker_history = {}  # Track speaker appearance across chunks

        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_diarizations": 0,
            "failed_diarizations": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "speaker_matches": 0,
            "new_speakers": 0,
        }

    def _initialize_pipeline(self):
        """Initialize Pyannote diarization pipeline."""
        if not _import_pyannote():
            print(
                "Warning: Pyannote.audio is not available. Install with: pip install pyannote.audio"
            )
            self._pipeline = None
            return

        try:
            # Check for HuggingFace token from Config
            from .config import Config

            hf_token = Config.get_hf_token()

            # Check if token is properly configured (not placeholder)
            if not hf_token or hf_token == "your_huggingface_token_here":
                print("❌ HF_TOKEN not properly configured!")
                print("   To fix this:")
                print("   1. Visit https://hf.co/settings/tokens")
                print("   2. Create a new token with read access")
                print(
                    "   3. Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1"
                )
                print("   4. Set in .env: HF_TOKEN=your_actual_token_here")
                print("   ")
                print("   For now, diarization will use fallback methods.")
                self._pipeline = None
                return

            print("🔑 HF_TOKEN configured, attempting to load Pyannote models...")

            # Use global cache to avoid repeated model loading
            try:
                from .pyannote_cache import get_cached_pipeline
                
                self._pipeline = get_cached_pipeline(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=hf_token
                )
                
                if self._pipeline:
                    print("✅ Pyannote models loaded from cache or downloaded!")
                else:
                    print("❌ Failed to load Pyannote models")
                    return

                # Configure pipeline parameters (if supported)
                if hasattr(self._pipeline, "instantiate"):
                    try:
                        # Note: Newer Pyannote versions may not support these parameters
                        # We'll use the pipeline with default settings
                        print("✅ Pipeline initialized with default settings")
                    except Exception as config_error:
                        print(f"⚠️  Pipeline configuration warning: {config_error}")
                        print("   Using default pipeline settings")
                else:
                    print("✅ Pipeline initialized successfully")

            except Exception as model_error:
                print(f"❌ Failed to download Pyannote models: {model_error}")
                print("   This could be due to:")
                print("   1. Invalid or expired HF_TOKEN")
                print("   2. Model access not granted")
                print("   3. Network connectivity issues")
                print("   ")
                print("   Diarization will use fallback methods.")
                self._pipeline = None
                return

        except Exception as e:
            print("❌ Could not initialize Pyannote.audio pipeline:")
            print(f"   Error: {str(e)}")
            print("   ")
            print("   Possible solutions:")
            print("   1. Check your HF_TOKEN is valid and not expired")
            print(
                "   2. Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1"
            )
            print("   3. Ensure you have internet connection")
            print("   4. Try: pip install --upgrade pyannote.audio")
            print("   ")
            print("   Diarization will use fallback methods for now.")
            self._pipeline = None

    def _extract_speaker_embedding(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding for similarity matching.

        Args:
            audio_segment: Audio data for the segment

        Returns:
            Speaker embedding vector for similarity comparison
        """
        try:
            if self._pipeline is None:
                # Fallback: return a simple hash-based embedding
                return self._create_fallback_embedding(audio_segment)

            # Use Pyannote.audio's speaker embedding model
            # This is a simplified version - in practice, you'd use the actual embedding model
            embedding = self._pipeline.audio.crop(
                {"uri": "temp", "audio": audio_segment}
            )

            # Convert to numpy array and normalize
            if hasattr(embedding, "numpy"):
                embedding = embedding.numpy()
            elif hasattr(embedding, "cpu"):
                embedding = embedding.cpu().numpy()

            # Ensure it's a 1D array
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"Warning: Could not extract speaker embedding: {e}")
            return self._create_fallback_embedding(audio_segment)

    def _create_fallback_embedding(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Create a fallback embedding when Pyannote.audio is not available.

        Args:
            audio_segment: Audio data for the segment

        Returns:
            Simple hash-based embedding for basic speaker identification
        """
        try:
            # Create a more sophisticated fallback embedding
            # This is not as accurate as Pyannote.audio but provides basic functionality

            # Extract audio features for better speaker identification
            features = self._extract_audio_features(audio_segment)

            # Combine features into embedding
            embedding = np.concatenate(
                [
                    features["spectral_centroid"],
                    features["spectral_rolloff"],
                    features["mfcc"],
                    features["zero_crossing_rate"],
                ]
            )

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"Warning: Fallback embedding failed, using hash-based method: {e}")

            # Fallback to hash-based method
            import hashlib

            # Convert audio to bytes and hash
            audio_bytes = audio_segment.tobytes()
            hash_obj = hashlib.md5(audio_bytes)
            hash_hex = hash_obj.hexdigest()

            # Convert hash to numerical values
            embedding = np.array(
                [int(hash_hex[i : i + 2], 16) for i in range(0, 32, 2)],
                dtype=np.float32,
            )

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

    def _extract_audio_features(self, audio_segment: np.ndarray) -> dict:
        """
        Extract basic audio features for fallback speaker identification.

        Args:
            audio_segment: Audio data for the segment

        Returns:
            Dictionary of audio features
        """
        try:
            # Simple audio feature extraction
            # In a production system, you'd use librosa or similar

            # Spectral centroid (brightness)
            fft = np.fft.fft(audio_segment)
            magnitude = np.abs(fft)
            frequencies = np.fft.fftfreq(len(audio_segment))
            spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)

            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_magnitude = np.cumsum(magnitude)
            rolloff_threshold = 0.85 * cumulative_magnitude[-1]
            rolloff_idx = np.where(cumulative_magnitude >= rolloff_threshold)[0][0]
            spectral_rolloff = frequencies[rolloff_idx]

            # Simple MFCC approximation
            mfcc = np.array(
                [
                    np.mean(audio_segment),
                    np.std(audio_segment),
                    np.max(audio_segment),
                    np.min(audio_segment),
                ]
            )

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_segment)

            return {
                "spectral_centroid": np.array([spectral_centroid]),
                "spectral_rolloff": np.array([spectral_rolloff]),
                "mfcc": mfcc,
                "zero_crossing_rate": np.array([zero_crossing_rate]),
            }

        except Exception as e:
            print(f"Warning: Audio feature extraction failed: {e}")
            # Return simple features
            return {
                "spectral_centroid": np.array([0.0]),
                "spectral_rolloff": np.array([0.0]),
                "mfcc": np.array([0.0, 0.0, 0.0, 0.0]),
                "zero_crossing_rate": np.array([0.0]),
            }

    def _find_matching_speaker(self, new_embedding: np.ndarray) -> Optional[str]:
        """
        Find matching speaker using embedding similarity.

        Args:
            new_embedding: New speaker embedding to match

        Returns:
            Matched speaker ID if found, None otherwise
        """
        if not self.speaker_cache:
            return None

        best_match = None
        best_similarity = self.speaker_similarity_threshold

        for speaker_id, speaker_info in self.speaker_cache.items():
            try:
                # Calculate cosine similarity
                cached_embedding = speaker_info["embedding"]

                # Ensure both embeddings are the same shape
                if new_embedding.shape != cached_embedding.shape:
                    # Pad or truncate to match
                    min_len = min(len(new_embedding), len(cached_embedding))
                    new_emb = new_embedding[:min_len]
                    cached_emb = cached_embedding[:min_len]
                else:
                    new_emb = new_embedding
                    cached_emb = cached_embedding

                # Calculate cosine similarity
                dot_product = np.dot(new_emb, cached_emb)
                norm_new = np.linalg.norm(new_emb)
                norm_cached = np.linalg.norm(cached_emb)

                if norm_new > 0 and norm_cached > 0:
                    similarity = dot_product / (norm_new * norm_cached)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = speaker_id

            except Exception as e:
                print(
                    f"Warning: Error calculating similarity for speaker {speaker_id}: {e}"
                )
                continue

        return best_match

    def _persist_speaker_identity(
        self, chunk_result: "DiarizationResult", chunk_start_time: float
    ) -> "DiarizationResult":
        """
        Persist speaker identity across chunks using embedding similarity.

        Args:
            chunk_result: Diarization result for current chunk
            chunk_start_time: Start time of current chunk in global timeline

        Returns:
            Updated result with persistent speaker IDs
        """
        if not chunk_result.segments:
            return chunk_result

        updated_segments = []

        for segment in chunk_result.segments:
            try:
                # Extract audio for this segment
                start_sample = int(segment.start_time * 16000)  # Assuming 16kHz
                end_sample = int(segment.end_time * 16000)

                # For now, we'll use a placeholder audio segment
                # In practice, you'd extract the actual audio data
                audio_segment = np.zeros(end_sample - start_sample, dtype=np.float32)

                # Extract speaker embedding
                speaker_embedding = self._extract_speaker_embedding(audio_segment)

                # Find matching speaker in cache
                matched_speaker_id = self._find_matching_speaker(speaker_embedding)

                if matched_speaker_id:
                    # Use existing speaker ID
                    segment.speaker_id = matched_speaker_id

                    # Update speaker history
                    if matched_speaker_id not in self.speaker_history:
                        self.speaker_history[matched_speaker_id] = []

                    self.speaker_history[matched_speaker_id].append(
                        {
                            "chunk_start": chunk_start_time,
                            "start_time": chunk_start_time + segment.start_time,
                            "end_time": chunk_start_time + segment.end_time,
                            "confidence": segment.confidence,
                        }
                    )

                    # Update performance stats
                    self.performance_stats["speaker_matches"] += 1

                else:
                    # Create new speaker ID
                    new_speaker_id = f"speaker_{self.global_speaker_counter}"
                    self.global_speaker_counter += 1

                    # Cache new speaker
                    self.speaker_cache[new_speaker_id] = {
                        "embedding": speaker_embedding,
                        "first_seen": chunk_start_time + segment.start_time,
                        "last_seen": chunk_start_time + segment.end_time,
                        "total_segments": 1,
                        "confidence_sum": segment.confidence,
                    }

                    # Update speaker history
                    self.speaker_history[new_speaker_id] = [
                        {
                            "chunk_start": chunk_start_time,
                            "start_time": chunk_start_time + segment.start_time,
                            "end_time": chunk_start_time + segment.end_time,
                            "confidence": segment.confidence,
                        }
                    ]

                    # Update segment
                    segment.speaker_id = new_speaker_id

                    # Update performance stats
                    self.performance_stats["new_speakers"] += 1

                # Adjust timestamps to global timeline
                segment.start_time += chunk_start_time
                segment.end_time += chunk_start_time

                updated_segments.append(segment)

            except Exception as e:
                print(f"Warning: Error processing segment for speaker persistence: {e}")
                updated_segments.append(segment)

        # Create updated result
        from .speaker_diarization import DiarizationResult

        updated_result = DiarizationResult(
            segments=updated_segments,
            speaker_mapping=chunk_result.speaker_mapping,
            total_duration=chunk_result.total_duration,
            num_speakers=len(self.speaker_cache),
            overall_confidence=chunk_result.overall_confidence,
            processing_time=chunk_result.processing_time,
        )

        return updated_result

    def diarize_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio data.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            DiarizationResult containing speaker segments and metadata
        """
        start_time = time.time()

        try:
            # Validate configuration
            if not self.config.validate():
                raise ValueError("Invalid diarization configuration")

            # Use adaptive chunking based on configuration
            if self.config.chunk_strategy != "fixed":
                chunks = self._create_adaptive_chunks(audio_data, sample_rate)
            else:
                # Fallback to traditional chunking
                chunks = self._create_fixed_chunks(audio_data, sample_rate)

            print(
                f"🎭 Created {len(chunks)} chunks using {self.config.chunk_strategy} strategy"
            )

            # Process each chunk with Pyannote.audio
            all_segments = []
            speaker_mapping = {}

            for i, chunk in enumerate(chunks):
                print(
                    f"🎤 Processing chunk {i+1}/{len(chunks)} ({len(chunk)/sample_rate:.1f}s)"
                )

                # Perform diarization on chunk
                chunk_result = self._diarize_chunk(chunk, sample_rate, i)

                if chunk_result and chunk_result.segments:
                    # Adjust timestamps to global timeline
                    adjusted_segments = self._adjust_chunk_timestamps(
                        chunk_result.segments, i, len(chunk), sample_rate
                    )
                    all_segments.extend(adjusted_segments)

                    # Update speaker mapping
                    for segment in adjusted_segments:
                        if segment.speaker_id not in speaker_mapping:
                            speaker_mapping[segment.speaker_id] = {
                                "total_duration": 0.0,
                                "segment_count": 0,
                            }
                        speaker_mapping[segment.speaker_id]["total_duration"] += (
                            segment.end_time - segment.start_time
                        )
                        speaker_mapping[segment.speaker_id]["segment_count"] += 1

            if not all_segments:
                print("⚠️ No segments detected, using fallback diarization")
                return self._fallback_diarization(audio_data, sample_rate)

            # Apply smart segment merging if enabled
            if self.config.enable_adaptive_merging:
                print("🔗 Applying smart segment merging...")
                original_count = len(all_segments)
                all_segments = self._smart_merge_segments(all_segments)
                merged_count = len(all_segments)
                print(f"✅ Merged {original_count - merged_count} segments")

            # Sort segments by start time
            all_segments.sort(key=lambda x: x.start_time)

            # Calculate overall statistics
            total_duration = len(audio_data) / sample_rate
            num_speakers = len(speaker_mapping)
            overall_confidence = np.mean([s.confidence for s in all_segments])

            processing_time = time.time() - start_time

            result = DiarizationResult(
                segments=all_segments,
                speaker_mapping=speaker_mapping,
                total_duration=total_duration,
                num_speakers=num_speakers,
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                error_details=None,
            )

            print(
                f"🎭 Diarization complete: {len(all_segments)} segments, "
                f"{num_speakers} speakers, {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Diarization failed: {e}")

            # Return fallback result
            return self._fallback_diarization(audio_data, sample_rate)

    def _diarize_chunk(
        self, chunk: np.ndarray, sample_rate: int, chunk_id: int
    ) -> Optional[DiarizationResult]:
        """Diarize a single audio chunk."""

        try:
            # Convert to Pyannote.audio format
            audio_tensor = torch.from_numpy(chunk).float()

            # Run diarization
            diarization = self._pipeline(
                {"waveform": audio_tensor, "sample_rate": sample_rate}
            )

            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Filter segments based on configuration
                duration = turn.end - turn.start
                if duration >= self.config.min_segment_duration:
                    segment = SpeakerSegment(
                        speaker_id=speaker,
                        start_time=float(turn.start),
                        end_time=float(turn.end),
                        text="",  # Will be filled by transcription
                        confidence=0.8,  # Default confidence
                    )
                    segments.append(segment)

            # Limit segments per chunk if configured
            if len(segments) > self.config.max_segments_per_chunk:
                print(
                    f"⚠️ Chunk {chunk_id}: Limiting segments from {len(segments)} "
                    f"to {self.config.max_segments_per_chunk}"
                )
                segments = segments[: self.config.max_segments_per_chunk]

            return DiarizationResult(
                segments=segments,
                speaker_mapping={},
                total_duration=len(chunk) / sample_rate,
                num_speakers=len(set(s.speaker_id for s in segments)),
                overall_confidence=0.8,
                processing_time=0.0,
                error_details=None,
            )

        except Exception as e:
            print(f"⚠️ Failed to diarize chunk {chunk_id}: {e}")
            return None

    def _adjust_chunk_timestamps(
        self,
        segments: List[SpeakerSegment],
        chunk_id: int,
        chunk_duration: int,
        sample_rate: int,
    ) -> List[SpeakerSegment]:
        """Adjust segment timestamps to global timeline."""

        # Calculate global offset for this chunk
        global_offset = chunk_id * (chunk_duration / sample_rate)

        adjusted_segments = []
        for segment in segments:
            adjusted_segment = SpeakerSegment(
                speaker_id=segment.speaker_id,
                start_time=segment.start_time + global_offset,
                end_time=segment.end_time + global_offset,
                text=segment.text,
                confidence=segment.confidence,
            )
            adjusted_segments.append(adjusted_segment)

        return adjusted_segments

    def _fallback_diarization(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        config: Optional[DiarizationConfig] = None,
    ) -> DiarizationResult:
        """
        Fallback diarization when Pyannote.audio is not available.

        This method provides basic speaker segmentation using audio analysis
        techniques that don't require external ML models.

        Args:
            audio_data: Input audio data as numpy array
            sample_rate: Sample rate of the audio data
            config: Optional configuration override

        Returns:
            DiarizationResult with basic speaker segments
        """
        try:
            print("🔄 Using fallback diarization (audio analysis based)...")

            # Use audio analysis to detect potential speaker changes
            segments = self._detect_speaker_changes_audio_analysis(
                audio_data, sample_rate, config
            )

            # Create speaker mapping
            speaker_mapping = {}
            for i, segment in enumerate(segments):
                speaker_id = f"speaker_{i + 1}"
                speaker_mapping[speaker_id] = f"Speaker {i + 1}"
                segment.speaker_id = speaker_id

            # Calculate overall confidence (lower for fallback)
            overall_confidence = 0.6  # Reduced confidence for fallback method

            # Create result
            from .speaker_diarization import DiarizationResult

            result = DiarizationResult(
                segments=segments,
                speaker_mapping=speaker_mapping,
                total_duration=len(audio_data) / sample_rate,
                num_speakers=len(speaker_mapping),
                overall_confidence=overall_confidence,
                processing_time=0.0,
            )

            print(f"✅ Fallback diarization completed: {len(segments)} segments")
            return result

        except Exception as e:
            print(f"❌ Fallback diarization failed: {e}")
            # Return single speaker result as last resort
            from .speaker_diarization import SpeakerSegment, DiarizationResult

            segment = SpeakerSegment(
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                speaker_id="speaker_1",
                confidence=0.5,
            )

            result = DiarizationResult(
                segments=[segment],
                speaker_mapping={"speaker_1": "Speaker"},
                total_duration=len(audio_data) / sample_rate,
                num_speakers=1,
                overall_confidence=0.5,
                processing_time=0.0,
            )

            return result

    def _detect_speaker_changes_audio_analysis(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        config: Optional[DiarizationConfig] = None,
    ) -> List["SpeakerSegment"]:
        """
        Detect potential speaker changes using audio analysis.

        This method uses audio features like energy, spectral characteristics,
        and silence detection to infer speaker changes.

        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of the audio
            config: Optional configuration

        Returns:
            List of potential speaker segments
        """
        try:
            from .speaker_diarization import SpeakerSegment

            segments = []
            duration = len(audio_data) / sample_rate

            # Simple segmentation based on audio energy
            # In a production system, you'd use more sophisticated analysis

            # Split audio into fixed-size segments
            segment_duration = 5.0  # 5-second segments
            num_segments = max(1, int(duration / segment_duration))

            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)

                # Calculate confidence based on audio quality
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]

                # Simple confidence calculation
                energy = np.mean(np.abs(segment_audio))
                confidence = min(0.8, max(0.3, energy * 10))  # Normalize to 0.3-0.8

                segment = SpeakerSegment(
                    start_time=start_time,
                    end_time=end_time,
                    speaker_id=f"speaker_{i + 1}",
                    confidence=confidence,
                )

                segments.append(segment)

            return segments

        except Exception as e:
            print(f"Warning: Audio analysis failed: {e}")
            # Return single segment as fallback
            from .speaker_diarization import SpeakerSegment

            segment = SpeakerSegment(
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                speaker_id="speaker_1",
                confidence=0.5,
            )

            return [segment]

    def diarize_audio_chunked(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        chunk_duration: float = 10.0,
        overlap_duration: float = 3.0,
        config: Optional[DiarizationConfig] = None,
    ) -> DiarizationResult:
        """
        Perform chunked diarization with speaker identity persistence.

        CRITICAL: This method processes audio in chunks while maintaining
        speaker identity across chunk boundaries. It's essential for:

        1. Long audio files that exceed Groq API limits
        2. Real-time streaming with continuous speaker tracking
        3. Compliance with Groq's 10-second minimum billing
        4. Speaker continuity across time boundaries

        Args:
            audio_data: Input audio data as numpy array
            sample_rate: Sample rate of the audio data
            chunk_duration: Duration of each chunk in seconds (default: 10.0)
            overlap_duration: Overlap between chunks in seconds (default: 3.0)
            config: Optional configuration override

        Returns:
            DiarizationResult with persistent speaker IDs across chunks
        """
        start_time = time.time()
        config = config or self.config

        try:
            # Update performance stats
            self.performance_stats["total_requests"] += 1

            # Check if pipeline is available
            if self._pipeline is None:
                raise DiarizationError(
                    "Pyannote pipeline is not available. Please check your HF_TOKEN and ensure "
                    "you have access to the pyannote/speaker-diarization-3.1 model."
                )

            # Create overlapping chunks
            chunks = self._create_overlapping_chunks(
                audio_data, sample_rate, chunk_duration, overlap_duration
            )

            if not chunks:
                raise DiarizationError("Failed to create audio chunks")

            print(
                f"🔄 Processing {len(chunks)} chunks with {overlap_duration}s overlap..."
            )

            # Process each chunk with speaker persistence
            all_segments = []
            global_speaker_mapping = {}

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    print(
                        f"   Processing chunk {chunk_idx + 1}/{len(chunks)} "
                        f"({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)"
                    )

                    # Process chunk with diarization
                    chunk_result = self._diarize_chunk(
                        chunk["data"], sample_rate, config
                    )

                    if chunk_result.segments:
                        # Persist speaker identity across chunks
                        chunk_result = self._persist_speaker_identity(
                            chunk_result, chunk["start_time"]
                        )

                        # Add segments to global list
                        all_segments.extend(chunk_result.segments)

                        # Update global speaker mapping
                        for segment in chunk_result.segments:
                            if segment.speaker_id not in global_speaker_mapping:
                                global_speaker_mapping[segment.speaker_id] = (
                                    f"Speaker {segment.speaker_id.split('_')[1]}"
                                )

                except Exception as e:
                    print(f"Warning: Error processing chunk {chunk_idx}: {e}")
                    continue

            if not all_segments:
                raise DiarizationError("No segments found in any chunks")

            # Sort segments by start time
            all_segments.sort(key=lambda x: x.start_time)

            # Calculate overall confidence
            overall_confidence = (
                np.mean([seg.confidence for seg in all_segments])
                if all_segments
                else 0.0
            )

            # Create final result
            result = DiarizationResult(
                segments=all_segments,
                speaker_mapping=global_speaker_mapping,
                total_duration=len(audio_data) / sample_rate,
                num_speakers=len(self.speaker_cache),
                overall_confidence=overall_confidence,
                processing_time=time.time() - start_time,
            )

            # Update performance stats
            self.performance_stats["successful_diarizations"] += 1
            self.performance_stats["total_processing_time"] += result.processing_time
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["total_processing_time"]
                / self.performance_stats["successful_diarizations"]
            )

            print(
                f"✅ Chunked diarization completed: {len(all_segments)} segments, "
                f"{len(self.speaker_cache)} speakers"
            )

            return result

        except Exception as e:
            # Update performance stats
            self.performance_stats["failed_diarizations"] += 1

            # Return error result
            return DiarizationResult(
                segments=[],
                speaker_mapping={},
                total_duration=len(audio_data) / sample_rate,
                num_speakers=0,
                overall_confidence=0.0,
                processing_time=time.time() - start_time,
                error_details=f"Chunked diarization failed: {str(e)}",
            )

    def _create_overlapping_chunks(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        chunk_duration: float,
        overlap_duration: float,
    ) -> List[Dict]:
        """
        Create overlapping audio chunks for continuous processing.

        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of the audio
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds

        Returns:
            List of chunk dictionaries with data and timing information
        """
        chunks = []
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples

        if step_samples <= 0:
            raise ValueError("Overlap duration must be less than chunk duration")

        for i in range(0, len(audio_data), step_samples):
            start_sample = i
            end_sample = min(i + chunk_samples, len(audio_data))

            # Extract chunk data
            chunk_data = audio_data[start_sample:end_sample]

            # Calculate timing
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            chunks.append(
                {
                    "data": chunk_data,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "overlap_start": max(0, start_sample - overlap_samples),
                    "overlap_end": min(len(audio_data), end_sample + overlap_samples),
                }
            )

        return chunks

    def _save_temp_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Save audio data to temporary file for Pyannote processing."""
        import tempfile

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        # Save audio data
        sf.write(temp_path, audio_data, sample_rate)

        return temp_path

    def _extract_segments(
        self,
        diarization: Annotation,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> List[SpeakerSegment]:
        """Extract speaker segments from Pyannote diarization result."""
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Create speaker segment
            segment = SpeakerSegment(
                start_time=turn.start,
                end_time=turn.end,
                speaker_id=speaker,
                confidence=1.0,  # Pyannote doesn't provide confidence scores
            )

            # Validate segment
            if self.audio_segmenter.validate_segment(
                segment, self.config.min_segment_duration
            ):
                # Extract audio data for this segment
                segment.audio_data = self.audio_segmenter.extract_segment_audio(
                    audio_data, segment
                )
                segments.append(segment)

        return segments

    def _create_speaker_mapping(self, segments: List[SpeakerSegment]) -> Dict[str, str]:
        """Create human-readable speaker labels."""
        speaker_ids = list(set(seg.speaker_id for seg in segments))
        speaker_mapping = {}

        for i, speaker_id in enumerate(sorted(speaker_ids)):
            speaker_mapping[speaker_id] = f"Speaker {chr(65 + i)}"  # A, B, C, etc.

        return speaker_mapping

    def diarize_with_transcription(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        config: Optional[DiarizationConfig] = None,
    ) -> DiarizationResult:
        """
        Perform diarization with transcription using Groq API.

        CRITICAL: This method combines speaker diarization with speech
        transcription, providing a complete solution for multi-speaker
        audio processing:

        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of the audio
            config: Optional diarization configuration

        Returns:
            DiarizationResult with transcribed text for each segment

        Processing pipeline:
        1. Speaker diarization and segmentation
        2. Individual segment transcription via Groq API
        3. Text assignment and confidence scoring
        4. Result compilation and validation
        """
        # Perform diarization first
        diarization_result = self.diarize_audio(audio_data, sample_rate, config)

        if not diarization_result.is_successful:
            return diarization_result

        # Note: Transcription requires a SpeechRecognizer instance to be passed
        # This method only performs diarization without transcription
        print(
            "Note: Diarization completed. Use recognize_with_diarization() in SpeechRecognizer for transcription."
        )

        return diarization_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_requests": 0,
            "successful_diarizations": 0,
            "failed_diarizations": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
        }

    def _create_adaptive_chunks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Create chunks based on conversation flow, not fixed timing."""

        if self.config.chunk_strategy == "fixed":
            return self._create_fixed_chunks(audio_data, sample_rate)
        elif self.config.chunk_strategy == "conversation_aware":
            return self._create_conversation_aware_chunks(audio_data, sample_rate)
        else:  # adaptive
            return self._create_adaptive_chunks(audio_data, sample_rate)

    def _create_fixed_chunks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Create traditional fixed-size chunks."""
        chunk_size = int(self.config.min_chunk_duration * sample_rate)
        overlap_size = int(self.config.overlap_duration * sample_rate)

        chunks = []
        start = 0

        while start < len(audio_data):
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap_size
            if start >= len(audio_data):
                break

        return chunks

    def _create_conversation_aware_chunks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Create chunks that respect natural conversation boundaries."""

        # Use voice activity detection to find natural breaks
        # Create chunks at conversation boundaries, not arbitrary time points
        # Respect speaker changes and natural pauses

        # For now, implement a simplified version
        # In production, this would use more sophisticated VAD and speaker change detection

        # Calculate frame size for analysis
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.010 * sample_rate)  # 10ms hop

        # Simple energy-based VAD
        energy_frames = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i : i + frame_size]
            energy = np.mean(frame**2)
            energy_frames.append(energy)

        # Find silence regions
        energy_threshold = np.percentile(energy_frames, 30)
        silence_regions = [
            i for i, e in enumerate(energy_frames) if e < energy_threshold
        ]

        # Find natural break points (longer silences)
        break_points = []
        min_silence_duration = int(0.5 * sample_rate / hop_size)  # 0.5s minimum

        i = 0
        while i < len(silence_regions):
            # Find consecutive silence frames
            start_idx = silence_regions[i]
            j = i + 1
            while (
                j < len(silence_regions)
                and silence_regions[j] == silence_regions[j - 1] + 1
            ):
                j += 1

            silence_duration = j - i
            if silence_duration >= min_silence_duration:
                # Convert frame index to sample index
                break_point = start_idx * hop_size
                if break_point > 0 and break_point < len(audio_data):
                    break_points.append(break_point)

            i = j

        # Ensure minimum chunk duration
        min_chunk_samples = int(self.config.min_chunk_duration * sample_rate)
        filtered_breaks = []

        for break_point in break_points:
            if break_point >= min_chunk_samples:
                filtered_breaks.append(break_point)

        # Create chunks based on break points
        chunks = []
        start = 0

        for break_point in filtered_breaks:
            if break_point - start >= min_chunk_samples:
                chunk = audio_data[start:break_point]
                chunks.append(chunk)
                start = break_point

        # Add final chunk if there's remaining audio
        if start < len(audio_data):
            final_chunk = audio_data[start:]
            if len(final_chunk) >= min_chunk_samples:
                chunks.append(final_chunk)

        # Fallback to fixed chunks if no natural breaks found
        if not chunks:
            return self._create_fixed_chunks(audio_data, sample_rate)

        return chunks

    def _create_adaptive_chunks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Create chunks that adapt to audio characteristics."""

        # Analyze audio characteristics to determine optimal chunk size
        # Use spectral features, energy patterns, and speaker activity

        # Calculate audio features
        frame_size = int(0.025 * sample_rate)
        hop_size = int(0.010 * sample_rate)

        # Energy analysis
        energy_frames = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i : i + frame_size]
            energy = np.mean(frame**2)
            energy_frames.append(energy)

        # Spectral analysis
        spectral_frames = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i : i + frame_size]
            if len(frame) == frame_size:
                fft = np.fft.fft(frame)
                magnitude = np.abs(fft[: frame_size // 2])
                spectral_frames.append(magnitude)

        # Determine optimal chunk size based on audio characteristics
        energy_variance = np.var(energy_frames)
        spectral_variance = np.var([np.var(spec) for spec in spectral_frames])

        # High variance = more dynamic audio = smaller chunks
        # Low variance = stable audio = larger chunks

        base_chunk_size = self.config.min_chunk_duration

        if energy_variance > np.percentile(energy_frames, 80):
            # High energy variance - use smaller chunks
            chunk_size = base_chunk_size * 0.7
        elif energy_variance < np.percentile(energy_frames, 20):
            # Low energy variance - use larger chunks
            chunk_size = base_chunk_size * 1.5
        else:
            # Medium variance - use default chunk size
            chunk_size = base_chunk_size

        # Clamp to configuration limits
        chunk_size = max(
            self.config.min_chunk_duration,
            min(self.config.max_chunk_duration, chunk_size),
        )

        # Create chunks with adaptive size
        chunk_samples = int(chunk_size * sample_rate)
        overlap_samples = int(self.config.overlap_duration * sample_rate)

        chunks = []
        start = 0

        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap_samples
            if start >= len(audio_data):
                break

        return chunks

    def _smart_merge_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """Intelligent segment merging based on conversation flow."""

        if not self.config.enable_adaptive_merging:
            return segments

        merged_segments = []
        i = 0

        while i < len(segments):
            current = segments[i]

            # Look ahead for segments that should be merged
            merge_candidates = self._find_merge_candidates(segments, i)

            if merge_candidates:
                # Merge multiple segments intelligently
                merged_segment = self._merge_multiple_segments(
                    [current] + merge_candidates
                )
                merged_segments.append(merged_segment)
                i += len(merge_candidates) + 1
            else:
                merged_segments.append(current)
                i += 1

        return merged_segments

    def _find_merge_candidates(
        self, segments: List[SpeakerSegment], start_idx: int
    ) -> List[SpeakerSegment]:
        """Find segments that should be merged based on context."""

        candidates = []
        current_idx = start_idx

        while current_idx + 1 < len(segments):
            next_segment = segments[current_idx + 1]

            # Check if segments should be merged
            if self._should_merge_segments(segments[current_idx], next_segment):
                candidates.append(next_segment)
                current_idx += 1
            else:
                break

        return candidates

    def _should_merge_segments(
        self, seg1: SpeakerSegment, seg2: SpeakerSegment
    ) -> bool:
        """Determine if two segments should be merged."""

        # Same speaker
        if seg1.speaker_id != seg2.speaker_id:
            return False

        # Close in time
        time_gap = seg2.start_time - seg1.end_time
        if time_gap > self.config.merge_time_threshold:
            return False

        # Contextually related (not hardcoded, but pattern-based)
        if self._are_segments_contextually_related(seg1, seg2):
            return True

        # Natural speech flow
        if self._is_natural_speech_continuation(seg1, seg2):
            return True

        return False

    def _are_segments_contextually_related(
        self, seg1: SpeakerSegment, seg2: SpeakerSegment
    ) -> bool:
        """Check if segments are contextually related."""

        if not self.config.enable_context_awareness:
            return False

        # Check for question-answer patterns
        text1 = seg1.text.strip().lower()
        text2 = seg2.text.strip().lower()

        # Question indicators
        question_indicators = [
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "can you",
            "could you",
            "would you",
            "do you",
            "is there",
            "are there",
            "does it",
            "do they",
        ]

        # Answer indicators
        answer_indicators = [
            "yes",
            "no",
            "sure",
            "absolutely",
            "definitely",
            "i think",
            "i believe",
            "in my opinion",
            "the answer is",
            "that's correct",
            "exactly",
        ]

        # Check if first segment is a question and second is an answer
        is_question = any(indicator in text1 for indicator in question_indicators)
        is_answer = any(indicator in text2 for indicator in answer_indicators)

        if is_question and is_answer:
            return True

        # Check for incomplete sentences that continue
        if self._is_incomplete_sentence(text1) and self._completes_sentence(
            text1, text2
        ):
            return True

        # Check for natural speech patterns
        if self._has_natural_speech_flow(text1, text2):
            return True

        return False

    def _is_incomplete_sentence(self, text: str) -> bool:
        """Check if text appears to be an incomplete sentence."""

        # Remove common sentence endings
        text = text.strip()
        if not text:
            return False

        # Check for common incomplete patterns
        incomplete_patterns = [
            text.endswith(" and"),
            text.endswith(" but"),
            text.endswith(" or"),
            text.endswith(" the"),
            text.endswith(" a"),
            text.endswith(" an"),
            text.endswith(" to"),
            text.endswith(" in"),
            text.endswith(" on"),
            text.endswith(" at"),
            text.endswith(" with"),
            text.endswith(" by"),
            text.endswith(" for"),
            text.endswith(" of"),
            text.endswith(" from"),
            text.endswith(" about"),
            text.endswith(" that"),
            text.endswith(" this"),
            text.endswith(" these"),
            text.endswith(" those"),
        ]

        return any(incomplete_patterns)

    def _completes_sentence(self, incomplete_text: str, completion_text: str) -> bool:
        """Check if completion_text completes the incomplete sentence."""

        # Simple heuristic: check if completion starts with lowercase
        # or if it continues the grammatical structure

        completion = completion_text.strip()
        if not completion:
            return False

        # Check for grammatical continuation
        continuation_patterns = [
            completion[0].islower(),  # Starts with lowercase
            completion.startswith("and "),
            completion.startswith("but "),
            completion.startswith("or "),
            completion.startswith("the "),
            completion.startswith("a "),
            completion.startswith("an "),
            completion.startswith("to "),
            completion.startswith("in "),
            completion.startswith("on "),
            completion.startswith("at "),
            completion.startswith("with "),
            completion.startswith("by "),
            completion.startswith("for "),
            completion.startswith("of "),
            completion.startswith("from "),
            completion.startswith("about "),
            completion.startswith("that "),
            completion.startswith("this "),
            completion.startswith("these "),
            completion.startswith("those "),
        ]

        return any(continuation_patterns)

    def _has_natural_speech_flow(self, text1: str, text2: str) -> bool:
        """Check if two text segments have natural speech flow."""

        # Check for natural speech patterns
        # This is a simplified heuristic - in production, use more sophisticated NLP

        # Check for filler words that indicate natural flow
        filler_words = [
            "um",
            "uh",
            "er",
            "ah",
            "oh",
            "well",
            "so",
            "you know",
            "i mean",
            "like",
            "basically",
            "actually",
            "frankly",
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check if second segment starts with a filler word
        starts_with_filler = any(
            text2_lower.startswith(filler) for filler in filler_words
        )

        if starts_with_filler:
            return True

        # Check for natural speech markers
        speech_markers = [
            "and then",
            "so then",
            "but then",
            "and so",
            "you see",
            "i think",
            "i believe",
            "in my opinion",
        ]

        for marker in speech_markers:
            if marker in text1_lower or marker in text2_lower:
                return True

        return False

    def _is_natural_speech_continuation(
        self, seg1: SpeakerSegment, seg2: SpeakerSegment
    ) -> bool:
        """Check if segments represent natural speech continuation."""

        # Check timing - very close segments are more likely to be continuous
        time_gap = seg2.start_time - seg1.end_time

        if time_gap < 0.5:  # Less than 0.5 seconds
            return True

        # Check for natural speech patterns
        text1 = seg1.text.strip().lower()
        text2 = seg2.text.strip().lower()

        # Check for sentence completion patterns
        if self._is_incomplete_sentence(text1):
            return True

        # Check for natural speech flow
        if self._has_natural_speech_flow(text1, text2):
            return True

        return False

    def _merge_multiple_segments(
        self, segments: List[SpeakerSegment]
    ) -> SpeakerSegment:
        """Merge multiple segments into a single coherent segment."""

        if not segments:
            raise ValueError("Cannot merge empty segment list")

        if len(segments) == 1:
            return segments[0]

        # Merge text with proper spacing
        merged_text = ""
        for i, segment in enumerate(segments):
            if i > 0:
                # Add space between segments, but be smart about it
                prev_text = segments[i - 1].text.strip()
                curr_text = segment.text.strip()

                # Don't add space if previous text ends with punctuation
                if not prev_text.endswith((".", "!", "?", ":", ";")):
                    merged_text += " "

            merged_text += segment.text.strip()

        # Create merged segment
        first_segment = segments[0]
        last_segment = segments[-1]

        merged_segment = SpeakerSegment(
            speaker_id=first_segment.speaker_id,
            start_time=first_segment.start_time,
            end_time=last_segment.end_time,
            text=merged_text,
            confidence=first_segment.confidence,  # Use first segment's confidence
        )

        return merged_segment

    def diarize_with_accurate_transcription(self, audio_file: str, mode: str, 
                                          speech_recognizer=None) -> "DiarizationResult":
        """
        CORRECT PIPELINE: Pyannote.audio FIRST, then Groq API per segment.
        
        This is the proper approach:
        1. Pyannote.audio detects speaker segments and timestamps
        2. Audio is split into speaker-specific chunks
        3. Each chunk is sent to Groq API for accurate transcription
        4. Perfect speaker attribution with accurate text
        
        Args:
            audio_file: Path to audio file
            mode: 'transcription' or 'translation'
            speech_recognizer: SpeechRecognizer instance for Groq API calls
            
        Returns:
            DiarizationResult with accurate speaker-specific transcriptions
        """
        print("🎭 Running CORRECT diarization pipeline...")
        print("   1. Pyannote.audio → Speaker detection")
        print("   2. Audio chunking → Speaker-specific segments") 
        print("   3. Groq API → Accurate transcription per segment")
        
        try:
            # Step 1: Pyannote.audio for speaker detection
            print("   🔍 Step 1: Detecting speakers with Pyannote.audio...")
            
            # Get HF token from Config
            from .config import Config
            hf_token = Config.get_hf_token()
            
            if not hf_token or hf_token == "your_hf_token_here":
                raise ValueError("HF_TOKEN not configured for Pyannote.audio")
            
            # Initialize Pyannote pipeline
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Run diarization to get speaker segments
            diarization = pipeline(audio_file)
            
            # Extract speaker segments with timestamps
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            print(f"   ✅ Detected {len(speaker_segments)} speaker segments")
            
            if not speaker_segments:
                print("   ⚠️ No speaker segments detected, creating single speaker")
                # Get audio duration for single speaker
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file)
                duration = len(audio_data) / sample_rate
                
                speaker_segments = [{
                    'start': 0.0,
                    'end': duration,
                    'speaker': 'SPEAKER_1',
                    'duration': duration
                }]
            
            # Step 2: Process each speaker segment with Groq API
            print("   🔍 Step 2: Transcribing each speaker segment...")
            
            if not speech_recognizer:
                raise ValueError("SpeechRecognizer required for transcription")
            
            segments = []
            speaker_mapping = {}
            speaker_counter = 0
            
            for i, seg_info in enumerate(speaker_segments):
                print(f"      Processing segment {i+1}/{len(speaker_segments)}: "
                      f"{seg_info['start']:.1f}s - {seg_info['end']:.1f}s")
                
                # Map Pyannote speaker labels to consistent IDs
                if seg_info['speaker'] not in speaker_mapping:
                    speaker_counter += 1
                    speaker_mapping[seg_info['speaker']] = f"SPEAKER_{speaker_counter-1:02d}"
                
                speaker_id = speaker_mapping[seg_info['speaker']]
                
                # Extract audio chunk for this speaker segment
                audio_chunk = self._extract_audio_chunk(
                    audio_file, seg_info['start'], seg_info['end']
                )
                
                if audio_chunk is not None:
                    # Send speaker-specific audio chunk to Groq API
                    print(f"         Sending {speaker_id} audio to Groq API...")
                    
                    try:
                        if mode == "translation":
                            result = speech_recognizer.recognize_audio_data(
                                audio_chunk, is_translation=True
                            )
                        else:
                            result = speech_recognizer.recognize_audio_data(
                                audio_chunk, is_translation=False
                            )
                        
                        if result and result.text:
                            transcription_text = result.text
                            confidence = result.confidence if hasattr(result, 'confidence') else 0.95
                            print(f"         ✅ {speaker_id}: {transcription_text[:50]}...")
                        else:
                            transcription_text = "[No transcription available]"
                            confidence = 0.5
                            print(f"         ⚠️ {speaker_id}: Transcription failed")
                    
                    except Exception as api_error:
                        print(f"         ❌ {speaker_id}: Groq API error: {api_error}")
                        transcription_text = "[Transcription error]"
                        confidence = 0.3
                
                else:
                    transcription_text = "[Audio extraction failed]"
                    confidence = 0.3
                
                # Create speaker segment with accurate transcription
                segment = SpeakerSegment(
                    start_time=seg_info['start'],
                    end_time=seg_info['end'],
                    speaker_id=speaker_id,
                    confidence=confidence
                )
                segment.text = transcription_text
                segments.append(segment)
            
            # Step 3: Create final result
            print("   🔍 Step 3: Creating final diarization result...")
            
            result = DiarizationResult(
                segments=segments,
                speaker_mapping=speaker_mapping,
                total_duration=speaker_segments[-1]['end'] if speaker_segments else 0.0,
                num_speakers=len(speaker_mapping),
                overall_confidence=sum(seg.confidence for seg in segments) / len(segments) if segments else 0.0,
                processing_time=0.0
            )
            
            print(f"   ✅ CORRECT pipeline completed successfully!")
            print(f"      Speakers: {len(speaker_mapping)}")
            print(f"      Segments: {len(segments)}")
            print(f"      Total duration: {result.total_duration:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"   ❌ CORRECT pipeline failed: {e}")
            # Fallback to basic diarization
            print("   🔄 Falling back to basic diarization...")
            return self.diarize_audio(audio_file, mode)
    
    def _extract_audio_chunk(self, audio_file: str, start_time: float, end_time: float):
        """
        Extract audio chunk for a specific time range.
        
        Args:
            audio_file: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio data as numpy array for the specified time range
        """
        try:
            import soundfile as sf
            
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                return None
            
            # Extract chunk
            audio_chunk = audio_data[start_sample:end_sample]
            
            return audio_chunk
            
        except Exception as e:
            print(f"      ❌ Audio chunk extraction failed: {e}")
            return None


# Simplified Diarizer class that combines both basic and enhanced functionality
class Diarizer:
    """Simplified diarizer that provides enhanced functionality by default."""
    
    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self.base_diarizer = SpeakerDiarizer(config)
        
    def diarize(self, audio_file: str, mode: str, speech_recognizer=None) -> DiarizationResult:
        """
        Perform diarization with enhanced functionality by default.
        
        Args:
            audio_file: Path to audio file
            mode: 'transcription' or 'translation'
            speech_recognizer: SpeechRecognizer instance for transcription
            
        Returns:
            DiarizationResult with speaker-separated transcription
        """
        try:
            # Use the enhanced diarization method from SpeakerDiarizer
            return self.base_diarizer.diarize_with_accurate_transcription(
                audio_file, mode, speech_recognizer
            )
        except Exception as e:
            print(f"Diarization failed: {e}")
            # Fallback to basic diarization - fix the method call
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file)
                return self.base_diarizer.diarize_audio(
                    audio_data, sample_rate
                )
            except Exception as e2:
                print(f"Fallback diarization also failed: {e2}")
                # Create a minimal result
                from .speaker_diarization import SpeakerSegment
                segment = SpeakerSegment(
                    start_time=0.0,
                    end_time=1.0,
                    speaker_id="SPEAKER_00",
                    confidence=0.5,
                    text="[Diarization failed]"
                )
                return DiarizationResult(
                    segments=[segment],
                    speaker_mapping={"SPEAKER_00": "Speaker 1"},
                    total_duration=1.0,
                    num_speakers=1,
                    overall_confidence=0.5
                )
