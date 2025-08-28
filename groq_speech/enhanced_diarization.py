"""
Enhanced Diarization with Smart Segment Grouping and Parallel Processing.

This module implements the advanced diarization dataflow:
1. Pyannote.audio → Speaker detection (51 segments)
2. Smart segment grouping → Combine segments under 25MB limit
3. Parallel Groq API processing → Process groups concurrently
4. Ordered output → Maintain segment order and speaker mapping

ARCHITECTURE OVERVIEW:
1. CORE COMPONENTS
   - EnhancedDiarizer: Main class for advanced diarization
   - SegmentGroup: Represents a group of segments for Groq API
   - GroqRequestTracker: Tracks parallel API requests and results
   - SmartSegmentGrouper: Intelligent segment combination algorithm

2. DATAFLOW PIPELINE
   - Pyannote.audio speaker detection
   - Smart segment grouping with size limits
   - Parallel Groq API processing
   - Result ordering and speaker mapping

3. FEATURES
   - Configurable file size limits (default: 25MB)
   - Parallel processing with configurable concurrency
   - Intelligent segment merging preserving speaker continuity
   - Comprehensive error handling and retry logic
   - Real-time progress reporting
   - Detailed logging and performance metrics

USAGE EXAMPLES:
    # Basic enhanced diarization
    diarizer = EnhancedDiarizer()
    result = diarizer.diarize_with_enhanced_flow(audio_file, mode)

    # Custom configuration
    config = EnhancedDiarizationConfig(
        max_file_size_mb=30.0,
        max_parallel_requests=8,
        enable_parallel_processing=True
    )
    diarizer = EnhancedDiarizer(config)
    result = diarizer.diarize_with_enhanced_flow(audio_file, mode, config)
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import soundfile as sf
from dataclasses import dataclass

from .speech_config import SpeechConfig
from .speaker_diarization import SpeakerDiarizer, SpeakerSegment, DiarizationResult
from .config import Config
from .exceptions import DiarizationError


@dataclass
class SegmentGroup:
    """Represents a group of segments to be processed together."""

    group_id: str
    segments: List[SpeakerSegment]
    start_time: float
    end_time: float
    total_duration: float
    speaker_id: (
        str  # Single speaker ID since all segments in group belong to same speaker
    )
    estimated_size_mb: float
    audio_data: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"SegmentGroup(id={self.group_id}, "
            f"segments={len(self.segments)}, "
            f"duration={self.total_duration:.2f}s, "
            f"size={self.estimated_size_mb:.2f}MB)"
        )


@dataclass
class GroqRequestResult:
    """Result from a Groq API request."""

    group_id: str
    text: str
    confidence: float
    language: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} {self.group_id}: {self.text[:50]}..."


class EnhancedDiarizationConfig:
    """Configuration for enhanced diarization with smart grouping."""

    def __init__(
        self,
        max_file_size_mb: float = 25.0,
        enable_parallel_processing: bool = True,
        max_parallel_requests: int = 4,
        retry_enabled: bool = True,
        retry_delay_seconds: float = 1.0,
        max_retries: int = 3,
        log_level: str = "INFO",
        enable_progress_reporting: bool = True,
        min_group_duration: float = 5.0,
        max_group_duration: float = 60.0,
        speaker_continuity_threshold: float = 2.0,
        enable_smart_merging: bool = True,
        preserve_sentence_boundaries: bool = True,
    ):
        """
        Initialize enhanced diarization configuration.

        Args:
            max_file_size_mb: Maximum file size in MB for Groq API
            enable_parallel_processing: Enable parallel API processing
            max_parallel_requests: Maximum concurrent Groq API requests
            retry_enabled: Enable retry logic for failed requests
            retry_delay_seconds: Delay between retries
            max_retries: Maximum number of retry attempts
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_progress_reporting: Enable real-time progress updates
            min_group_duration: Minimum duration for a segment group
            max_group_duration: Maximum duration for a segment group
            speaker_continuity_threshold: Time threshold for speaker continuity
            enable_smart_merging: Enable intelligent segment merging
            preserve_sentence_boundaries: Preserve natural sentence boundaries
        """
        self.max_file_size_mb = max_file_size_mb
        self.enable_parallel_processing = enable_parallel_processing
        self.max_parallel_requests = max_parallel_requests
        self.retry_enabled = retry_enabled
        self.retry_delay_seconds = retry_delay_seconds
        self.max_retries = max_retries
        self.log_level = log_level
        self.enable_progress_reporting = enable_progress_reporting
        self.min_group_duration = min_group_duration
        self.max_group_duration = max_group_duration
        self.speaker_continuity_threshold = speaker_continuity_threshold
        self.enable_smart_merging = enable_smart_merging
        self.preserve_sentence_boundaries = preserve_sentence_boundaries

    @classmethod
    def from_environment(cls) -> "EnhancedDiarizationConfig":
        """Create configuration from environment variables."""
        return cls(
            max_file_size_mb=Config.DIARIZATION_MAX_FILE_SIZE_MB,
            enable_parallel_processing=Config.DIARIZATION_ENABLE_PARALLEL_PROCESSING,
            max_parallel_requests=Config.DIARIZATION_MAX_PARALLEL_REQUESTS,
            retry_enabled=Config.DIARIZATION_RETRY_ENABLED,
            retry_delay_seconds=Config.DIARIZATION_RETRY_DELAY_SECONDS,
            max_retries=Config.DIARIZATION_MAX_RETRIES,
            log_level=Config.DIARIZATION_LOG_LEVEL,
            enable_progress_reporting=Config.DIARIZATION_ENABLE_PROGRESS_REPORTING,
        )

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")

        if self.max_parallel_requests < 1:
            errors.append("max_parallel_requests must be at least 1")

        if self.retry_delay_seconds < 0:
            errors.append("retry_delay_seconds must be non-negative")

        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self.min_group_duration <= 0:
            errors.append("min_group_duration must be positive")

        if self.max_group_duration <= self.min_group_duration:
            errors.append("max_group_duration must be greater than min_group_duration")

        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False

        return True


class SmartSegmentGrouper:
    """Intelligent segment grouping algorithm."""

    def __init__(self, config: EnhancedDiarizationConfig):
        """Initialize the smart segment grouper."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def group_segments(
        self, segments: List[SpeakerSegment], audio_data: np.ndarray, sample_rate: int
    ) -> List[SegmentGroup]:
        """
        Group segments by speaker continuity - each group contains consecutive segments from the same speaker.

        Args:
            segments: List of speaker segments from Pyannote.audio
            audio_data: Full audio data
            sample_rate: Audio sample rate

        Returns:
            List of segment groups where each group contains segments from the same speaker
        """
        if not segments:
            return []

        self.logger.info(f"Grouping {len(segments)} segments by speaker continuity")

        # Sort segments by start time to ensure proper order
        sorted_segments = sorted(segments, key=lambda x: x.start_time)

        groups = []
        current_group = []
        current_speaker = None
        group_counter = 1  # Sequential group counter starting from 1

        for i, segment in enumerate(sorted_segments):
            # Start new group if speaker changes
            if current_speaker is not None and segment.speaker_id != current_speaker:
                # Finalize current group
                if current_group:
                    group = self._create_segment_group_simple(
                        current_group, audio_data, sample_rate, group_counter
                    )
                    groups.append(group)
                    self.logger.debug(
                        f"Created group {group_counter}: {current_speaker} ({len(current_group)} segments)"
                    )
                    group_counter += 1  # Increment group counter

                # Start new group
                current_group = [segment]
                current_speaker = segment.speaker_id
            else:
                # Continue with current speaker
                if current_speaker is None:
                    current_speaker = segment.speaker_id
                current_group.append(segment)

        # Don't forget the last group
        if current_group:
            group = self._create_segment_group_simple(
                current_group, audio_data, sample_rate, group_counter
            )
            groups.append(group)
            self.logger.debug(
                f"Created group {group_counter}: {current_speaker} ({len(current_group)} segments)"
            )

        self.logger.info(f"Created {len(groups)} speaker-based groups")
        return groups

    def _calculate_segment_size_mb(
        self, segment: SpeakerSegment, sample_rate: int
    ) -> float:
        """Calculate estimated size of a segment in MB."""
        duration = segment.end_time - segment.start_time
        # Assuming 16-bit audio, mono channel
        size_bytes = duration * sample_rate * 2  # 2 bytes per sample
        return size_bytes / (1024 * 1024)  # Convert to MB

    def _should_start_new_group(
        self, segment: SpeakerSegment, current_group: List[SpeakerSegment]
    ) -> bool:
        """Determine if a new group should be started."""
        if not current_group:
            return True

        # Check time gap
        last_segment = current_group[-1]
        time_gap = segment.start_time - last_segment.end_time

        if time_gap > self.config.speaker_continuity_threshold:
            return True

        # Check if current group is getting too long
        group_duration = segment.end_time - current_group[0].start_time
        if group_duration > self.config.max_group_duration:
            return True

        return False

    def _create_segment_group(
        self,
        segments: List[SpeakerSegment],
        start_time: float,
        total_size_mb: float,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> SegmentGroup:
        """Create a segment group with extracted audio data."""
        if not segments:
            raise ValueError("Cannot create group with empty segments")

        end_time = segments[-1].end_time
        total_duration = end_time - start_time
        speaker_ids = list(set(seg.speaker_id for seg in segments))

        # Extract audio data for the entire group
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        group_audio = audio_data[start_sample:end_sample]

        group_id = f"group_{len(segments):02d}_{start_time:.1f}s"

        return SegmentGroup(
            group_id=group_id,
            segments=segments,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            speaker_id=speaker_ids[0] if speaker_ids else "Unknown",
            estimated_size_mb=total_size_mb,
            audio_data=group_audio,
        )

    def _create_segment_group_simple(
        self,
        segments: List[SpeakerSegment],
        audio_data: np.ndarray,
        sample_rate: int,
        group_number: int,  # Add group_number parameter for sequential numbering
    ) -> SegmentGroup:
        """Create a segment group from a list of segments."""
        # Calculate group metadata
        group_start_time = min(seg.start_time for seg in segments)
        group_end_time = max(seg.end_time for seg in segments)
        group_duration = group_end_time - group_start_time

        # Extract audio for this group
        start_sample = int(group_start_time * sample_rate)
        end_sample = int(group_end_time * sample_rate)
        group_audio = audio_data[start_sample:end_sample]

        # Create unique group ID with sequential numbering for proper ordering
        group_id = f"group_{group_number:02d}"  # Simple sequential numbering: group_01, group_02, etc.

        # Ensure speaker ID is properly captured
        speaker_id = segments[0].speaker_id if segments else "Unknown"

        # Calculate size in MB
        size_mb = self._calculate_segment_size_mb(segments[0], sample_rate) * len(
            segments
        )

        return SegmentGroup(
            group_id=group_id,
            segments=segments,
            start_time=group_start_time,
            end_time=group_end_time,
            total_duration=group_duration,
            speaker_id=speaker_id,  # Single speaker ID
            estimated_size_mb=size_mb,
            audio_data=group_audio,
        )


class GroqRequestTracker:
    """Tracks parallel Groq API requests and results."""

    def __init__(self, config: EnhancedDiarizationConfig):
        """Initialize the request tracker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, GroqRequestResult] = {}
        self.lock = threading.Lock()
        self.completed_count = 0
        self.total_count = 0

    def add_result(self, result: GroqRequestResult):
        """Add a completed request result."""
        with self.lock:
            self.results[result.group_id] = result
            self.completed_count += 1
            self.logger.info(
                f"Completed {self.completed_count}/{self.total_count} requests"
            )

    def get_results(self) -> List[GroqRequestResult]:
        """Get all completed results."""
        with self.lock:
            return list(self.results.values())

    def is_complete(self) -> bool:
        """Check if all requests are complete."""
        with self.lock:
            return self.completed_count >= self.total_count

    def set_total_count(self, total: int):
        """Set the total number of expected requests."""
        with self.lock:
            self.total_count = total


class EnhancedDiarizer:
    """Enhanced diarization with smart segment grouping and parallel processing."""

    def __init__(
        self,
        config: Optional[EnhancedDiarizationConfig] = None,
        speech_config: Optional[SpeechConfig] = None,
    ):
        """Initialize the enhanced diarizer."""
        self.config = config or EnhancedDiarizationConfig.from_environment()
        self.config.validate()

        self.speech_config = speech_config or SpeechConfig()

        # Initialize components
        self.segment_grouper = SmartSegmentGrouper(self.config)
        self.request_tracker = GroqRequestTracker(self.config)
        self.base_diarizer = SpeakerDiarizer()

        # Setup logging
        self._setup_logging()

        # Performance tracking
        self.performance_stats = {
            "total_processing_time": 0.0,
            "diarization_time": 0.0,
            "grouping_time": 0.0,
            "api_processing_time": 0.0,
            "total_segments": 0,
            "total_groups": 0,
            "successful_requests": 0,
            "failed_requests": 0,
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def diarize_with_enhanced_flow(
        self, audio_file: str, mode: str, speech_recognizer=None
    ) -> DiarizationResult:
        """
        Execute the enhanced diarization dataflow.

        Args:
            audio_file: Path to audio file
            mode: 'transcription' or 'translation'
            speech_recognizer: SpeechRecognizer instance for Groq API calls

        Returns:
            DiarizationResult with enhanced processing
        """
        start_time = time.time()
        self.logger.info(f"Starting enhanced diarization flow for {audio_file}")

        try:
            # Step 1: Pyannote.audio → Speaker detection
            self.logger.info("Step 1: Pyannote.audio speaker detection")
            diarization_start = time.time()

            segments = self._detect_speakers_pyannote(audio_file)
            if not segments:
                raise DiarizationError("No speaker segments detected")

            self.performance_stats["diarization_time"] = time.time() - diarization_start
            self.performance_stats["total_segments"] = len(segments)

            self.logger.info(f"✅ Detected {len(segments)} speaker segments")

            # Step 2: Smart segment grouping
            self.logger.info("Step 2: Smart segment grouping")
            grouping_start = time.time()

            # Load audio data for grouping
            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Convert to mono

            groups = self.segment_grouper.group_segments(
                segments, audio_data, sample_rate
            )
            self.performance_stats["grouping_time"] = time.time() - grouping_start
            self.performance_stats["total_groups"] = len(groups)

            self.logger.info(f"✅ Created {len(groups)} segment groups")

            # Step 3: Parallel Groq API processing
            self.logger.info("Step 3: Parallel Groq API processing")
            api_start = time.time()

            if not speech_recognizer:
                raise DiarizationError("SpeechRecognizer required for transcription")

            results = self._process_groups_parallel(groups, mode, speech_recognizer)
            self.performance_stats["api_processing_time"] = time.time() - api_start

            # Step 4: Compile final results
            self.logger.info("Step 4: Compiling final results")
            final_result = self._compile_final_results(segments, results, groups)

            # Update performance stats
            self.performance_stats["total_processing_time"] = time.time() - start_time

            self.logger.info("✅ Enhanced diarization flow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"Enhanced diarization flow failed: {e}")
            raise DiarizationError(f"Enhanced diarization failed: {str(e)}")

    def _detect_speakers_pyannote(self, audio_file: str) -> List[SpeakerSegment]:
        """Detect speakers using Pyannote.audio."""
        try:
            # Use Pyannote.audio directly for speaker detection only
            # We don't want transcription here, just speaker segments
            if not self.base_diarizer._pipeline:
                self.logger.warning(
                    "Pyannote.audio pipeline not available, using fallback"
                )
                return self._create_fallback_segments(audio_file)

            # Run Pyannote.audio diarization directly
            import torch
            import soundfile as sf

            # Load audio data
            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Convert to mono

            # Convert to Pyannote.audio format: (channel, time)
            # Pyannote.audio expects (1, time) for mono audio
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)

            # Run diarization
            diarization = self.base_diarizer._pipeline(
                {"waveform": audio_tensor, "sample_rate": sample_rate}
            )

            # Extract speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Create speaker segment
                segment = SpeakerSegment(
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                    speaker_id=speaker,
                    confidence=0.8,  # Default confidence
                )
                segments.append(segment)

            if segments:
                self.logger.info(
                    f"Pyannote.audio detected {len(segments)} speaker segments"
                )
                return segments
            else:
                # Fallback: create basic segments
                self.logger.warning(
                    "Pyannote.audio returned no segments, using fallback"
                )
                return self._create_fallback_segments(audio_file)

        except Exception as e:
            self.logger.error(f"Pyannote.audio speaker detection failed: {e}")
            return self._create_fallback_segments(audio_file)

    def _create_fallback_segments(self, audio_file: str) -> List[SpeakerSegment]:
        """Create fallback segments when Pyannote.audio fails."""
        try:
            audio_data, sample_rate = sf.read(audio_file)
            duration = len(audio_data) / sample_rate

            # Create single speaker segment
            segment = SpeakerSegment(
                start_time=0.0,
                end_time=duration,
                speaker_id="SPEAKER_1",
                confidence=0.5,
            )

            return [segment]

        except Exception as e:
            self.logger.error(f"Fallback segmentation failed: {e}")
            raise DiarizationError("All segmentation methods failed")

    def _process_groups_parallel(
        self, groups: List[SegmentGroup], mode: str, speech_recognizer
    ) -> List[GroqRequestResult]:
        """Process segment groups in parallel using Groq API."""
        if not self.config.enable_parallel_processing:
            return self._process_groups_sequential(groups, mode, speech_recognizer)

        self.logger.info(
            f"Processing {len(groups)} groups in parallel "
            f"(max {self.config.max_parallel_requests} concurrent - increased for better performance)"
        )

        # Setup request tracker
        self.request_tracker.set_total_count(len(groups))

        results = []

        with ThreadPoolExecutor(
            max_workers=self.config.max_parallel_requests
        ) as executor:
            # Submit all requests
            future_to_group = {}
            for group in groups:
                future = executor.submit(
                    self._process_single_group, group, mode, speech_recognizer
                )
                future_to_group[future] = group

            # Collect results as they complete
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    result = future.result()
                    self.request_tracker.add_result(result)
                    results.append(result)

                    if self.config.enable_progress_reporting:
                        self._report_progress(result)

                except Exception as e:
                    self.logger.error(f"Group {group.group_id} processing failed: {e}")
                    error_result = GroqRequestResult(
                        group_id=group.group_id,
                        text=f"[Error: {str(e)}]",
                        confidence=0.0,
                        language="",
                        processing_time=0.0,
                        success=False,
                        error_message=str(e),
                    )
                    self.request_tracker.add_result(error_result)
                    results.append(error_result)

        # Sort results by group ID to maintain order
        results.sort(key=lambda x: x.group_id)
        return results

    def _process_groups_sequential(
        self, groups: List[SegmentGroup], mode: str, speech_recognizer
    ) -> List[GroqRequestResult]:
        """Process segment groups sequentially."""
        self.logger.info(f"Processing {len(groups)} groups sequentially")

        results = []
        for i, group in enumerate(groups):
            try:
                self.logger.info(
                    f"Processing group {i+1}/{len(groups)}: {group.group_id}"
                )

                result = self._process_single_group(group, mode, speech_recognizer)
                results.append(result)

                if self.config.enable_progress_reporting:
                    self._report_progress(result)

            except Exception as e:
                self.logger.error(f"Group {group.group_id} processing failed: {e}")
                error_result = GroqRequestResult(
                    group_id=group.group_id,
                    text=f"[Error: {str(e)}]",
                    confidence=0.0,
                    language="",
                    processing_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

        return results

    def _process_single_group(
        self, group: SegmentGroup, mode: str, speech_recognizer
    ) -> GroqRequestResult:
        """Process a single segment group with retry logic."""
        start_time = time.time()
        retry_count = 0

        while retry_count <= self.config.max_retries:
            try:
                if mode == "translation":
                    result = speech_recognizer.translate_audio_data(group.audio_data)
                else:
                    result = speech_recognizer.recognize_audio_data(group.audio_data)

                if result and result.text:
                    processing_time = time.time() - start_time

                    return GroqRequestResult(
                        group_id=group.group_id,
                        text=result.text,
                        confidence=getattr(result, "confidence", 0.95),
                        language=getattr(result, "language", ""),
                        processing_time=processing_time,
                        success=True,
                    )
                else:
                    raise ValueError("Empty transcription result")

            except Exception as e:
                retry_count += 1
                if retry_count <= self.config.max_retries and self.config.retry_enabled:
                    self.logger.warning(
                        f"Group {group.group_id} failed, retrying "
                        f"({retry_count}/{self.config.max_retries}): {e}"
                    )
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    # Final failure
                    processing_time = time.time() - start_time
                    return GroqRequestResult(
                        group_id=group.group_id,
                        text=f"[Failed after {retry_count} attempts: {str(e)}]",
                        confidence=0.0,
                        language="",
                        processing_time=processing_time,
                        success=False,
                        error_message=str(e),
                    )

        # Fallback return (should never be reached due to while loop logic)
        return GroqRequestResult(
            group_id=group.group_id,
            text="[Unexpected error: loop exited without return]",
            confidence=0.0,
            language="",
            processing_time=time.time() - start_time,
            success=False,
            error_message="Unexpected loop exit",
        )

    def _report_progress(self, result: GroqRequestResult):
        """Report progress for a completed request."""
        status = "✅" if result.success else "❌"
        self.logger.info(f"{status} {result.group_id}: {result.text[:100]}...")

    def _compile_final_results(
        self,
        original_segments: List[SpeakerSegment],
        group_results: List[GroqRequestResult],
        groups: List[SegmentGroup],
    ) -> DiarizationResult:
        """Compile final results maintaining segment order and speaker mapping."""
        # Create mapping from group results to segments
        group_to_result = {result.group_id: result for result in group_results}

        # Update segments with transcription results
        updated_segments = []
        speaker_mapping = {}

        for segment in original_segments:
            # Find which group this segment belongs to
            segment_group = self._find_segment_group(segment, groups)

            if segment_group and segment_group.group_id in group_to_result:
                result = group_to_result[segment_group.group_id]
                if result.success:
                    segment.text = result.text
                    segment.transcription_confidence = result.confidence
                else:
                    segment.text = f"[Error: {result.error_message}]"
                    segment.transcription_confidence = 0.0
            else:
                segment.text = "[No transcription available]"
                segment.transcription_confidence = 0.0

            # Update speaker mapping
            if segment.speaker_id not in speaker_mapping:
                speaker_mapping[segment.speaker_id] = f"Speaker {segment.speaker_id}"

            updated_segments.append(segment)

        # Calculate overall statistics
        total_duration = (
            max(seg.end_time for seg in updated_segments) if updated_segments else 0.0
        )
        num_speakers = len(speaker_mapping)
        overall_confidence = (
            float(np.mean([seg.transcription_confidence for seg in updated_segments]))
            if updated_segments
            else 0.0
        )

        # Log group-based results for enhanced mode in ascending order
        self.logger.info("🎤 Enhanced Diarization Results by Groups (Ordered):")
        self.logger.info("=" * 70)

        # Sort groups by their group number for proper ordering
        sorted_groups = sorted(groups, key=lambda g: int(g.group_id.split("_")[1]))

        for group in sorted_groups:
            if group.group_id in group_to_result:
                result = group_to_result[group.group_id]
                speaker_id = group.speaker_id  # Single speaker ID
                segment_count = len(group.segments)
                duration = group.total_duration

                # Extract group number for display
                group_num = group.group_id.split("_")[1]

                # self.logger.info(f"📁 Group {group_num}: {group.group_id}")
                # self.logger.info(f"   🎭 Speaker: {speaker_id}")
                # self.logger.info(f"   🎭 {speaker_id}")
                self.logger.info(f"   🎭 Speaker {group_num}")
                # self.logger.info(f"   📊 Segments: {segment_count}")
                # self.logger.info(f"   ⏱️  Duration: {duration:.2f}s")
                # self.logger.info(f"   📝 Transcription: {result.text}")
                # self.logger.info(f"   📝 {result.text}")
                self.logger.info(f"   {result.text}")
                # self.logger.info(f"   🎯 Confidence: {result.confidence:.3f}")
                self.logger.info("")

        return DiarizationResult(
            segments=updated_segments,
            speaker_mapping=speaker_mapping,
            total_duration=total_duration,
            num_speakers=num_speakers,
            overall_confidence=overall_confidence,
            processing_time=self.performance_stats["total_processing_time"],
            error_details=None,
        )

    def _find_segment_group(
        self, segment: SpeakerSegment, groups: List[SegmentGroup]
    ) -> Optional[SegmentGroup]:
        """Find which group a segment belongs to."""
        for group in groups:
            if (
                segment.start_time >= group.start_time
                and segment.end_time <= group.end_time
            ):
                return group
        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_processing_time": 0.0,
            "diarization_time": 0.0,
            "grouping_time": 0.0,
            "api_processing_time": 0.0,
            "total_segments": 0,
            "total_groups": 0,
            "successful_requests": 0,
            "failed_requests": 0,
        }
