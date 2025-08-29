"""
Enhanced Diarization with Smart Segment Grouping and Parallel Processing.

This module implements the advanced diarization dataflow:
1. Pyannote.audio → Speaker detection
2. Smart segment grouping → Combine segments by speaker continuity
3. Parallel Groq API processing → Process groups concurrently
4. Ordered output → Maintain segment order and speaker mapping
"""

import time
import logging
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
    speaker_id: str
    estimated_size_mb: float
    audio_data: Optional[np.ndarray] = None


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


class EnhancedDiarizationConfig:
    """Configuration for enhanced diarization with smart grouping."""

    def __init__(
        self,
        max_file_size_mb: float = 25.0,
        enable_parallel_processing: bool = True,
        max_parallel_requests: int = 8,
        retry_enabled: bool = True,
        retry_delay_seconds: float = 1.0,
        max_retries: int = 3,
        log_level: str = "INFO",
        enable_progress_reporting: bool = True,
    ):
        self.max_file_size_mb = max_file_size_mb
        self.enable_parallel_processing = enable_parallel_processing
        self.max_parallel_requests = max_parallel_requests
        self.retry_enabled = retry_enabled
        self.retry_delay_seconds = retry_delay_seconds
        self.max_retries = max_retries
        self.log_level = log_level
        self.enable_progress_reporting = enable_progress_reporting

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
        if self.max_file_size_mb <= 0 or self.max_parallel_requests < 1:
            return False
        return True


class SmartSegmentGrouper:
    """Intelligent segment grouping algorithm."""

    def __init__(self, config: EnhancedDiarizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def group_segments(
        self, segments: List[SpeakerSegment], audio_data: np.ndarray, sample_rate: int
    ) -> List[SegmentGroup]:
        """Group segments by speaker continuity."""
        if not segments:
            return []

        self.logger.info(f"Grouping {len(segments)} segments by speaker continuity")
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        groups = []
        current_group = []
        current_speaker = None
        group_counter = 1

        for segment in sorted_segments:
            if current_speaker is not None and segment.speaker_id != current_speaker:
                if current_group:
                    group = self._create_segment_group(
                        current_group, audio_data, sample_rate, group_counter
                    )
                    groups.append(group)
                    group_counter += 1
                current_group = [segment]
                current_speaker = segment.speaker_id
            else:
                if current_speaker is None:
                    current_speaker = segment.speaker_id
                current_group.append(segment)

        if current_group:
            group = self._create_segment_group(
                current_group, audio_data, sample_rate, group_counter
            )
            groups.append(group)

        self.logger.info(f"Created {len(groups)} speaker-based groups")
        return groups

    def _calculate_segment_size_mb(
        self, segment: SpeakerSegment, sample_rate: int
    ) -> float:
        """Calculate estimated size of a segment in MB."""
        duration = segment.end_time - segment.start_time
        size_bytes = duration * sample_rate * 2  # 16-bit mono
        return size_bytes / (1024 * 1024)

    def _create_segment_group(
        self,
        segments: List[SpeakerSegment],
        audio_data: np.ndarray,
        sample_rate: int,
        group_number: int,
    ) -> SegmentGroup:
        """Create a segment group from a list of segments."""
        group_start_time = min(seg.start_time for seg in segments)
        group_end_time = max(seg.end_time for seg in segments)
        group_duration = group_end_time - group_start_time

        start_sample = int(group_start_time * sample_rate)
        end_sample = int(group_end_time * sample_rate)
        group_audio = audio_data[start_sample:end_sample]

        group_id = f"group_{group_number:02d}"
        speaker_id = segments[0].speaker_id if segments else "Unknown"
        size_mb = self._calculate_segment_size_mb(segments[0], sample_rate) * len(
            segments
        )

        return SegmentGroup(
            group_id=group_id,
            segments=segments,
            start_time=group_start_time,
            end_time=group_end_time,
            total_duration=group_duration,
            speaker_id=speaker_id,
            estimated_size_mb=size_mb,
            audio_data=group_audio,
        )


class EnhancedDiarizer:
    """Enhanced diarization with smart segment grouping and parallel processing."""

    def __init__(
        self,
        config: Optional[EnhancedDiarizationConfig] = None,
        speech_config: Optional[SpeechConfig] = None,
    ):
        self.config = config or EnhancedDiarizationConfig.from_environment()
        self.config.validate()
        self.speech_config = speech_config or SpeechConfig()
        self.segment_grouper = SmartSegmentGrouper(self.config)
        self.base_diarizer = SpeakerDiarizer()
        self._setup_logging()

        self.performance_stats = {
            "total_processing_time": 0.0,
            "diarization_time": 0.0,
            "grouping_time": 0.0,
            "api_processing_time": 0.0,
            "total_segments": 0,
            "total_groups": 0,
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
        """Execute the enhanced diarization dataflow."""
        start_time = time.time()
        self.logger.info(f"Starting enhanced diarization flow for {audio_file}")

        try:
            # Step 1: Pyannote.audio speaker detection
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
            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

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
            self.performance_stats["total_processing_time"] = time.time() - start_time

            self.logger.info("✅ Enhanced diarization flow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"Enhanced diarization flow failed: {e}")
            raise DiarizationError(f"Enhanced diarization failed: {str(e)}")

    def _detect_speakers_pyannote(self, audio_file: str) -> List[SpeakerSegment]:
        """Detect speakers using Pyannote.audio."""
        try:
            if not self.base_diarizer._pipeline:
                self.logger.warning(
                    "Pyannote.audio pipeline not available, using fallback"
                )
                return self._create_fallback_segments(audio_file)

            import torch

            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
            diarization = self.base_diarizer._pipeline(
                {"waveform": audio_tensor, "sample_rate": sample_rate}
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                    speaker_id=speaker,
                    confidence=0.8,
                )
                segments.append(segment)

            if segments:
                self.logger.info(
                    f"Pyannote.audio detected {len(segments)} speaker segments"
                )
                return segments
            else:
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
            f"(max {self.config.max_parallel_requests} concurrent)"
        )
        results = []

        with ThreadPoolExecutor(
            max_workers=self.config.max_parallel_requests
        ) as executor:
            future_to_group = {}
            for group in groups:
                future = executor.submit(
                    self._process_single_group, group, mode, speech_recognizer
                )
                future_to_group[future] = group

            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    result = future.result()
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
                        f"Group {group.group_id} failed, retrying ({retry_count}/{self.config.max_retries}): {e}"
                    )
                    time.sleep(self.config.retry_delay_seconds)
                else:
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

        # Fallback return (should never be reached)
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
        group_to_result = {result.group_id: result for result in group_results}
        updated_segments = []
        speaker_mapping = {}

        for segment in original_segments:
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

            if segment.speaker_id not in speaker_mapping:
                speaker_mapping[segment.speaker_id] = f"Speaker {segment.speaker_id}"
            updated_segments.append(segment)

        total_duration = (
            max(seg.end_time for seg in updated_segments) if updated_segments else 0.0
        )
        num_speakers = len(speaker_mapping)
        overall_confidence = (
            float(np.mean([seg.transcription_confidence for seg in updated_segments]))
            if updated_segments
            else 0.0
        )

        # Log group-based results in ascending order
        self.logger.info("🎤 Enhanced Diarization Results by Groups (Ordered):")
        self.logger.info("=" * 70)

        sorted_groups = sorted(groups, key=lambda g: int(g.group_id.split("_")[1]))
        for group in sorted_groups:
            if group.group_id in group_to_result:
                result = group_to_result[group.group_id]
                group_num = group.group_id.split("_")[1]
                self.logger.info(f"   🎭 Speaker {group_num}")
                self.logger.info(f"   📝 {result.text}")
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
        }
