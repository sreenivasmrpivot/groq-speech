#!/usr/bin/env python3
"""
CORRECT Speech Demo with Proper Diarization Architecture.

This demo implements the CORRECT pipeline:
1. Pyannote.audio FIRST → Speaker detection
2. Audio chunking → Speaker-specific segments  
3. Groq API SECOND → Accurate transcription per segment
4. Perfect speaker attribution with accurate text

Usage:
    python speech_demo.py --file audio.wav --operation transcription
    python speech_demo.py --microphone-mode single --operation transcription
    python speech_demo.py --help
"""

import argparse
import sys
import os
from pathlib import Path
import time


# Add the parent directory to the path to import groq_speech
sys.path.insert(0, str(Path(__file__).parent.parent))

from groq_speech.speech_recognizer import SpeechRecognizer
from groq_speech.speech_config import SpeechConfig
from groq_speech.config import Config
from groq_speech.enhanced_diarization import EnhancedDiarizationConfig


def validate_environment():
    """Validate the environment configuration."""
    print("🔧 Validating Environment Configuration")
    print("=" * 50)

    # Check required environment variables
    required_vars = ["GROQ_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return False

    # Check HF_TOKEN for proper diarization
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_hf_token_here":
        print("⚠️  HF_TOKEN not configured - Limited diarization capability")
        print("💡 For full speaker diarization, configure HF_TOKEN:")
        print("   1. Get token from: " "https://huggingface.co/settings/tokens")
        print(
            "   2. Accept license: "
            "https://huggingface.co/pyannote/"
            "speaker-diarization-3.1"
        )
        print("   3. Update groq_speech/.env with: " "HF_TOKEN=your_actual_token_here")
    else:
        print("✅ HF_TOKEN configured - Full Pyannote.audio diarization enabled")

    print("✅ Environment validation passed")
    return True


def process_audio_file(audio_file: str, mode: str, recognizer: SpeechRecognizer):
    """
    Process audio file with CORRECT diarization pipeline.

    CORRECT FLOW:
    1. Pyannote.audio → Speaker detection
    2. Audio chunking → Speaker-specific segments
    3. Groq API → Accurate transcription per segment
    """
    print(f"\n📁 Processing Audio File: {audio_file}")
    print("=" * 60)
    print("🎭 CORRECT Pipeline: Pyannote.audio FIRST, then Groq API per segment")

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None

    start_time = time.time()

    try:
        # Use the CORRECT diarization method
        result = recognizer.recognize_with_correct_diarization(audio_file, mode)

        processing_time = time.time() - start_time

        if result and hasattr(result, "segments"):
            # Display diarization results
            print(f"\n✅ CORRECT diarization completed in {processing_time:.2f}s")
            print(f"🎭 Speakers detected: {result.num_speakers}")
            print(f"📊 Total segments: {len(result.segments)}")
            print(f"⏱️  Total duration: {result.total_duration:.1f}s")
            print(f"🎯 Overall confidence: {result.overall_confidence:.3f}")

            print(f"\n🎤 Speaker Segments with Accurate Transcription:")
            print("=" * 70)

            for i, segment in enumerate(result.segments):
                speaker = segment.speaker_id
                start_t = segment.start_time
                end_t = segment.end_time
                duration = end_t - start_t
                text = segment.text if hasattr(segment, "text") else "[No text]"
                confidence = (
                    segment.confidence if hasattr(segment, "confidence") else 0.0
                )

                print(f"\n🎤 {speaker}:")
                print(
                    f"   {i+1}. {start_t:8.2f}s - {end_t:8.2f}s " f"({duration:5.2f}s)"
                )
                print(f"      {text}")
                print(f"      Confidence: {confidence:.3f}")

            return result

        else:
            # Fallback to basic transcription
            print(f"🔄 Diarization failed, using basic {mode}...")

            # Load audio file and use recognize_audio_data
            try:
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_file)

                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    from scipy import signal

                    audio_data = signal.resample(
                        audio_data, int(len(audio_data) * 16000 / sample_rate)
                    )

                # Use the correct method
                if mode == "translation":
                    basic_result = recognizer.translate_audio_data(audio_data)
                else:
                    basic_result = recognizer.recognize_audio_data(audio_data)

                if basic_result and basic_result.text:
                    print(f"✅ Basic {mode} completed: {basic_result.text[:200]}...")
                    return basic_result
                else:
                    print(f"❌ Basic {mode} also failed")
                    return None

            except Exception as audio_error:
                print(f"❌ Audio loading failed: {audio_error}")
                return None

    except Exception as e:
        print(f"❌ File processing failed: {e}")
        return None


def process_audio_file_enhanced(
    audio_file: str, mode: str, recognizer: SpeechRecognizer
):
    """
    Process audio file with ENHANCED diarization dataflow.

    ENHANCED FLOW:
    1. Pyannote.audio → Speaker detection (51 segments)
    2. Smart segment grouping → Combine segments under 25MB limit
    3. Parallel Groq API processing → Process groups concurrently
    4. Ordered output → Maintain segment order and speaker mapping
    """
    print(f"\n📁 Processing Audio File: {audio_file}")
    print("=" * 60)
    print("🚀 ENHANCED Pipeline: Smart grouping + Parallel processing")

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None

    start_time = time.time()

    try:
        # Create enhanced configuration
        enhanced_config = EnhancedDiarizationConfig(
            max_file_size_mb=25.0,  # Configurable 25MB limit
            enable_parallel_processing=True,
            max_parallel_requests=4,
            retry_enabled=True,
            retry_delay_seconds=1.0,
            max_retries=3,
            log_level="INFO",
            enable_progress_reporting=True,
        )

        # Use the ENHANCED diarization method
        result = recognizer.recognize_with_enhanced_diarization(
            audio_file, mode, enhanced_config
        )

        processing_time = time.time() - start_time

        if result and hasattr(result, "segments"):
            # Display enhanced diarization results
            print(f"\n✅ ENHANCED diarization completed in {processing_time:.2f}s")
            print(f"🎭 Speakers detected: {result.num_speakers}")
            print(f"📊 Total segments: {len(result.segments)}")
            print(f"⏱️  Total duration: {result.total_duration:.1f}s")
            print(f"🎯 Overall confidence: {result.overall_confidence:.3f}")

            # print(f"\n🎤 Speaker Segments with Enhanced Transcription:")
            # print("=" * 70)

            # for i, segment in enumerate(result.segments):
            #     speaker = segment.speaker_id
            #     start_t = segment.start_time
            #     end_t = segment.end_time
            #     duration = end_t - start_t
            #     text = segment.text if hasattr(segment, "text") else "[No text]"
            #     confidence = (
            #         segment.confidence if hasattr(segment, "confidence") else 0.0
            #     )

            #     print(f"\n🎤 {speaker}:")
            #     print(f"   {i+1}. {start_t:8.2f}s - {end_t:8.2f}s ({duration:5.2f}s)")
            #     print(f"      {text}")
            #     print(f"      Confidence: {confidence:.3f}")

            return result

        else:
            print(f"❌ Enhanced diarization failed")
            return None

    except Exception as e:
        print(f"❌ Enhanced file processing failed: {e}")
        return None


def process_microphone(mode: str, recognizer: SpeechRecognizer):
    """
    Process microphone input with CORRECT diarization pipeline.

    For microphone input, we use longer segments (15-30 seconds) to allow
    Pyannote.audio to properly detect speaker patterns.
    """
    print(f"\n🎤 Microphone Input with CORRECT Diarization")
    print("=" * 60)
    print("🎭 CORRECT Pipeline: Longer segments for accurate speaker detection")
    print("💡 Recording in 30-second segments for optimal diarization")
    print("💡 Press Ctrl+C to stop")

    # Check HF_TOKEN configuration
    hf_token = Config.HF_TOKEN
    if not hf_token or hf_token == "your_hf_token_here":
        print("\n⚠️  HF_TOKEN not configured - Cannot perform proper diarization")
        print("💡 For microphone diarization, configure HF_TOKEN first")
        print("🔄 Falling back to basic transcription...")
        return process_microphone_basic(mode, recognizer)

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile

        # Use 30-second segments for proper speaker detection
        SEGMENT_DURATION = 30.0

        print(f"🎤 Recording audio in {SEGMENT_DURATION}s segments...")
        print("💡 Speak naturally - longer segments enable better speaker detection")

        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("🎤 Recording started... (Press Ctrl+C to stop)")

        frames = []
        start_time = time.time()
        segment_count = 0

        try:
            while True:
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                # Check if we have enough audio for a segment
                elapsed = time.time() - start_time
                if elapsed >= SEGMENT_DURATION:
                    segment_count += 1
                    print(
                        f"\n🎭 Processing Segment {segment_count} ({elapsed:.1f}s)..."
                    )

                    # Convert frames to numpy array
                    audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)

                    # Save to temporary file for Pyannote.audio processing
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as temp_file:
                        temp_filename = temp_file.name
                        sf.write(temp_filename, audio_data, RATE, format="WAV")

                    try:
                        # Use CORRECT pipeline on the temporary file
                        print("   🎭 Running CORRECT diarization pipeline...")
                        result = recognizer.recognize_with_correct_diarization(
                            temp_filename, mode
                        )

                        if result and hasattr(result, "segments"):
                            # Display results
                            print(f"\n🎭 Speaker Segments for Segment {segment_count}:")
                            print("=" * 50)

                            for i, segment in enumerate(result.segments):
                                speaker = segment.speaker_id
                                start_t = segment.start_time
                                end_t = segment.end_time
                                duration = end_t - start_t
                                text = (
                                    segment.text
                                    if hasattr(segment, "text")
                                    else "[No text]"
                                )

                                print(f"\n🎤 {speaker}:")
                                print(
                                    f"   {i+1}. {start_t:8.2f}s - {end_t:8.2f}s ({duration:5.2f}s)"
                                )
                                print(f"      {text}")

                        else:
                            print("   ⚠️ Diarization failed for this segment")

                        # Clean up temporary file
                        os.unlink(temp_filename)

                    except Exception as segment_error:
                        print(f"   ❌ Segment processing failed: {segment_error}")
                        # Clean up temporary file
                        if os.path.exists(temp_filename):
                            os.unlink(temp_filename)

                    # Reset for next segment
                    frames = []
                    start_time = time.time()
                    print(f"🎤 Recording next segment... (Press Ctrl+C to stop)")

        except KeyboardInterrupt:
            print("\n🛑 Stopping microphone recording...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    except Exception as e:
        print(f"❌ Microphone processing failed: {e}")
        return None


def process_microphone_single(mode: str, recognizer: SpeechRecognizer):
    """Single-shot microphone transcription without diarization."""
    print(f"\n🎤 Single Microphone {mode.title()} (No Diarization)")
    print("=" * 50)
    print("💡 Single-shot transcription mode - record once, process, show result")
    print("💡 Press Ctrl+C to stop recording and process audio")

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile

        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000

        print("🎤 Recording started - speak naturally...")
        print("💡 Press Ctrl+C when you want to stop recording and process the audio")

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []
        print("🎤 Recording... Press Ctrl+C to stop")

        try:
            # Record continuously until user interrupts
            while True:
                data = stream.read(CHUNK)
                frames.append(data)

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user, processing audio...")

            # Convert to numpy array
            audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)

            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_data, RATE)

            try:
                # Process the recorded audio
                if mode == "translation":
                    result = recognizer.translate_audio_data(audio_data)
                else:
                    result = recognizer.recognize_audio_data(audio_data)

                # Clean up temporary file
                os.unlink(temp_path)

                if result and result.text:
                    print(f"✅ {mode.title()} completed successfully!")
                    return result
                else:
                    print(f"❌ {mode.title()} failed - no text detected")
                    return None

            except Exception as e:
                print(f"❌ Audio processing failed: {e}")
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return None

            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

        except Exception as e:
            print(f"❌ Recording failed: {e}")
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass
            return None

    except ImportError:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return None
    except Exception as e:
        print(f"❌ Single microphone processing failed: {e}")
        return None


def process_microphone_basic(mode: str, recognizer: SpeechRecognizer):
    """Continuous microphone transcription without diarization."""
    print(f"\n🎤 Continuous Microphone {mode.title()} (No Diarization)")
    print("=" * 50)
    print("💡 Continuous transcription mode - real-time streaming results")
    print("💡 Press Ctrl+C to stop")

    try:
        # Set up event handlers for real-time output
        def on_recognizing(event_data):
            """Handle real-time recognition updates."""
            if hasattr(event_data, "text") and event_data.text:
                print(f"🎤 Recognizing: {event_data.text}")

        def on_recognized(event_data):
            """Handle completed recognition."""
            if hasattr(event_data, "text") and event_data.text:
                print(f"✅ Recognized: {event_data.text}")

        # Connect event handlers
        recognizer.connect("recognizing", on_recognizing)
        recognizer.connect("recognized", on_recognized)

        # Start continuous recognition
        recognizer.start_continuous_recognition()

        # Keep running until interrupted
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping basic recognition...")
        recognizer.stop_continuous_recognition()

        # Return a proper result object to indicate successful completion
        class BasicResult:
            def __init__(self):
                self.text = "[Continuous recognition stopped by user]"
                self.segments = []
                self.num_speakers = 0
                self.total_duration = 0.0
                self.overall_confidence = 0.0

        return BasicResult()
    except Exception as e:
        print(f"❌ Basic microphone processing failed: {e}")
        return None


def process_microphone_enhanced(mode: str, recognizer: SpeechRecognizer):
    """
    Process microphone input with ENHANCED diarization dataflow (single-shot).

    For microphone input, we use longer segments (15-30 seconds) to allow
    Pyannote.audio to properly detect speaker patterns, then apply the
    enhanced grouping and parallel processing.
    """
    print(f"\n🎤 Microphone Input with ENHANCED Diarization (Single-shot)")
    print("=" * 60)
    print("🚀 ENHANCED Pipeline: Smart grouping + Parallel processing")
    print("💡 Recording in 30-second segments for optimal diarization")
    print("💡 Press Ctrl+C to stop")

    # Check HF_TOKEN configuration
    hf_token = Config.HF_TOKEN
    if not hf_token or hf_token == "your_hf_token_here":
        print("\n⚠️  HF_TOKEN not configured - Cannot perform enhanced diarization")
        print("💡 For enhanced microphone diarization, configure HF_TOKEN first")
        print("🔄 Falling back to basic transcription...")
        return process_microphone_basic(mode, recognizer)

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile

        # Use 30-second segments for proper speaker detection
        SEGMENT_DURATION = 30.0

        print(f"🎤 Recording audio in {SEGMENT_DURATION}s segments...")
        print("💡 Speak naturally - longer segments enable better speaker detection")

        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("🎤 Recording started... Press Ctrl+C to stop")

        try:
            segment_count = 0

            while True:  # Continuous loop for continuous mode
                segment_count += 1
                print(
                    f"\n🔄 Recording segment {segment_count} ({SEGMENT_DURATION}s)..."
                )

                # Record segment
                frames = []
                for _ in range(0, int(RATE / CHUNK * SEGMENT_DURATION)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                # Convert to numpy array
                audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)

                # Save temporary file for enhanced diarization
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, audio_data, RATE)

                try:
                    print(
                        f"🔄 Processing segment {segment_count} with enhanced diarization..."
                    )

                    # Create enhanced configuration for microphone - match file mode settings
                    enhanced_config = EnhancedDiarizationConfig(
                        max_file_size_mb=25.0,  # Match file mode limit
                        enable_parallel_processing=True,
                        max_parallel_requests=4,  # Match file mode concurrency
                        retry_enabled=True,
                        retry_delay_seconds=1.0,  # Match file mode timing
                        max_retries=3,  # Match file mode retries
                        log_level="INFO",
                        enable_progress_reporting=True,
                    )

                    # Process with enhanced diarization
                    result = recognizer.recognize_with_enhanced_diarization(
                        temp_path, mode, enhanced_config
                    )

                    if result and result.segments:
                        print(
                            f"✅ Enhanced diarization completed for segment {segment_count}!"
                        )
                        print(f"🎭 Speakers: {result.num_speakers}")
                        print(f"📊 Segments: {len(result.segments)}")
                        print(f"⏱️  Total duration: {result.total_duration:.1f}s")
                        print(f"🎯 Overall confidence: {result.overall_confidence:.3f}")

                        # Display results in the same format as file mode - use the grouped results from logs
                        # The enhanced diarization already provides grouped results in the logs
                        # We don't need to display individual segments - the logs show the grouped results
                        # This matches the file mode behavior (which has display commented out)

                        # Note: The grouped results are already displayed in the enhanced diarization logs
                        # Lines like "🎭 Speaker 01: 📝 [grouped text]" show the correct grouped results
                        # Individual segments are for internal processing, not display

                    else:
                        print(
                            f"⚠️  Enhanced diarization failed for segment {segment_count}"
                        )

                except Exception as e:
                    print(f"❌ Segment {segment_count} processing failed: {e}")

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                # Continue to next segment
                print(f"🎤 Recording next segment... (Press Ctrl+C to stop)")

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user")
            return None

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    except ImportError:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return None
    except Exception as e:
        print(f"❌ Enhanced microphone processing failed: {e}")
        return None


def process_microphone_enhanced_continuous(mode: str, recognizer: SpeechRecognizer):
    """
    Process microphone input with ENHANCED diarization dataflow (continuous).

    For continuous microphone input, we record and process multiple 30-second segments
    until the user stops with Ctrl+C. Each segment is processed with enhanced diarization.
    """
    print(f"\n🎤 Microphone Input with ENHANCED Diarization (Continuous)")
    print("=" * 60)
    print("🚀 ENHANCED Pipeline: Smart grouping + Parallel processing")
    print("💡 Recording in 30-second segments for optimal diarization")
    print("💡 Press Ctrl+C to stop")

    # Check HF_TOKEN configuration
    hf_token = Config.HF_TOKEN
    if not hf_token or hf_token == "your_hf_token_here":
        print("\n⚠️  HF_TOKEN not configured - Cannot perform enhanced diarization")
        print("💡 For enhanced microphone diarization, configure HF_TOKEN first")
        print("🔄 Falling back to basic transcription...")
        return process_microphone_basic(mode, recognizer)

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile

        # Use 30-second segments for proper speaker detection
        SEGMENT_DURATION = 30.0

        print(f"🎤 Recording audio in {SEGMENT_DURATION}s segments...")
        print("💡 Speak naturally - longer segments enable better speaker detection")

        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("🎤 Recording started... Press Ctrl+C to stop")

        segment_count = 0

        try:
            while True:  # Continuous loop for continuous mode
                segment_count += 1
                print(
                    f"\n🔄 Recording segment {segment_count} ({SEGMENT_DURATION}s)..."
                )

                # Record segment
                frames = []
                for _ in range(0, int(RATE / CHUNK * SEGMENT_DURATION)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                # Convert to numpy array
                audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)

                # Save temporary file for enhanced diarization
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, audio_data, RATE)

                try:
                    print(
                        f"🔄 Processing segment {segment_count} with enhanced diarization..."
                    )

                    # Create enhanced configuration for microphone - match file mode settings
                    enhanced_config = EnhancedDiarizationConfig(
                        max_file_size_mb=25.0,  # Match file mode limit
                        enable_parallel_processing=True,
                        max_parallel_requests=4,  # Match file mode concurrency
                        retry_enabled=True,
                        retry_delay_seconds=1.0,  # Match file mode timing
                        max_retries=3,  # Match file mode retries
                        log_level="INFO",
                        enable_progress_reporting=True,
                    )

                    # Process with enhanced diarization
                    result = recognizer.recognize_with_enhanced_diarization(
                        temp_path, mode, enhanced_config
                    )

                    if result and result.segments:
                        print(
                            f"✅ Enhanced diarization completed for segment {segment_count}!"
                        )
                        print(f"🎭 Speakers: {result.num_speakers}")
                        print(f"📊 Segments: {len(result.segments)}")
                        print(f"⏱️  Total duration: {result.total_duration:.1f}s")
                        print(f"🎯 Overall confidence: {result.overall_confidence:.3f}")

                        # Display results in the same format as file mode - use the grouped results from logs
                        # The enhanced diarization already provides grouped results in the logs
                        # We don't need to display individual segments - the logs show the grouped results
                        # This matches the file mode behavior (which has display commented out)

                        # Note: The grouped results are already displayed in the enhanced diarization logs
                        # Lines like "🎭 Speaker 01: 📝 [grouped text]" show the correct grouped results
                        # Individual segments are for internal processing, not display

                    else:
                        print(
                            f"⚠️  Enhanced diarization failed for segment {segment_count}"
                        )

                except Exception as e:
                    print(f"❌ Segment {segment_count} processing failed: {e}")

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                # Continue to next segment
                print(f"🎤 Recording next segment... (Press Ctrl+C to stop)")

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user")

            # Return a proper result object to indicate successful completion
            class ContinuousResult:
                def __init__(self):
                    self.text = "[Continuous enhanced diarization stopped by user]"
                    self.segments = []
                    self.num_speakers = 0
                    self.total_duration = 0.0
                    self.overall_confidence = 0.0

            return ContinuousResult()

        finally:
            # Only clean up stream when the entire function exits
            stream.stop_stream()
            stream.close()
            p.terminate()

    except ImportError:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return None
    except Exception as e:
        print(f"❌ Enhanced microphone processing failed: {e}")
        return None


def main():
    """Main function to handle command line arguments and execute the demo."""
    parser = argparse.ArgumentParser(
        description="Enhanced Speech Demo with Smart Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # File-based transcription (default operation, enhanced diarization)
    python speech_demo.py --file audio.wav
    
    # File-based translation (enhanced diarization)
    python speech_demo.py --file audio.wav --operation translation
    
    # Microphone transcription (default operation, no diarization)
    python speech_demo.py --microphone-mode single
    
    # Microphone translation (no diarization)
    python speech_demo.py --microphone-mode single --operation translation
    
    # Microphone with enhanced diarization
    python speech_demo.py --microphone-mode single --diarize true
    
    # Microphone continuous with enhanced diarization and translation
    python speech_demo.py --microphone-mode continuous --operation translation --diarize true
    
    # File with enhanced diarization (always enabled for files)
    python speech_demo.py --file audio.wav --diarize true
        """,
    )

    parser.add_argument(
        "--operation",
        choices=["transcription", "translation"],
        default="transcription",
        help="Operation: transcription (speech-to-text) or translation "
        "(speech-to-text in target language). Defaults to transcription.",
    )

    parser.add_argument(
        "--microphone-mode",
        choices=["single", "continuous"],
        help="Microphone mode: single (record once, process, show result) "
        "or continuous (real-time streaming results)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Audio file path for recognition",
    )

    parser.add_argument(
        "--diarize",
        default=False,
        type=bool,
        help="Enable enhanced speaker diarization with smart grouping and "
        "parallel processing. Defaults to False.",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.file and not args.microphone_mode:
        parser.error("Either --file or --microphone-mode must be specified")

    if args.file and args.microphone_mode:
        parser.error("Cannot specify both --file and --microphone-mode")

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Create speech recognizer
    try:
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config)
        print("✅ Speech recognizer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize speech recognizer: {e}")
        sys.exit(1)

    # Process based on arguments
    try:
        if args.file:
            # For file processing, use enhanced diarization by default
            # (better performance)
            result = process_audio_file_enhanced(args.file, args.operation, recognizer)
        else:  # microphone
            if args.diarize:
                # Use enhanced diarization for microphone when explicitly
                # requested (better performance with smart grouping)
                if args.microphone_mode == "single":
                    result = process_microphone_enhanced(args.operation, recognizer)
                else:  # continuous
                    result = process_microphone_enhanced_continuous(
                        args.operation, recognizer
                    )
            else:
                # Use basic microphone processing (no diarization) by default
                if args.microphone_mode == "single":
                    result = process_microphone_single(args.operation, recognizer)
                else:  # continuous
                    result = process_microphone_basic(args.operation, recognizer)

        if result:
            print(f"\n✅ Processing completed successfully!")
            print(f"🎯 Operation: {args.operation}")
            if hasattr(result, "segments"):
                print(f"🎭 Speakers: {result.num_speakers}")
                print(f"📊 Segments: {len(result.segments)}")
            else:
                print(f"📝 Text: {result.text[:200]}...")
        else:
            print(f"\n❌ Processing failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n🛑 Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    exit(main())
