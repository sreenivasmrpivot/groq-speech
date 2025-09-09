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
import warnings


# Global verbose flag for logging control
VERBOSE_MODE = False

def configure_warnings(verbose: bool = False):
    """Configure warning display based on verbose mode."""
    if not verbose:
        # Suppress PyTorch/TorchAudio deprecation warnings in production mode
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        warnings.filterwarnings("ignore", message=".*torchaudio.*")
        warnings.filterwarnings("ignore", message=".*TorchCodec.*")
    else:
        # Show all warnings in verbose mode
        warnings.resetwarnings()

def log_debug(message: str):
    """Log debug message only in verbose mode."""
    if VERBOSE_MODE:
        print(f"🔍 {message}")

def log_info(message: str):
    """Log info message always."""
    print(f"ℹ️  {message}")

def log_success(message: str):
    """Log success message always."""
    print(f"✅ {message}")

def log_warning(message: str):
    """Log warning message always."""
    print(f"⚠️  {message}")

def log_error(message: str):
    """Log error message always."""
    print(f"❌ {message}")


# Add the parent directory to the path to import groq_speech
sys.path.insert(0, str(Path(__file__).parent.parent))

from groq_speech.speech_recognizer import SpeechRecognizer
from groq_speech.speech_config import SpeechConfig
from groq_speech.speaker_diarization import DiarizationConfig


def validate_environment(enable_diarization: bool = False):
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

    # Only check HF_TOKEN if diarization is needed
    if enable_diarization:
        hf_token = SpeechConfig.get_hf_token()
        if not hf_token:
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
    else:
        print("ℹ️  Diarization not requested - Skipping HF_TOKEN validation")

    print("✅ Environment validation passed")
    return True


def process_audio_file(audio_file: str, mode: str, recognizer: SpeechRecognizer, enable_diarization: bool = True, verbose: bool = False):
    """
    Process audio file with diarization pipeline.

    FLOW:
    1. If diarization enabled: Pyannote.audio → Speaker detection → Groq API per segment
    2. If diarization disabled: Direct Groq API processing
    """
    print(f"\n📁 Processing Audio File: {audio_file}")
    print("=" * 60)
    
    if enable_diarization:
        print("🎭 Diarization Pipeline: Pyannote.audio FIRST, then Groq API per segment")
        log_debug("Running CORRECT diarization pipeline...")
        log_debug("1. Pyannote.audio → Speaker detection")
        log_debug("2. Audio chunking → Speaker-specific segments") 
        log_debug("3. Groq API → Accurate transcription per segment")
    else:
        print("🎯 Direct Pipeline: Groq API processing without diarization")

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None

    start_time = time.time()

    try:
        if enable_diarization:
            # Use the diarization method
            result = recognizer._process_audio_with_diarization(audio_file, mode)
        else:
            # Use direct processing without diarization
            if mode == "translation":
                result = recognizer.translate_file(audio_file, enable_diarization=False)
            else:
                result = recognizer.recognize_file(audio_file, enable_diarization=False)

        if not result:
            # Fallback to basic transcription
            print(f"🔄 Basic {mode} failed, attempting fallback...")

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
                    basic_result = recognizer.translate_audio_data(audio_data, sample_rate)
                else:
                    basic_result = recognizer.recognize_audio_data(audio_data, sample_rate)

                if basic_result and basic_result.text:
                    print(f"✅ Basic {mode} completed: {basic_result.text[:200]}...")
                    return basic_result
                else:
                    print(f"❌ Basic {mode} also failed")
                    return None

            except Exception as audio_error:
                print(f"❌ Audio loading failed: {audio_error}")
                return None

        else:
            return result

    except Exception as e:
        print(f"❌ File processing failed: {e}")
        return None

def process_microphone_single(mode: str, recognizer: SpeechRecognizer, enable_diarization: bool = False, verbose: bool = False):
    """Simple single-shot microphone recording - record until Ctrl+C, then process everything."""
    print(f"\n🎤 Single Microphone {mode.title()}")
    print("=" * 50)
    if enable_diarization:
        print("💡 Single-shot transcription with diarization - record once, process, show result")
    else:
        print("💡 Single-shot transcription mode - record once, process, show result")
    print("💡 Press Ctrl+C to stop recording and process audio")

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile

        # Audio recording parameters
        CHUNK = 8192
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

        all_frames = []
        print("🎤 Recording... Press Ctrl+C to stop")

        # Simple visual feedback
        last_visual_update = time.time()
        visual_update_interval = 1.0  # Update every second

        try:
            # Simple recording loop - just record until Ctrl+C
            while True:
                data = stream.read(CHUNK)
                all_frames.append(data)
                
                # Simple visual feedback
                current_time = time.time()
                if current_time - last_visual_update >= visual_update_interval:
                    duration = len(all_frames) * CHUNK / RATE
                    estimated_size_mb = (len(all_frames) * CHUNK * 4) / (1024 * 1024)  # 32-bit float = 4 bytes
                    print(f"\r🎤 Recording... {duration:.1f}s | {estimated_size_mb:.1f}MB", end="", flush=True)
                    last_visual_update = current_time

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user, processing audio...")
            
            if not all_frames:
                print("❌ No audio recorded")
                return None

            # Convert all audio to numpy array
            audio_data = np.frombuffer(b"".join(all_frames), dtype=np.float32)
            total_duration = len(audio_data) / RATE
            print(f"📊 Total recording: {total_duration:.1f}s ({len(audio_data)} samples)")

            # Process the entire recording
            return _process_audio_chunk(audio_data, RATE, mode, recognizer, enable_diarization)

        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    except ImportError:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return None
    except Exception as e:
        print(f"❌ Single microphone processing failed: {e}")
        return None

def _process_audio_chunk(audio_data, sample_rate, mode, recognizer, enable_diarization):
    """Process a single audio chunk."""
    try:
        import tempfile
        import soundfile as sf
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)

        try:
            # Process the recorded audio
            if enable_diarization:
                # Use diarization
                if mode == "translation":
                    result = recognizer.translate_file(temp_path, enable_diarization=True)
                else:
                    result = recognizer.recognize_file(temp_path, enable_diarization=True)
            else:
                # Direct processing without diarization
                if mode == "translation":
                    result = recognizer.translate_audio_data(audio_data, sample_rate)
                else:
                    result = recognizer.recognize_audio_data(audio_data, sample_rate)

            if result and hasattr(result, "text") and result.text:
                print(f"✅ {mode.title()} completed successfully!")
                if hasattr(result, "segments") and result.segments:
                    print(f"🎭 Speakers detected: {result.num_speakers}")
                    print(f"📊 Total segments: {len(result.segments)}")
                return result
            elif result and hasattr(result, "segments") and result.segments:
                print(f"✅ {mode.title()} completed successfully!")
                print(f"🎭 Speakers detected: {result.num_speakers}")
                print(f"📊 Total segments: {len(result.segments)}")
                return result
            else:
                print(f"❌ {mode.title()} failed - no text detected")
                return None

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return None

def _process_large_audio_in_chunks(audio_data, sample_rate, mode, recognizer, enable_diarization):
    """Process large audio by splitting into 24MB chunks."""
    try:
        import numpy as np
        
        # 24MB limit calculation
        MAX_SAMPLES = int(sample_rate * 390)  # 6.5 minutes
        chunk_size = MAX_SAMPLES
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        
        print(f"🔄 Processing {total_chunks} chunks of audio...")
        
        all_results = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk_num = i // chunk_size + 1
            chunk_audio = audio_data[i:i + chunk_size]
            chunk_duration = len(chunk_audio) / sample_rate
            
            print(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({chunk_duration:.1f}s)...")
            
            result = _process_audio_chunk(chunk_audio, sample_rate, mode, recognizer, enable_diarization)
            
            if result and hasattr(result, "text") and result.text:
                all_results.append(result)
                print(f"✅ Chunk {chunk_num}: {result.text[:100]}{'...' if len(result.text) > 100 else ''}")
            elif result and hasattr(result, "segments") and result.segments:
                all_results.append(result)
                print(f"✅ Chunk {chunk_num}: Diarization completed ({result.num_speakers} speakers)")
            else:
                print(f"⚠️  Chunk {chunk_num}: No text detected")
        
        if all_results:
            # Combine all results
            combined_text = " ".join([r.text for r in all_results if hasattr(r, "text") and r.text])
            print(f"\n✅ Combined {mode.title()} completed successfully!")
            print(f"📊 Processed {len(all_results)} chunks")
            
            # Create a combined result object
            from groq_speech.speech_recognizer import SpeechRecognitionResult, ResultReason
            combined_result = SpeechRecognitionResult(
                text=combined_text,
                reason=ResultReason.RecognizedSpeech,
                confidence=sum(r.confidence for r in all_results if hasattr(r, "confidence")) / len(all_results) if all_results else 0.0
            )
            return combined_result
        else:
            print(f"❌ No text detected in any chunk")
            return None
            
    except Exception as e:
        print(f"❌ Large audio processing failed: {e}")
        return None

def process_microphone_continuous(mode: str, recognizer: SpeechRecognizer, enable_diarization: bool = False, verbose: bool = False):
    """Continuous microphone transcription with real-time processing and 24MB chunking."""
    print(f"\n🎤 Continuous Microphone {mode.title()}")
    print("=" * 50)
    if enable_diarization:
        print("💡 Continuous transcription with diarization - real-time streaming results")
    else:
        print("💡 Continuous transcription mode - real-time streaming results")
    print("💡 Press Ctrl+C to stop")

    try:
        import pyaudio
        import numpy as np
        import soundfile as sf
        import tempfile
        import threading
        import queue
        import time

        # Audio recording parameters
        CHUNK = 8192  # Further increased buffer size to prevent overflow
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000
        
        # 24MB limit calculation for 16kHz, 32-bit float audio
        # 24MB = 25,165,824 bytes
        # 32-bit float = 4 bytes per sample
        # Max samples = 25,165,824 / 4 = 6,291,456 samples
        # Max duration = 6,291,456 / 16,000 ≈ 393 seconds ≈ 6.5 minutes
        MAX_SAMPLES = int(RATE * 390)  # Conservative 6.5 minutes
        MAX_BYTES = 24 * 1024 * 1024  # 24MB in bytes
        MAX_DURATION_SECONDS = 390  # 6.5 minutes

        print(f"🎤 Recording continuously...")
        print(f"💡 Audio will be chunked at 24MB limit ({MAX_SAMPLES/RATE/60:.1f} minutes) for optimal processing")
        print("💡 Press Ctrl+C to stop")

        # Queue for audio processing
        audio_queue = queue.Queue()
        processing_active = True

        def audio_processor():
            """Background thread for processing audio chunks."""
            while processing_active:
                try:
                    # Get audio data from queue with timeout
                    audio_data, segment_num = audio_queue.get(timeout=1.0)
                    
                    # Process the audio chunk
                    _process_continuous_audio_chunk_async(audio_data, RATE, mode, recognizer, enable_diarization, segment_num)
                    
                    audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Audio processing error: {e}")

        # Start background processing thread
        processor_thread = threading.Thread(target=audio_processor, daemon=True)
        processor_thread.start()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        chunk_count = 0
        accumulated_audio = []
        accumulated_bytes = 0
        last_visual_update = time.time()
        visual_update_interval = 0.5  # Update visual feedback every 0.5 seconds

        try:
            while True:  # Continuous loop until Ctrl+C
                # Read audio data continuously with error handling
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"⚠️  Audio read error: {e}, continuing...")
                    continue
                
                # Add raw bytes to accumulated audio (more efficient)
                accumulated_audio.append(data)
                accumulated_bytes += len(data)
                
                # Convert current audio to numpy for VAD analysis
                audio_bytes = b"".join(accumulated_audio)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                duration = len(audio_array) / RATE
                
                # Visual feedback - show audio level and status
                current_time = time.time()
                if current_time - last_visual_update >= visual_update_interval:
                    audio_level = recognizer.vad_service.get_audio_level(audio_array[-RATE:])  # Last 1 second
                    level_bars = "█" * int(audio_level * 20) + "░" * (20 - int(audio_level * 20))
                    print(f"\r🎤 Listening... [{level_bars}] {audio_level:.2f} | {duration:.1f}s | {accumulated_bytes/1024/1024:.1f}MB", end="", flush=True)
                    last_visual_update = current_time
                
                # Check if we should create a chunk using VAD
                should_create, reason = recognizer.vad_service.should_create_chunk(
                    audio_array, RATE, MAX_DURATION_SECONDS
                )
                
                if should_create:
                    chunk_count += 1
                    print(f"\n🔄 Chunk {chunk_count} created: {reason}")
                    log_debug(f"Chunk {chunk_count}: {len(audio_array)} samples, {duration:.1f}s, {accumulated_bytes/1024/1024:.1f}MB")
                    
                    # Queue for processing
                    audio_queue.put((audio_array, chunk_count))
                    
                    # Reset accumulated audio
                    accumulated_audio = []
                    accumulated_bytes = 0

        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous recognition...")
            processing_active = False
            
            # Process any remaining audio
            if accumulated_audio:
                chunk_count += 1
                print(f"🔄 Processing final chunk {chunk_count} ({accumulated_bytes/1024/1024:.1f}MB)...")
                
                # Convert accumulated bytes to numpy array
                audio_bytes = b"".join(accumulated_audio)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                duration = len(audio_array) / RATE
                log_debug(f"Final chunk {chunk_count}: {len(audio_array)} samples, {duration:.1f}s")
                
                audio_queue.put((audio_array, chunk_count))
            
            # Wait for processing to complete
            print("⏳ Waiting for final processing to complete...")
            audio_queue.join()
            
            print("✅ Continuous processing completed successfully!")
            return None

        finally:
            processing_active = False
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass

    except ImportError:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return None
    except Exception as e:
        print(f"❌ Continuous microphone processing failed: {e}")
        return None

def _process_continuous_audio_chunk_async(audio_data, sample_rate, mode, recognizer, enable_diarization, segment_num=None):
    """Process a single audio chunk for continuous mode asynchronously."""
    try:
        import tempfile
        import soundfile as sf
        import numpy as np
        
        # Check if audio has sufficient volume (simple silence detection)
        audio_rms = np.sqrt(np.mean(audio_data**2))
        print(f"🔍 Chunk {segment_num}: Audio RMS = {audio_rms:.6f}, Samples = {len(audio_data)}")
        
        if audio_rms < 0.001:  # Very quiet audio, likely silence (lowered threshold)
            print(f"⚠️  Chunk {segment_num}: Audio too quiet (silence detected, RMS={audio_rms:.6f})")
            return
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)

        try:
            # Process the recorded audio
            if enable_diarization:
                # Use diarization
                if mode == "translation":
                    result = recognizer.translate_file(temp_path, enable_diarization=True)
                else:
                    result = recognizer.recognize_file(temp_path, enable_diarization=True)
            else:
                # Direct processing without diarization
                if mode == "translation":
                    result = recognizer.translate_audio_data(audio_data, sample_rate)
                else:
                    result = recognizer.recognize_audio_data(audio_data, sample_rate)

            if result and hasattr(result, "text") and result.text and len(result.text.strip()) > 0:
                segment_info = f"Segment {segment_num}" if segment_num else "Audio"
                print(f"✅ {segment_info} {mode.title()}: {result.text}")
                if hasattr(result, "segments") and result.segments:
                    print(f"🎭 Speakers: {result.num_speakers}, Segments: {len(result.segments)}")
            elif result and hasattr(result, "segments") and result.segments:
                segment_info = f"Segment {segment_num}" if segment_num else "Audio"
                print(f"✅ {segment_info} {mode.title()}: Diarization completed")
                print(f"🎭 Speakers: {result.num_speakers}, Segments: {len(result.segments)}")
                
                # Display individual speaker segments with their text
                for i, segment in enumerate(result.segments):
                    speaker = segment.speaker_id
                    text = getattr(segment, 'text', '') or getattr(segment, 'transcription', '') or '[No text]'
                    if text and text.strip():
                        print(f"🎤 {speaker}: {text}")
                    else:
                        print(f"🎤 {speaker}: [No text detected]")
            else:
                segment_info = f"Segment {segment_num}" if segment_num else "Audio"
                print(f"⚠️  {segment_info} {mode.title()}: No text detected")

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        segment_info = f"Segment {segment_num}" if segment_num else "Audio"
        print(f"❌ {segment_info} processing failed: {e}")

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
    
    # Microphone single mode (record until Ctrl+C, then process)
    python speech_demo.py --microphone-mode single
    
    # Microphone single mode with translation
    python speech_demo.py --microphone-mode single --operation translation
    
    # Microphone single mode with diarization
    python speech_demo.py --microphone-mode single --diarize true
    
    # Microphone continuous mode (real-time processing with silence detection)
    python speech_demo.py --microphone-mode continuous --operation translation --diarize true
    
    # File with enhanced diarization (always enabled for files)
    python speech_demo.py --file audio.wav --diarize true
    
    # Production mode (clean output for demos)
    python speech_demo.py --microphone-mode continuous --diarize true
    
    # Debug mode (verbose logging for development/troubleshooting)
    python speech_demo.py --microphone-mode continuous --diarize true --verbose
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
        help="Microphone mode: single (record until Ctrl+C, then process everything) "
        "or continuous (real-time processing with silence detection)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Audio file path for recognition",
    )

    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable enhanced speaker diarization with smart grouping and "
        "parallel processing. Defaults to False.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging for development and troubleshooting. "
        "Default: production mode with clean output.",
    )

    args = parser.parse_args()
    
    # Set global verbose mode
    global VERBOSE_MODE
    VERBOSE_MODE = args.verbose
    
    # Configure warning display based on verbose mode
    configure_warnings(args.verbose)

    # Validate arguments
    if not args.file and not args.microphone_mode:
        parser.error("Either --file or --microphone-mode must be specified")

    if args.file and args.microphone_mode:
        parser.error("Cannot specify both --file and --microphone-mode")

    # Validate environment (only check HF_TOKEN if diarization is needed)
    if not validate_environment(args.diarize):
        sys.exit(1)

    # Create speech recognizer
    try:
        speech_config = SpeechConfig()
        
        # Configure translation if requested
        if args.operation == "translation":
            speech_config.enable_translation = True
            speech_config.set_translation_target_language("en")
            print("🔀 Translation mode enabled (target: English)")
        
        recognizer = SpeechRecognizer(speech_config)
        print("✅ Speech recognizer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize speech recognizer: {e}")
        sys.exit(1)

    # Process based on arguments
    try:
        if args.file:
            # For file processing, use diarization based on --diarize parameter
            result = process_audio_file(args.file, args.operation, recognizer, args.diarize, args.verbose)
        else:  # microphone
            if args.microphone_mode == "single":
                result = process_microphone_single(args.operation, recognizer, args.diarize, args.verbose)
            else:  # continuous
                result = process_microphone_continuous(args.operation, recognizer, args.diarize, args.verbose)

        if result:
            print(f"\n✅ Processing completed successfully!")
            if hasattr(result, "segments"):
                print(f"🎭 Speakers: {result.num_speakers}")
                print(f"📊 Speaker Segment Groups: {len(result.segments)}")
                for i, segment in enumerate(result.segments):
                    speaker = segment.speaker_id
                    text = (
                        segment.text
                        if hasattr(segment, "text")
                        else "[No text]"
                    )

                    print(f"\n🎤 {speaker}:")
                    print(f"      {text}")

            elif hasattr(result, "text"):
                print(f"📝 Text: {result.text}")
        elif args.microphone_mode == "continuous":
            # Continuous mode returns None by design (real-time processing)
            print(f"\n✅ Continuous processing completed successfully!")
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
