#!/usr/bin/env python3
"""
CLI Speech Recognition Example using Groq Speech SDK

This example demonstrates how to use the groq_speech library directly
for speech recognition and translation from the command line.

Usage:
    python cli_speech_recognition.py --mode transcription --language en-US
    python cli_speech_recognition.py --mode translation --language es-ES
    python cli_speech_recognition.py --file audio.wav --language en-US

SEQUENCE OF OPERATIONS:
1. INITIALIZATION
   - Parse command line arguments (main() -> argparse)
   - Validate GROQ_API_KEY environment variable (validate_api_key())
   - Setup signal handlers for graceful shutdown (SIGINT, SIGTERM)
     (_setup_signal_handlers())
   - Create SpeechConfig and SpeechRecognizer instances
     (setup_speech_config(), SpeechRecognizer())

2. CONFIGURATION
   - Set recognition mode (transcription/translation) (main() -> args.mode)
   - Configure target language for translation (setup_speech_config())
   - Setup audio input source (microphone or file) (main() -> args.file)

3. EXECUTION
   - If file mode: Process audio file and return results
     (recognize_from_file())
   - If microphone mode: Start continuous recognition with event handlers
     (recognize_from_microphone())
   - Handle real-time speech recognition results via events
   - Display transcription/translation with confidence scores

4. SHUTDOWN
   - Stop recording when signal received (Ctrl+C)
     (_signal_handler() -> stop_recording())
   - Clean up audio resources and release microphone access
   - Exit gracefully (main() exception handling)

KEY COMPONENTS:
- Event-driven architecture using built-in continuous recognition
- Proper signal handling ensures clean shutdown
- Event handlers provide real-time results
- Error handling provides user-friendly feedback
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from typing import Optional
import numpy as np

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, AudioConfig
from groq_speech.config import Config


class CLISpeechRecognition:
    """CLI-based speech recognition using Groq Speech SDK."""

    def __init__(self) -> None:
        self.recognizer: Optional[SpeechRecognizer] = None
        self.is_recording: bool = False
        self.is_single_mode_recording: bool = False  # Track single mode recording
        self.audio_config: Optional[AudioConfig] = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.

        CRITICAL: Signal handling is essential in this speech recognition
        application because:
        1. User Control: Users need to stop continuous microphone recording
           with Ctrl+C
        2. Resource Cleanup: Prevents microphone threads from running
           indefinitely and consuming system resources
        3. System Integration: Allows the OS to gracefully terminate the
           process when needed
        4. Professional Behavior: Ensures clean shutdown without crashes or
           resource leaks
        5. Thread Safety: Prevents microphone threads from being orphaned
           when the main process exits

        Without proper signal handling, the application could:
        - Leave microphone threads running in the background
        - Consume system resources indefinitely
        - Require force-killing the process (which can cause audio driver
          issues)
        - Fail to clean up audio device connections properly
        """
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interrupt signals gracefully."""
        if self.is_single_mode_recording:
            # For single mode, set flag to stop recording loop
            print(f"\n🛑 Received signal {signum}, stopping single mode recording...")
            self.is_single_mode_recording = False
            return
        elif not self.is_recording:
            # Already stopping, exit immediately
            sys.exit(0)

        print(f"\n\n🛑 Received signal {signum}, stopping recording...")
        self.stop_recording()

    def stop_recording(self) -> None:
        """
        Stop microphone recording gracefully.

        CRITICAL: This method ensures proper cleanup of audio resources and
        prevents resource leaks. It's called by signal handlers and during
        normal shutdown to:

        1. Resource Management: Sets recording flag to False, signaling the
           continuous recognition to stop
        2. Thread Cleanup: Uses the built-in stop_continuous_recognition method
           which properly manages threads
        3. Audio Device Cleanup: Allows the SpeechRecognizer to properly
           release microphone access and close audio streams
        4. System Stability: Prevents orphaned threads that could consume
           CPU and memory resources
        5. User Experience: Provides clear feedback about the shutdown
           process status
        """
        if not self.is_recording:
            return

        self.is_recording = False
        if self.recognizer and hasattr(self.recognizer, "stop_continuous_recognition"):
            print("⏳ Stopping continuous recognition...")
            self.recognizer.stop_continuous_recognition()
            print("✅ Continuous recognition stopped")

    def setup_speech_config(
        self, model: Optional[str] = None, target_language: str = "en"
    ) -> SpeechConfig:
        """
        Setup speech configuration.

        CRITICAL: This method ensures proper configuration of the speech
        recognition system. It's called by the main application to:

        1. Language Auto-Detection: Uses Groq API to automatically detect
           the language of the speech input
        2. Translation Configuration: Sets the target language for translation
           if specified
        3. Model Selection: Handles model selection if specified
        4. Error Handling: Provides clear error messages if configuration fails
        """
        config = SpeechConfig()
        # Language auto-detected by Groq API
        if model:
            # Note: SpeechConfig doesn't have a model attribute
            # Model is handled by the SpeechRecognizer
            pass
        # Set target language for translation
        config.set_translation_target_language(target_language)
        return config

    def validate_api_key(self) -> bool:
        """Validate that the API key is configured."""
        try:
            api_key = Config.get_api_key()
            if not api_key:
                print("❌ Error: GROQ_API_KEY not configured")
                print(
                    "Please set the GROQ_API_KEY environment variable or "
                    "add it to your .env file"
                )
                return False
            print("✅ API key validated")
            return True
        except Exception as e:
            print(f"❌ Error validating API key: {e}")
            return False

    def print_available_models(self):
        """Print available Groq models."""
        models = [
            {
                "id": "whisper-large-v3",
                "name": "Whisper Large V3",
                "description": (
                    "High quality, supports transcription and " "translation"
                ),
            },
            {
                "id": "whisper-large-v3-turbo",
                "name": "Whisper Large V3 Turbo",
                "description": ("Fast transcription only " "(no translation)"),
            },
        ]

        print("\n📋 Available Models:")
        for model in models:
            print(f"  • {model['name']} ({model['id']}) - " f"{model['description']}")

    def print_available_languages(self):
        """Print available languages."""
        print("\n🌍 Language Support:")
        print("  • Groq API automatically detects the language")
        print("  • Supports all major languages including:")
        print("    - English, Spanish, French, German")
        print("    - Italian, Portuguese, Russian")
        print("    - Japanese, Korean, Chinese")
        print("    - And many more...")
        print("  • No need to specify language - just speak naturally!")

    def recognize_from_file(
        self,
        file_path: str,
        is_translation: bool = False,
        target_language: str = "en",
    ) -> None:
        """Recognize speech from an audio file."""
        try:
            print(f"\n🎵 Processing audio file: {file_path}")

            # Setup speech configuration
            config = self.setup_speech_config(target_language=target_language)
            if is_translation:
                config.enable_translation = True
                print(f"🔀 Translation mode enabled " f"(target: {target_language})")

            # Create recognizer
            self.recognizer = SpeechRecognizer(config)

            # Perform recognition
            print("🎤 Starting recognition...")
            start_time = time.time()

            result = self.recognizer.recognize_once_async()

            processing_time = time.time() - start_time

            # Process result
            if result.reason == ResultReason.RecognizedSpeech:
                print("\n✅ Recognition successful!")
                print(f"📝 Text: {result.text}")
                print(f"🎯 Confidence: {result.confidence:.2f}")
                print(f"🌍 Language: {result.language}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")

                if result.timestamps:
                    print(f"📊 Word timestamps: {len(result.timestamps)} words")

            elif result.reason == ResultReason.NoMatch:
                print("\n❌ No speech detected in the audio file")

            elif result.reason == ResultReason.Canceled:
                if result.cancellation_details:
                    print(
                        f"\n❌ Recognition canceled: "
                        f"{result.cancellation_details.error_details}"
                    )
                else:
                    print("\n❌ Recognition canceled")

        except Exception as e:
            print(f"\n❌ Error during recognition: {e}")

    def recognize_from_microphone(
        self,
        is_translation: bool = False,
        target_language: str = "en",
        recognition_mode: str = "continuous",
    ) -> None:
        """Recognize speech from microphone input using single or continuous recognition."""
        try:
            print("\n🎤 Starting microphone recognition...")
            print("💡 Speak into your microphone (Press Ctrl+C to stop)")

            # Setup speech configuration
            config = self.setup_speech_config(target_language=target_language)
            if is_translation:
                config.enable_translation = True
                print(f"🔀 Translation mode enabled " f"(target: {target_language})")

            # Create recognizer
            self.recognizer = SpeechRecognizer(config)

            if recognition_mode == "single":
                print("🎯 Single recognition mode - speak once and get result")
                print("🎤 Listening for speech... (Press Ctrl+C to stop)")

                # For single mode, we need to capture audio manually and then process it
                # This avoids the continuous recognition logic
                if not self.audio_config:
                    self.audio_config = AudioConfig()

                try:
                    with self.audio_config as audio:
                        print("🎤 Starting microphone stream...")
                        audio.start_microphone_stream()
                        print("🎤 Recording audio... (Press Ctrl+C to stop recording)")

                        # Set single mode recording flag
                        self.is_single_mode_recording = True

                        # Collect audio chunks until user stops
                        audio_chunks = []
                        start_time = time.time()
                        max_duration = 120  # 2 minutes max

                        while (
                            time.time() - start_time < max_duration
                            and self.is_single_mode_recording
                        ):
                            try:
                                # Check if we should stop recording
                                if not self.is_single_mode_recording:
                                    print("\n🛑 Recording stopped by signal handler")
                                    break

                                chunk = audio.read_audio_chunk(1024)
                                if chunk and len(chunk) > 0:
                                    audio_chunks.append(chunk)
                                    print(".", end="", flush=True)  # Show progress

                                # Small delay to allow signal handling to work
                                time.sleep(0.01)

                            except Exception as e:
                                print(f"\n❌ Error reading audio: {e}")
                                break

                        print()  # New line after progress dots

                        if not audio_chunks:
                            print("❌ No audio captured")
                            return

                        # Combine audio chunks and convert to numpy array
                        combined_audio = b"".join(audio_chunks)
                        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                        audio_array_float = audio_array.astype(np.float32) / 32768.0

                        print(
                            f"🎵 Processed {len(audio_chunks)} audio chunks ({len(combined_audio)} bytes)"
                        )
                        print("🔄 Sending audio to Groq API for recognition...")

                        # Process the audio data
                        result = self.recognizer.recognize_audio_data(audio_array_float)

                        # Display result
                        if result.reason == ResultReason.RecognizedSpeech:
                            print("\n✅ Recognition successful!")
                            print(f"📝 Text: {result.text}")
                            print(f"🎯 Confidence: {result.confidence:.2f}")
                            print(f"🌍 Language: {result.language}")

                            if result.timestamps:
                                print(
                                    f"📊 Word timestamps: {len(result.timestamps)} words"
                                )

                        elif result.reason == ResultReason.NoMatch:
                            print("\n❌ No speech detected")

                        elif result.reason == ResultReason.Canceled:
                            if result.cancellation_details:
                                print(
                                    f"\n❌ Recognition canceled: "
                                    f"{result.cancellation_details.error_details}"
                                )
                            else:
                                print("\n❌ Recognition canceled")

                except Exception as e:
                    print(f"\n❌ Error in single recognition: {e}")
                    return
                finally:
                    # Always reset the flag
                    self.is_single_mode_recording = False

            else:  # continuous mode
                print("🔄 Continuous recognition mode - speak continuously")
                print("🎤 Listening for speech... (Press Ctrl+C to stop)")

                # Set recording state for continuous mode
                self.is_recording = True

                # Set up event handlers for real-time results
                self.recognizer.connect("recognized", self._on_recognized)
                self.recognizer.connect("canceled", self._on_canceled)
                self.recognizer.connect("session_started", self._on_session_started)
                self.recognizer.connect("session_stopped", self._on_session_stopped)

                # Start continuous recognition
                self.recognizer.start_continuous_recognition()

                # Keep the main thread alive while recognition is running
                while self.is_recording:
                    time.sleep(0.1)  # Small delay to prevent busy waiting

        except Exception as e:
            print(f"\n❌ Error during microphone recognition: {e}")
            self.stop_recording()

    def _on_recognized(self, result):
        """Handle recognized speech events."""
        if result.reason == ResultReason.RecognizedSpeech:
            print(f"\n📝 {result.text}")
            if result.confidence < 0.8:
                print(f"⚠️  Low confidence: {result.confidence:.2f}")

            # Handle language display for translation vs transcription
            if hasattr(self.recognizer, "speech_config") and getattr(
                self.recognizer.speech_config, "enable_translation", False
            ):
                # Translation mode - show that source language was auto-detected
                print("🌍 Source language: Auto-detected by Groq API")
                print("🇺🇸 Translated to: English")
            else:
                # Transcription mode - show detected language
                if result.language:
                    print(f"🌍 Detected language: {result.language}")

            print("🎤 Listening for next speech...")

    def _on_canceled(self, result):
        """Handle canceled recognition events."""
        if result.cancellation_details:
            print(
                f"\n❌ Recognition canceled: "
                f"{result.cancellation_details.error_details}"
            )
        else:
            print("\n❌ Recognition canceled")

    def _on_session_started(self, event):
        """Handle session started events."""
        print("🎬 Recognition session started")

    def _on_session_stopped(self, event):
        """Handle session stopped events."""
        print("🏁 Recognition session stopped")

    def print_usage_examples(self):
        """Print usage examples."""
        print("\n📖 Usage Examples:")
        print("  # Quick start (default transcription, continuous mode):")
        print("  python cli_speech_recognition.py")
        print()
        print("  # Transcribe from microphone (continuous mode):")
        print("  python cli_speech_recognition.py --mode transcription")
        print()
        print("  # Transcribe from microphone (single mode):")
        print(
            "  python cli_speech_recognition.py --mode transcription --recognition-mode single"
        )
        print()
        print("  # Translate from microphone (continuous mode):")
        print(
            "  python cli_speech_recognition.py --mode translation --target-language en"
        )
        print()
        print("  # Single translation mode:")
        print(
            "  python cli_speech_recognition.py --mode translation --recognition-mode single"
        )
        print()
        print("  # Transcribe from audio file:")
        print("  python cli_speech_recognition.py --file audio.wav")
        print()
        print("  # List available options:")
        print("  python cli_speech_recognition.py --list-models")
        print("  python cli_speech_recognition.py --list-languages")
        print("  python cli_speech_recognition.py --help")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="CLI Speech Recognition using Groq Speech SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-shot transcription (speak once, get result)
  python cli_speech_recognition.py --mode transcription --recognition-mode single
  
  # Continuous transcription (speak continuously, real-time results)
  python cli_speech_recognition.py --mode transcription --recognition-mode continuous
  
  # Single-shot translation to Spanish
  python cli_speech_recognition.py --mode translation --recognition-mode single --target-language es
  
  # File-based recognition
  python cli_speech_recognition.py --file audio.wav
  
  # List available models and languages
  python cli_speech_recognition.py --list-models
  python cli_speech_recognition.py --list-languages
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["transcription", "translation"],
        help="Operation mode: transcription (speech-to-text) or translation (speech-to-text in target language)",
    )

    parser.add_argument(
        "--recognition-mode",
        choices=["single", "continuous"],
        default="continuous",
        help="Recognition mode: single (record once, process, show result) or continuous (real-time streaming results)",
    )

    parser.add_argument(
        "--target-language",
        default="en",
        help="Target language code for translation (e.g., 'es' for Spanish, 'fr' for French)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Audio file path for recognition",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Groq model to use",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )

    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available languages",
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = CLISpeechRecognition()

    # Validate API key
    if not cli.validate_api_key():
        sys.exit(1)

    # Handle list commands
    if args.list_models:
        cli.print_available_models()
        return

    if args.list_languages:
        cli.print_available_languages()
        return

    # Validate arguments
    if not args.mode and not args.file:
        print("🎤 Groq Speech Recognition CLI")
        print("=" * 40)
        print("✅ API key validated")
        print()
        print("📝 No mode specified. Starting default transcription mode...")
        print("💡 Tip: Use --help to see all available options")
        print()

        # Set default mode to transcription
        args.mode = "transcription"
        print(f"🎯 Default mode: {args.mode}")
        print("🌍 Language: Auto-detected by Groq API")
        print()
        print("🎤 Starting microphone transcription...")
        print("💡 Speak into your microphone (Press Ctrl+C to stop)")
        print()

    if args.mode and args.file:
        print("❌ Error: Cannot use both --mode and --file")
        sys.exit(1)

    # Determine if translation mode
    is_translation = args.mode == "translation"

    try:
        if args.file:
            # File-based recognition
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ Error: File not found: {args.file}")
                sys.exit(1)

            # Pass target language for translation
            target_lang = args.target_language if is_translation else "en"
            cli.recognize_from_file(str(file_path), is_translation, target_lang)
        else:
            # Microphone-based recognition
            # Pass target language for translation
            target_lang = args.target_language if is_translation else "en"
            cli.recognize_from_microphone(
                is_translation, target_lang, args.recognition_mode
            )

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🎤 Groq Speech Recognition CLI")
    print("=" * 40)

    main()
