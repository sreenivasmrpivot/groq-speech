#!/usr/bin/env python3
"""
CLI Speech Recognition Example using Groq Speech SDK

This example demonstrates how to use the groq_speech library directly
for speech recognition and translation from the command line.

Usage:
    python cli_speech_recognition.py --mode transcription --language en-US
    python cli_speech_recognition.py --mode translation --language es-ES
    python cli_speech_recognition.py --file audio.wav --language en-US
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason
from groq_speech.config import Config


class CLISpeechRecognition:
    """CLI-based speech recognition using Groq Speech SDK."""

    def __init__(self):
        self.recognizer: Optional[SpeechRecognizer] = None
        self.is_recording = False

    def setup_speech_config(
        self, model: Optional[str] = None, target_language: str = "en"
    ) -> SpeechConfig:
        """Setup speech configuration."""
        config = SpeechConfig()
        # Language auto-detected by Groq API
        if model:
            config.model = model
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
                    "Please set the GROQ_API_KEY environment variable or add it to your .env file"
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
                "description": "High quality, supports transcription and translation",
            },
            {
                "id": "whisper-large-v3-turbo",
                "name": "Whisper Large V3 Turbo",
                "description": "Fast transcription only (no translation)",
            },
        ]

        print("\n📋 Available Models:")
        for model in models:
            print(f"  • {model['name']} ({model['id']}) - {model['description']}")

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

    async def recognize_from_file(
        self, file_path: str, is_translation: bool = False, target_language: str = "en"
    ) -> None:
        """Recognize speech from an audio file."""
        try:
            print(f"\n🎵 Processing audio file: {file_path}")

            # Setup speech configuration
            config = self.setup_speech_config(target_language=target_language)
            if is_translation:
                config.enable_translation = True
                print(f"🔀 Translation mode enabled (target: {target_language})")

            # Create recognizer
            self.recognizer = SpeechRecognizer(config)

            # Perform recognition
            print("🎤 Starting recognition...")
            start_time = time.time()

            result = self.recognizer.recognize_once_async()

            processing_time = time.time() - start_time

            # Process result
            if result.reason == ResultReason.RecognizedSpeech:
                print(f"\n✅ Recognition successful!")
                print(f"📝 Text: {result.text}")
                print(f"🎯 Confidence: {result.confidence:.2f}")
                print(f"🌍 Language: {result.language}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")

                if result.timestamps:
                    print(f"📊 Word timestamps: {len(result.timestamps)} words")

            elif result.reason == ResultReason.NoMatch:
                print("\n❌ No speech detected in the audio file")

            elif result.reason == ResultReason.Canceled:
                print(
                    f"\n❌ Recognition canceled: {result.cancellation_details.error_details}"
                )

        except Exception as e:
            print(f"\n❌ Error during recognition: {e}")

    async def recognize_from_microphone(
        self, is_translation: bool = False, target_language: str = "en"
    ) -> None:
        """Recognize speech from microphone input."""
        try:
            print(f"\n🎤 Starting microphone recognition...")
            print("💡 Speak into your microphone (Press Ctrl+C to stop)")

            # Setup speech configuration
            config = self.setup_speech_config(target_language=target_language)
            if is_translation:
                config.enable_translation = True
                print(f"🔀 Translation mode enabled (target: {target_language})")

            # Create recognizer
            self.recognizer = SpeechRecognizer(config)
            self.is_recording = True

            # Start continuous recognition
            while self.is_recording:
                try:
                    result = self.recognizer.recognize_once_async()

                    if result.reason == ResultReason.RecognizedSpeech:
                        print(f"\n📝 {result.text}")
                        if result.confidence < 0.8:
                            print(f"⚠️  Low confidence: {result.confidence:.2f}")

                    elif result.reason == ResultReason.NoMatch:
                        print(".", end="", flush=True)

                    elif result.reason == ResultReason.Canceled:
                        print(
                            f"\n❌ Recognition canceled: {result.cancellation_details.error_details}"
                        )
                        break

                except KeyboardInterrupt:
                    print("\n\n🛑 Stopping recognition...")
                    self.is_recording = False
                    break

        except Exception as e:
            print(f"\n❌ Error during microphone recognition: {e}")

    def print_usage_examples(self):
        """Print usage examples."""
        print("\n📖 Usage Examples:")
        print("  # Quick start (default transcription):")
        print("  python cli_speech_recognition.py")
        print()
        print("  # Transcribe from microphone:")
        print("  python cli_speech_recognition.py --mode transcription")
        print()
        print("  # Translate from microphone:")
        print(
            "  python cli_speech_recognition.py --mode translation --target-language en"
        )
        print()
        print("  # Transcribe from audio file:")
        print("  python cli_speech_recognition.py --file audio.wav")
        print()
        print("  # List available options:")
        print("  python cli_speech_recognition.py --list-models")
        print("  python cli_speech_recognition.py --list-languages")
        print("  python cli_speech_recognition.py --help")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="CLI Speech Recognition using Groq Speech SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_speech_recognition.py --mode transcription
  python cli_speech_recognition.py --mode translation --target-language en
  python cli_speech_recognition.py --file audio.wav
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["transcription", "translation"],
        help="Recognition mode (transcription or translation)",
    )

    parser.add_argument(
        "--target-language",
        default="en",
        help="Target language for translation (default: en)",
    )

    # Language parameter removed - Groq API auto-detects language

    parser.add_argument("--file", type=str, help="Audio file path for recognition")

    parser.add_argument("--model", type=str, help="Groq model to use")

    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )

    parser.add_argument(
        "--list-languages", action="store_true", help="List available languages"
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
            await cli.recognize_from_file(str(file_path), is_translation, target_lang)
        else:
            # Microphone-based recognition
            # Pass target language for translation
            target_lang = args.target_language if is_translation else "en"
            await cli.recognize_from_microphone(is_translation, target_lang)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🎤 Groq Speech Recognition CLI")
    print("=" * 40)

    asyncio.run(main())
