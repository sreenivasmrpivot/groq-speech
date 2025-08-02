#!/usr/bin/env python3
"""
Quick test script to verify transcription accuracy improvements.
"""

import os
import sys
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, Config


def test_basic_recognition():
    """Test basic speech recognition with improvements."""

    print("🧪 Testing Basic Speech Recognition")
    print("=" * 40)

    try:
        # Validate API key
        Config.get_api_key()
        print("✅ API key validated")

        # Initialize speech recognizer
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config=speech_config)

        print("\n🎤 Speak a simple phrase when prompted...")
        input("Press Enter when ready to speak...")

        # Perform recognition
        print("Listening...")
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognized: '{result.text}'")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Language: {result.language}")

            # Check for common issues
            text_lower = result.text.lower()
            if len(result.text.strip()) == 0:
                print("❌ Empty transcription")
                return False
            elif len(result.text.split()) < 2:
                print("⚠️  Very short transcription")
            else:
                print("✅ Good transcription length")

            return True
        else:
            print(f"❌ Recognition failed: {result.reason}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_continuous_recognition():
    """Test continuous recognition with improvements."""

    print("\n🔄 Testing Continuous Recognition")
    print("=" * 40)

    try:
        # Initialize speech recognizer
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config=speech_config)

        transcripts = []

        def on_recognized(result):
            if result.reason == ResultReason.RecognizedSpeech:
                transcript = {
                    "text": result.text,
                    "confidence": result.confidence,
                    "timestamp": time.time(),
                }
                transcripts.append(transcript)
                print(f"🎤 '{result.text}' (confidence: {result.confidence:.2f})")

        # Connect event handler
        recognizer.connect("recognized", on_recognized)

        print("🎤 Starting continuous recognition...")
        print("Speak several phrases with pauses between them.")
        print("Press Ctrl+C to stop.")

        try:
            # Start continuous recognition
            recognizer.start_continuous_recognition()

            # Run for 15 seconds
            start_time = time.time()
            while time.time() - start_time < 15:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping continuous recognition...")
        finally:
            recognizer.stop_continuous_recognition()

        # Print results
        print(f"\n📊 Results:")
        print(f"  Total segments: {len(transcripts)}")

        if transcripts:
            avg_confidence = sum(t["confidence"] for t in transcripts) / len(
                transcripts
            )
            print(f"  Average confidence: {avg_confidence:.2f}")

            print("\n📝 Transcripts:")
            for i, transcript in enumerate(transcripts, 1):
                print(f"  {i}. '{transcript['text']}'")

        return len(transcripts) > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Quick Transcription Accuracy Test")
    print("=" * 50)

    # Test basic recognition
    success1 = test_basic_recognition()

    # Test continuous recognition
    success2 = test_continuous_recognition()

    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"Basic Recognition: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Continuous Recognition: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1 and success2:
        print("\n🎉 All tests passed! Improvements are working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")

    print("\n💡 Tips for better accuracy:")
    print("  - Speak clearly and at a normal pace")
    print("  - Minimize background noise")
    print("  - Use a good quality microphone")
    print("  - Pause briefly between phrases")


if __name__ == "__main__":
    main()
