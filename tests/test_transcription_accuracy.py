#!/usr/bin/env python3
"""
Test transcription accuracy improvements.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, Config


def test_transcription_accuracy():
    """Test transcription accuracy with various audio inputs."""

    print("🧪 Testing Transcription Accuracy Improvements")
    print("=" * 50)

    # Validate API key
    try:
        Config.get_api_key()
        print("✅ API key validated")
    except ValueError as e:
        print(f"❌ API key error: {e}")
        return False

    # Initialize speech recognizer
    speech_config = SpeechConfig()
    recognizer = SpeechRecognizer(speech_config=speech_config)

    # Test cases with expected transcriptions
    test_cases = [
        {
            "description": "Simple phrase",
            "expected_keywords": ["hello", "world"],
            "prompt": "Say 'Hello world' clearly",
        },
        {
            "description": "Numbers",
            "expected_keywords": ["one", "two", "three"],
            "prompt": "Count 'One two three'",
        },
        {
            "description": "Common words",
            "expected_keywords": ["the", "and", "is"],
            "prompt": "Say 'The cat and dog is here'",
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Prompt: {test_case['prompt']}")

        # Get user input
        input(f"Press Enter when ready to speak...")

        try:
            # Perform recognition
            print("🎤 Listening...")
            result = recognizer.recognize_once_async()

            if result.reason == ResultReason.RecognizedSpeech:
                print(f"✅ Recognized: '{result.text}'")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Language: {result.language}")

                # Check for expected keywords
                text_lower = result.text.lower()
                found_keywords = []
                for keyword in test_case["expected_keywords"]:
                    if keyword.lower() in text_lower:
                        found_keywords.append(keyword)

                accuracy = len(found_keywords) / len(test_case["expected_keywords"])
                print(
                    f"   Accuracy: {accuracy:.2f} ({len(found_keywords)}/{len(test_case['expected_keywords'])} keywords)"
                )

                results.append(
                    {
                        "test": test_case["description"],
                        "recognized": result.text,
                        "confidence": result.confidence,
                        "accuracy": accuracy,
                        "found_keywords": found_keywords,
                    }
                )

            else:
                print(f"❌ Recognition failed: {result.reason}")
                results.append(
                    {
                        "test": test_case["description"],
                        "recognized": "",
                        "confidence": 0.0,
                        "accuracy": 0.0,
                        "found_keywords": [],
                    }
                )

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {
                    "test": test_case["description"],
                    "recognized": "",
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "found_keywords": [],
                }
            )

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TRANSCRIPTION ACCURACY SUMMARY")
    print("=" * 50)

    total_accuracy = 0.0
    total_confidence = 0.0
    successful_tests = 0

    for result in results:
        print(f"\n{result['test']}:")
        print(f"  Recognized: '{result['recognized']}'")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.2f}")
        print(f"  Keywords found: {result['found_keywords']}")

        if result["confidence"] > 0:
            total_accuracy += result["accuracy"]
            total_confidence += result["confidence"]
            successful_tests += 1

    if successful_tests > 0:
        avg_accuracy = total_accuracy / successful_tests
        avg_confidence = total_confidence / successful_tests

        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Average Accuracy: {avg_accuracy:.2f}")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Successful Tests: {successful_tests}/{len(results)}")

        if avg_accuracy >= 0.8:
            print("✅ Excellent transcription accuracy!")
        elif avg_accuracy >= 0.6:
            print("✅ Good transcription accuracy")
        elif avg_accuracy >= 0.4:
            print("⚠️  Moderate transcription accuracy")
        else:
            print("❌ Poor transcription accuracy - needs improvement")
    else:
        print("❌ No successful recognitions")

    return True


def test_continuous_recognition():
    """Test continuous recognition accuracy."""

    print("\n🔄 Testing Continuous Recognition")
    print("=" * 50)

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

        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⏹️  Stopping continuous recognition...")
    finally:
        recognizer.stop_continuous_recognition()

    # Print results
    print(f"\n📊 Continuous Recognition Results:")
    print(f"  Total segments: {len(transcripts)}")

    if transcripts:
        avg_confidence = sum(t["confidence"] for t in transcripts) / len(transcripts)
        print(f"  Average confidence: {avg_confidence:.2f}")

        print("\n📝 Transcripts:")
        for i, transcript in enumerate(transcripts, 1):
            print(f"  {i}. '{transcript['text']}'")

    return len(transcripts) > 0


def main():
    """Main test function."""
    print("🧪 Transcription Accuracy Test Suite")
    print("=" * 50)

    # Test single recognition
    success1 = test_transcription_accuracy()

    # Test continuous recognition
    success2 = test_continuous_recognition()

    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"Single Recognition: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Continuous Recognition: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1 and success2:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
