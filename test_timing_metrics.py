#!/usr/bin/env python3
"""
Test script to demonstrate timing metrics for transcription pipeline.
"""

import os
import sys
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, Config


def test_timing_metrics():
    """Test timing metrics for transcription pipeline."""

    print("🧪 Testing Timing Metrics")
    print("=" * 50)

    try:
        # Validate API key
        Config.get_api_key()
        print("✅ API key validated")

        # Initialize speech recognizer
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config=speech_config)

        print("\n🎤 Speak a phrase when prompted...")
        input("Press Enter when ready to speak...")

        # Perform recognition
        print("Listening...")
        result = recognizer.recognize_once_async()

        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognized: '{result.text}'")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Language: {result.language}")

            # Display timing metrics
            if result.timing_metrics:
                timing = result.timing_metrics.get_metrics()
                print("\n⏱️  TIMING METRICS:")
                print("=" * 30)

                for key, value in timing.items():
                    if value is not None:
                        print(
                            f"   {key.replace('_', ' ').title()}: {value * 1000:.1f}ms"
                        )

                # Calculate percentages
                total_time = timing.get("total_time", 0)
                if total_time > 0:
                    print("\n📊 TIMING BREAKDOWN:")
                    print("=" * 30)

                    mic_time = timing.get("microphone_capture", 0)
                    api_time = timing.get("api_call", 0)
                    proc_time = timing.get("response_processing", 0)

                    print(
                        f"   Microphone Capture: {mic_time/total_time*100:.1f}% ({mic_time*1000:.1f}ms)"
                    )
                    print(
                        f"   API Call: {api_time/total_time*100:.1f}% ({api_time*1000:.1f}ms)"
                    )
                    print(
                        f"   Response Processing: {proc_time/total_time*100:.1f}% ({proc_time*1000:.1f}ms)"
                    )
                    print(f"   Total Time: {total_time*1000:.1f}ms")

                    # Performance analysis
                    print("\n📈 PERFORMANCE ANALYSIS:")
                    print("=" * 30)

                    if api_time > total_time * 0.7:
                        print(
                            "   ⚠️  API call is taking most of the time (network bottleneck)"
                        )
                    elif mic_time > total_time * 0.5:
                        print("   ⚠️  Microphone capture is taking significant time")
                    else:
                        print("   ✅ Timing distribution looks good")

                    if total_time > 5.0:
                        print("   ⚠️  Total time is quite high (>5s)")
                    elif total_time < 1.0:
                        print("   ✅ Very fast response (<1s)")
                    else:
                        print("   ✅ Reasonable response time")

            else:
                print("❌ No timing metrics available")

            return True
        else:
            print(f"❌ Recognition failed: {result.reason}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_continuous_timing():
    """Test continuous recognition with timing metrics."""

    print("\n🔄 Testing Continuous Recognition Timing")
    print("=" * 50)

    try:
        # Initialize speech recognizer
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config=speech_config)

        transcripts = []
        timing_data = []

        def on_recognized(result):
            if result.reason == ResultReason.RecognizedSpeech:
                transcript = {
                    "text": result.text,
                    "confidence": result.confidence,
                    "timestamp": time.time(),
                }

                if result.timing_metrics:
                    timing = result.timing_metrics.get_metrics()
                    transcript["timing"] = timing
                    timing_data.append(timing)

                transcripts.append(transcript)
                print(f"🎤 '{result.text}' (confidence: {result.confidence:.2f})")

                if result.timing_metrics:
                    timing = result.timing_metrics.get_metrics()
                    total_time = timing.get("total_time", 0) * 1000
                    print(f"   ⏱️  Total time: {total_time:.1f}ms")

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
        print(f"\n📊 Continuous Recognition Results:")
        print(f"  Total segments: {len(transcripts)}")

        if timing_data:
            total_times = [t.get("total_time", 0) * 1000 for t in timing_data]
            api_times = [t.get("api_call", 0) * 1000 for t in timing_data]
            mic_times = [t.get("microphone_capture", 0) * 1000 for t in timing_data]

            print(f"  Average total time: {sum(total_times)/len(total_times):.1f}ms")
            print(f"  Average API time: {sum(api_times)/len(api_times):.1f}ms")
            print(f"  Average mic time: {sum(mic_times)/len(mic_times):.1f}ms")
            print(f"  Min total time: {min(total_times):.1f}ms")
            print(f"  Max total time: {max(total_times):.1f}ms")

        return len(transcripts) > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Timing Metrics Test Suite")
    print("=" * 50)

    # Test single recognition timing
    success1 = test_timing_metrics()

    # Test continuous recognition timing
    success2 = test_continuous_timing()

    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"Single Recognition Timing: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Continuous Recognition Timing: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1 and success2:
        print("\n🎉 All timing tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")

    print("\n💡 Timing Analysis Tips:")
    print("  - API call time indicates network performance")
    print("  - Microphone time indicates audio capture efficiency")
    print("  - Processing time indicates response parsing speed")
    print("  - Total time should ideally be under 3 seconds")


if __name__ == "__main__":
    main()
