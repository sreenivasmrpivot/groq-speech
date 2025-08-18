#!/usr/bin/env python3
"""
Test script to verify language detection is working correctly.
This tests the exact same audio format and parameters that the frontend sends.
"""

import requests
import base64
import numpy as np


def test_language_detection():
    """Test language detection with frontend-like audio data."""
    try:
        print("🧪 Testing Language Detection Fix...")
        print("=" * 60)

        # Create audio data exactly like the frontend sends
        # 16kHz sample rate, mono, 3 seconds duration
        sample_rate = 16000
        duration = 3.0
        samples = int(sample_rate * duration)

        # Create a simple audio pattern (not just silence)
        # This simulates what the frontend might send
        t = np.linspace(0, duration, samples, False)
        # Create a simple tone to simulate speech
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz tone
        audio_data = audio_data.astype(np.float32)

        # Convert to int16 like the frontend does
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Convert to base64 like the frontend does
        audio_bytes = audio_data_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        print(f"🎵 Created test audio: {samples} samples, {len(audio_bytes)} bytes")
        print(f"📊 Sample rate: {sample_rate} Hz (matches CLI AudioConfig)")
        print(f"📊 Duration: {duration} seconds")
        print(f"📊 Base64 length: {len(audio_base64)} characters")

        # Send to API with the exact same parameters as frontend
        url = "http://localhost:8000/api/v1/recognize"
        payload = {
            "audio_data": audio_base64,
            "enable_timestamps": True,
            "enable_language_detection": True,
            "target_language": "en",
        }

        print(f"\n📤 Sending request to API with frontend parameters...")
        print(f"📤 URL: {url}")
        print(f"📤 Payload keys: {list(payload.keys())}")
        print(f"📤 Enable language detection: {payload['enable_language_detection']}")

        response = requests.post(url, json=payload)

        print(f"\n📥 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recognition successful!")
            print(f"📝 Text: '{result.get('text', 'N/A')}'")
            print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
            print(f"🌍 Language: {result.get('language', 'N/A')}")
            print(f"🔍 Success: {result.get('success', 'N/A')}")

            # Check if language detection is working
            if result.get("language") == "English":
                print("✅ Language detection working correctly (English detected)")
                print("🎯 The frontend should now work correctly!")
            else:
                print(
                    f"❌ Language detection issue: Expected 'English', got '{result.get('language')}'"
                )
                print(
                    "🔍 This suggests there's still an issue with the API or audio processing"
                )

        else:
            print(f"❌ Recognition failed: {response.text}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run the test."""
    print("🎯 Language Detection Fix Test")
    print("=" * 60)
    print("This test verifies that the API now correctly detects English language")
    print("using the same audio format and parameters that the frontend sends.")
    print()

    test_language_detection()


if __name__ == "__main__":
    main()
