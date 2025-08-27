#!/usr/bin/env python3
"""
Simple test script to check if basic transcription works.
"""

import os
import sys
import tempfile
import soundfile as sf
import numpy as np

# Add the groq_speech directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "groq_speech"))

from groq_speech.speech_recognizer import SpeechRecognizer
from groq_speech.speech_config import SpeechConfig


def test_transcription():
    """Test basic transcription functionality."""
    print("🧪 Testing Basic Transcription")
    print("=" * 40)

    try:
        # Create a speech recognizer
        print("1. Creating speech recognizer...")
        speech_config = SpeechConfig()
        recognizer = SpeechRecognizer(speech_config)
        print("✅ Speech recognizer created")

        # Create a simple test audio file (2 seconds of audio)
        print("\n2. Creating test audio file...")
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        samples = int(sample_rate * duration)

        # Generate some test audio (sine wave)
        t = np.linspace(0, duration, samples, False)
        audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            sf.write(temp_filename, audio_data, sample_rate, format="WAV")
            print(f"✅ Test audio file created: {temp_filename}")

            try:
                # Test basic transcription
                print("\n3. Testing basic transcription...")

                # Try to get audio data from the recognizer
                if hasattr(recognizer, "audio_config") and recognizer.audio_config:
                    audio_data_for_api = recognizer.audio_config.get_file_audio_data(
                        temp_filename
                    )
                    print(
                        f"✅ Audio config available, audio data: {type(audio_data_for_api)}"
                    )
                else:
                    print("⚠️ No audio config available")
                    audio_data_for_api = None

                # Try direct transcription
                basic_result = recognizer.recognize_audio_data(audio_data_for_api)

                if basic_result and basic_result.text:
                    print(f"✅ Basic transcription works: {basic_result.text}")
                else:
                    print("⚠️ Basic transcription returned no text")
                    if basic_result:
                        print(f"   Result object: {basic_result}")
                        print(f"   Has text attribute: {hasattr(basic_result, 'text')}")
                        print(
                            f"   Text value: {getattr(basic_result, 'text', 'NO_TEXT_ATTR')}"
                        )
                    else:
                        print("   Result is None")

            finally:
                # Clean up
                try:
                    os.unlink(temp_filename)
                    print(f"\n🧹 Cleaned up test file")
                except Exception:
                    pass

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_transcription()
