#!/usr/bin/env python3
"""
Demo script for Groq Speech SDK.
This script demonstrates basic speech recognition functionality.
"""

import os
import sys
import time

# Add the current directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_speech import (
    SpeechConfig, 
    SpeechRecognizer, 
    AudioConfig, 
    ResultReason, 
    CancellationReason,
    PropertyId,
    Config
)


def demo_microphone_recognition():
    """Demonstrate microphone-based speech recognition."""
    print("=== Microphone Recognition Demo ===")
    
    try:
        # Create speech configuration (uses .env settings)
        speech_config = SpeechConfig()
        
        # Create speech recognizer
        speech_recognizer = SpeechRecognizer(speech_config=speech_config)
        
        print("Speak into your microphone...")
        print("(The recognition will start automatically and stop after detecting silence)")
        
        # Perform recognition
        result = speech_recognizer.recognize_once_async()
        
        # Handle the result
        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognized: {result.text}")
            if result.confidence > 0:
                print(f"   Confidence: {result.confidence:.2f}")
            if result.language:
                print(f"   Language: {result.language}")
            return True
        elif result.reason == ResultReason.NoMatch:
            print("❌ No speech could be recognized")
            if result.no_match_details:
                print(f"   Details: {result.no_match_details.error_details}")
            return False
        elif result.reason == ResultReason.Canceled:
            print("❌ Speech recognition was canceled")
            if result.cancellation_details:
                print(f"   Reason: {result.cancellation_details.reason}")
                if result.cancellation_details.error_details:
                    print(f"   Error: {result.cancellation_details.error_details}")
            return False
        
    except Exception as e:
        print(f"❌ Error during recognition: {e}")
        return False


def demo_language_change():
    """Demonstrate changing the recognition language."""
    print("\n=== Language Change Demo ===")
    
    try:
        # Create speech configuration with German language (uses .env settings)
        speech_config = SpeechConfig()
        speech_config.speech_recognition_language = "de-DE"
        
        # Create speech recognizer
        speech_recognizer = SpeechRecognizer(speech_config=speech_config)
        
        print("Speak in German into your microphone...")
        
        # Perform recognition
        result = speech_recognizer.recognize_once_async()
        
        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognized (German): {result.text}")
            return True
        elif result.reason == ResultReason.NoMatch:
            print("❌ No speech could be recognized")
            return False
        elif result.reason == ResultReason.Canceled:
            print("❌ Speech recognition was canceled")
            return False
        
    except Exception as e:
        print(f"❌ Error during recognition: {e}")
        return False


def demo_semantic_segmentation():
    """Demonstrate semantic segmentation."""
    print("\n=== Semantic Segmentation Demo ===")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return False
    
    try:
        # Create speech configuration with semantic segmentation (uses .env settings)
        speech_config = SpeechConfig()
        speech_config.set_property(PropertyId.Speech_SegmentationStrategy, "Semantic")
        
        # Create speech recognizer
        speech_recognizer = SpeechRecognizer(speech_config=speech_config)
        
        print("Speak a longer sentence with natural pauses...")
        print("(Semantic segmentation will provide better sentence boundaries)")
        
        # Perform recognition
        result = speech_recognizer.recognize_once_async()
        
        if result.reason == ResultReason.RecognizedSpeech:
            print(f"✅ Recognized (with semantic segmentation): {result.text}")
            return True
        elif result.reason == ResultReason.NoMatch:
            print("❌ No speech could be recognized")
            return False
        elif result.reason == ResultReason.Canceled:
            print("❌ Speech recognition was canceled")
            return False
        
    except Exception as e:
        print(f"❌ Error during recognition: {e}")
        return False


def demo_audio_devices():
    """Demonstrate listing available audio devices."""
    print("\n=== Audio Devices Demo ===")
    
    try:
        audio_config = AudioConfig()
        devices = audio_config.get_audio_devices()
        
        print(f"Found {len(devices)} audio input device(s):")
        for i, device in enumerate(devices):
            print(f"  {i+1}. ID: {device['id']}")
            print(f"     Name: {device['name']}")
            print(f"     Channels: {device['channels']}")
            print(f"     Sample Rate: {device['sample_rate']} Hz")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing audio devices: {e}")
        return False


def main():
    """Main demo function."""
    print("🎤 Groq Speech SDK Demo")
    print("=" * 50)
    
    # Check if API key is set in .env
    try:
        Config.get_api_key()
        print("✅ API key found in .env file")
    except ValueError:
        print("⚠️  Warning: GROQ_API_KEY not set in .env file.")
        print("   Some demos may not work without a valid API key.")
        print("   Please update your .env file with your API key.")
        print()
    
    # Run demos
    demos = [
        ("Audio Devices", demo_audio_devices),
        ("Microphone Recognition", demo_microphone_recognition),
        ("Language Change", demo_language_change),
        ("Semantic Segmentation", demo_semantic_segmentation),
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            print(f"\n🔄 Running {name} demo...")
            result = demo_func()
            results.append((name, result))
            time.sleep(1)  # Brief pause between demos
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {name} demo: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Demo Summary:")
    for name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"   {name}: {status}")
    
    successful = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {successful}/{total} demos passed")
    
    if successful == total:
        print("🎉 All demos passed! The SDK is working correctly.")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    return successful == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 