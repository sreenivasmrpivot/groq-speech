#!/usr/bin/env python3
"""
Basic speech recognition examples using Groq Speech SDK.
This example demonstrates single-shot recognition from microphone and file.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationReason, PropertyId, Config


def from_microphone():
    """Recognize speech from microphone input."""
    print("=== Microphone Recognition Example ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config)
    
    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async()
    
    # Handle the result
    if speech_recognition_result.reason == ResultReason.RecognizedSpeech:
        print(f"Recognized: {speech_recognition_result.text}")
    elif speech_recognition_result.reason == ResultReason.NoMatch:
        print(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
    elif speech_recognition_result.reason == ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print(f"Speech Recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
            print("Did you set the Groq API key?")


def from_file(filename: str):
    """Recognize speech from an audio file."""
    print(f"=== File Recognition Example: {filename} ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Create audio configuration for file input
    audio_config = AudioConfig(filename=filename)
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    speech_recognition_result = speech_recognizer.recognize_once_async()
    
    # Handle the result
    if speech_recognition_result.reason == ResultReason.RecognizedSpeech:
        print(f"Recognized: {speech_recognition_result.text}")
    elif speech_recognition_result.reason == ResultReason.NoMatch:
        print(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
    elif speech_recognition_result.reason == ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print(f"Speech Recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")


def change_language():
    """Demonstrate changing the recognition language."""
    print("=== Language Change Example ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Change the recognition language to German
    speech_config.speech_recognition_language = "de-DE"
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config)
    
    print("Speak in German into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async()
    
    if speech_recognition_result.reason == ResultReason.RecognizedSpeech:
        print(f"Recognized (German): {speech_recognition_result.text}")
    elif speech_recognition_result.reason == ResultReason.NoMatch:
        print(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
    elif speech_recognition_result.reason == ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print(f"Speech Recognition canceled: {cancellation_details.reason}")


def semantic_segmentation():
    """Demonstrate semantic segmentation for better recognition results."""
    print("=== Semantic Segmentation Example ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Enable semantic segmentation
    speech_config.set_property(PropertyId.Speech_SegmentationStrategy, "Semantic")
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config)
    
    print("Speak a longer sentence with natural pauses.")
    speech_recognition_result = speech_recognizer.recognize_once_async()
    
    if speech_recognition_result.reason == ResultReason.RecognizedSpeech:
        print(f"Recognized (with semantic segmentation): {speech_recognition_result.text}")
    elif speech_recognition_result.reason == ResultReason.NoMatch:
        print(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
    elif speech_recognition_result.reason == ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print(f"Speech Recognition canceled: {cancellation_details.reason}")


if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Please set your GROQ_API_KEY environment variable or update the examples with your API key.")
        print("You can get your API key from: https://console.groq.com/")
        sys.exit(1)
    
    # Run examples
    try:
        # Example 1: Microphone recognition
        from_microphone()
        print("\n" + "="*50 + "\n")
        
        # Example 2: File recognition (if file exists)
        test_file = "test_audio.wav"
        if os.path.exists(test_file):
            from_file(test_file)
            print("\n" + "="*50 + "\n")
        else:
            print(f"Test file {test_file} not found. Skipping file recognition example.")
            print("\n" + "="*50 + "\n")
        
        # Example 3: Language change
        change_language()
        print("\n" + "="*50 + "\n")
        
        # Example 4: Semantic segmentation
        semantic_segmentation()
        
    except KeyboardInterrupt:
        print("\nRecognition interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set up your Groq API key correctly.") 