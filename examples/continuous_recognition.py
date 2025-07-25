#!/usr/bin/env python3
"""
Continuous speech recognition example using Groq Speech SDK.
This example demonstrates continuous recognition with event handling.
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationReason, PropertyId, Config


def continuous_recognition():
    """Demonstrate continuous speech recognition with event handling."""
    print("=== Continuous Recognition Example ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Create audio configuration
    audio_config = AudioConfig()
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Set up event handlers
    done = False
    
    def stop_cb(evt):
        """Callback to stop continuous recognition."""
        print(f'CLOSING on {evt}')
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
    
    def recognizing_cb(evt):
        """Callback for intermediate recognition results."""
        print(f'RECOGNIZING: {evt}')
    
    def recognized_cb(evt):
        """Callback for final recognition results."""
        print(f'RECOGNIZED: {evt}')
        if evt.reason == ResultReason.RecognizedSpeech:
            print(f"Final result: {evt.text}")
        elif evt.reason == ResultReason.NoMatch:
            print(f"No match: {evt.no_match_details}")
    
    def session_started_cb(evt):
        """Callback for session started event."""
        print(f'SESSION STARTED: {evt}')
    
    def session_stopped_cb(evt):
        """Callback for session stopped event."""
        print(f'SESSION STOPPED: {evt}')
    
    def canceled_cb(evt):
        """Callback for canceled event."""
        print(f'CANCELED: {evt}')
        if evt.cancellation_details:
            print(f"Cancellation reason: {evt.cancellation_details.reason}")
            if evt.cancellation_details.reason == CancellationReason.Error:
                print(f"Error details: {evt.cancellation_details.error_details}")
    
    # Connect event handlers
    speech_recognizer.connect('recognizing', recognizing_cb)
    speech_recognizer.connect('recognized', recognized_cb)
    speech_recognizer.connect('session_started', session_started_cb)
    speech_recognizer.connect('session_stopped', session_stopped_cb)
    speech_recognizer.connect('canceled', canceled_cb)
    
    # Connect stop callbacks
    speech_recognizer.connect('session_stopped', stop_cb)
    speech_recognizer.connect('canceled', stop_cb)
    
    # Start continuous recognition
    print("Starting continuous recognition. Speak continuously...")
    print("Press Ctrl+C to stop.")
    
    try:
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        while not done:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping recognition...")
        speech_recognizer.stop_continuous_recognition()
    
    print("Continuous recognition completed.")


def continuous_recognition_with_file(filename: str):
    """Demonstrate continuous recognition from an audio file."""
    print(f"=== Continuous Recognition from File: {filename} ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Create audio configuration for file input
    audio_config = AudioConfig(filename=filename)
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Set up event handlers
    done = False
    results = []
    
    def stop_cb(evt):
        """Callback to stop continuous recognition."""
        print(f'CLOSING on {evt}')
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
    
    def recognized_cb(evt):
        """Callback for final recognition results."""
        print(f'RECOGNIZED: {evt}')
        if evt.reason == ResultReason.RecognizedSpeech:
            results.append(evt.text)
            print(f"Final result: {evt.text}")
    
    def session_started_cb(evt):
        """Callback for session started event."""
        print(f'SESSION STARTED: {evt}')
    
    def session_stopped_cb(evt):
        """Callback for session stopped event."""
        print(f'SESSION STOPPED: {evt}')
    
    # Connect event handlers
    speech_recognizer.connect('recognized', recognized_cb)
    speech_recognizer.connect('session_started', session_started_cb)
    speech_recognizer.connect('session_stopped', session_stopped_cb)
    
    # Connect stop callbacks
    speech_recognizer.connect('session_stopped', stop_cb)
    
    # Start continuous recognition
    print("Starting continuous recognition from file...")
    
    try:
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        while not done:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping recognition...")
        speech_recognizer.stop_continuous_recognition()
    
    print("Continuous recognition completed.")
    print(f"Total results: {len(results)}")
    print("Full transcription:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")


def language_identification():
    """Demonstrate language identification with continuous recognition."""
    print("=== Language Identification Example ===")
    
    # Create speech configuration (uses .env settings)
    speech_config = SpeechConfig()
    
    # Enable language identification
    speech_config.set_property(PropertyId.Speech_Recognition_EnableLanguageIdentification, "true")
    speech_config.set_property(PropertyId.Speech_Recognition_LanguageIdentificationMode, "Continuous")
    
    # Create audio configuration
    audio_config = AudioConfig()
    
    # Create speech recognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Set up event handlers
    done = False
    
    def stop_cb(evt):
        """Callback to stop continuous recognition."""
        print(f'CLOSING on {evt}')
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
    
    def recognized_cb(evt):
        """Callback for final recognition results."""
        print(f'RECOGNIZED: {evt}')
        if evt.reason == ResultReason.RecognizedSpeech:
            print(f"Text: {evt.text}")
            print(f"Language: {evt.language}")
            print(f"Confidence: {evt.confidence}")
    
    def session_started_cb(evt):
        """Callback for session started event."""
        print(f'SESSION STARTED: {evt}')
    
    def session_stopped_cb(evt):
        """Callback for session stopped event."""
        print(f'SESSION STOPPED: {evt}')
    
    # Connect event handlers
    speech_recognizer.connect('recognized', recognized_cb)
    speech_recognizer.connect('session_started', session_started_cb)
    speech_recognizer.connect('session_stopped', session_stopped_cb)
    
    # Connect stop callbacks
    speech_recognizer.connect('session_stopped', stop_cb)
    
    # Start continuous recognition
    print("Starting language identification. Speak in different languages...")
    print("Press Ctrl+C to stop.")
    
    try:
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        while not done:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping recognition...")
        speech_recognizer.stop_continuous_recognition()
    
    print("Language identification completed.")


if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Please set your GROQ_API_KEY environment variable or update the examples with your API key.")
        print("You can get your API key from: https://console.groq.com/")
        sys.exit(1)
    
    # Run examples
    try:
        # Example 1: Continuous recognition from microphone
        continuous_recognition()
        print("\n" + "="*50 + "\n")
        
        # Example 2: Continuous recognition from file (if file exists)
        test_file = "test_audio.wav"
        if os.path.exists(test_file):
            continuous_recognition_with_file(test_file)
            print("\n" + "="*50 + "\n")
        else:
            print(f"Test file {test_file} not found. Skipping file recognition example.")
            print("\n" + "="*50 + "\n")
        
        # Example 3: Language identification
        language_identification()
        
    except KeyboardInterrupt:
        print("\nRecognition interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set up your Groq API key correctly.") 