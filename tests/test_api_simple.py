#!/usr/bin/env python3
"""
Simple API Test Script

This script tests the API endpoints with actual requests to validate functionality.
"""

import requests
import base64
import os
import json
import time

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_models():
    """Test models endpoint."""
    print("\n🔍 Testing models endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_basic_recognition():
    """Test basic recognition."""
    print("\n🔍 Testing basic recognition...")
    
    # Load test audio file
    if not os.path.exists("examples/test1.wav"):
        print("❌ test1.wav not found")
        return False
    
    with open("examples/test1.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "audio_data": audio_data,
        "model": "whisper-large-v3-turbo",
        "enable_timestamps": False,
        "enable_language_detection": True,
        "enable_diarization": False
    }
    
    response = requests.post(f"{API_BASE_URL}/api/v1/recognize", json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Text length: {len(data.get('text', ''))} chars")
        print(f"Confidence: {data.get('confidence', 0)}")
        print(f"Language: {data.get('language', 'unknown')}")
        print(f"Sample text: {data.get('text', '')[:100]}...")
        return data['success']
    else:
        print(f"Error: {response.text}")
        return False

def test_translation():
    """Test translation."""
    print("\n🔍 Testing translation...")
    
    # Load test audio file
    if not os.path.exists("examples/test1.wav"):
        print("❌ test1.wav not found")
        return False
    
    with open("examples/test1.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "audio_data": audio_data,
        "model": "whisper-large-v3-turbo",
        "target_language": "es",
        "enable_timestamps": False,
        "enable_language_detection": True,
        "enable_diarization": False
    }
    
    response = requests.post(f"{API_BASE_URL}/api/v1/translate", json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Text length: {len(data.get('text', ''))} chars")
        print(f"Confidence: {data.get('confidence', 0)}")
        print(f"Language: {data.get('language', 'unknown')}")
        print(f"Sample text: {data.get('text', '')[:100]}...")
        return data['success']
    else:
        print(f"Error: {response.text}")
        return False

def test_diarization():
    """Test diarization."""
    print("\n🔍 Testing diarization...")
    
    # Load test audio file
    if not os.path.exists("examples/test1.wav"):
        print("❌ test1.wav not found")
        return False
    
    with open("examples/test1.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "audio_data": audio_data,
        "model": "whisper-large-v3-turbo",
        "enable_timestamps": False,
        "enable_language_detection": True,
        "enable_diarization": True
    }
    
    response = requests.post(f"{API_BASE_URL}/api/v1/recognize", json=payload, timeout=120)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Speakers: {data.get('num_speakers', 0)}")
        segments = data.get('segments', [])
        print(f"Segments: {len(segments)}")
        if segments:
            print(f"First segment: {segments[0]}")
        return data['success']
    else:
        print(f"Error: {response.text}")
        return False

def test_log_endpoint():
    """Test log endpoint."""
    print("\n🔍 Testing log endpoint...")
    
    payload = {
        "component": "TEST",
        "level": "INFO",
        "message": "Test log message",
        "data": {
            "test": True,
            "timestamp": "2025-01-10T20:51:17.989Z"
        },
        "timestamp": "2025-01-10T20:51:17.989Z"
    }
    
    response = requests.post(f"{API_BASE_URL}/api/log", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def main():
    """Run all tests."""
    print("🚀 Starting API tests...")
    
    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models),
        ("Basic Recognition", test_basic_recognition),
        ("Translation", test_translation),
        ("Diarization", test_diarization),
        ("Log Endpoint", test_log_endpoint),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print(f"{'✅' if result else '❌'} {name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            results.append((name, False))
    
    print(f"\n📊 Test Results:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
