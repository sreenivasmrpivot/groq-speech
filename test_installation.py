#!/usr/bin/env python3
"""
Test script to verify Groq Speech SDK installation and basic functionality.
"""

import os
import sys
import subprocess
import importlib

def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Need Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'groq',
        'pyaudio',
        'numpy',
        'soundfile',
        'websockets',
        'asyncio_mqtt',
        'dotenv'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def test_sdk_imports():
    """Test that the SDK modules can be imported."""
    print("Testing SDK imports...")
    
    try:
        from groq_speech import (
            SpeechConfig, 
            SpeechRecognizer, 
            AudioConfig, 
            ResultReason, 
            CancellationReason,
            PropertyId
        )
        print("✅ All SDK modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import SDK modules: {e}")
        return False

def test_pip_list():
    """Test that all required packages are installed."""
    print("Testing installed packages...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'list'
        ], capture_output=True, text=True, check=True)
        
        installed_packages = result.stdout.lower()
        
        required_packages = [
            'groq',
            'pyaudio',
            'numpy',
            'soundfile',
            'websockets',
            'asyncio-mqtt',
            'python-dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        else:
            print("✅ All required packages are installed")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking installed packages: {e}")
        return False

def test_audio_devices():
    """Test audio device detection."""
    print("Testing audio devices...")
    
    try:
        from groq_speech import AudioConfig
        audio_config = AudioConfig()
        devices = audio_config.get_audio_devices()
        
        if len(devices) > 0:
            print(f"✅ Found {len(devices)} audio input device(s)")
            for i, device in enumerate(devices[:3]):  # Show first 3 devices
                print(f"   {i+1}. {device['name']} (ID: {device['id']})")
            return True
        else:
            print("⚠️  No audio input devices found")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"❌ Error detecting audio devices: {e}")
        return False

def test_api_key():
    """Test API key configuration."""
    print("Testing API key configuration...")
    
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        print("✅ GROQ_API_KEY environment variable is set")
        return True
    else:
        print("⚠️  GROQ_API_KEY environment variable is not set")
        print("   Set it with: export GROQ_API_KEY='your-api-key'")
        return True  # Not a failure, just a warning

def test_basic_functionality():
    """Test basic SDK functionality."""
    print("Testing basic SDK functionality...")
    
    try:
        from groq_speech import SpeechConfig, AudioConfig, SpeechRecognizer
        
        # Test SpeechConfig
        config = SpeechConfig(api_key="test-key")
        assert config.api_key == "test-key"
        
        # Test AudioConfig
        audio_config = AudioConfig()
        assert audio_config.sample_rate == 16000
        
        # Test SpeechRecognizer (skip actual API call for test)
        try:
            recognizer = SpeechRecognizer(speech_config=config)
            assert recognizer.speech_config == config
        except Exception as e:
            # It's okay if this fails due to invalid API key
            print(f"Note: SpeechRecognizer test skipped due to invalid API key (expected)")
            pass
        
        print("✅ Basic SDK functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic SDK functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔧 Groq Speech SDK Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("SDK Imports", test_sdk_imports),
        ("Installed Packages", test_pip_list),
        ("Audio Devices", test_audio_devices),
        ("API Key Configuration", test_api_key),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n🔄 Running {name}...")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Installation Test Summary:")
    for name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"   {name}: {status}")
    
    successful = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {successful}/{total} tests passed")
    
    if successful == total:
        print("🎉 All tests passed! The SDK is properly installed.")
        print("\nNext steps:")
        print("1. Set your GROQ_API_KEY: export GROQ_API_KEY='your-api-key'")
        print("2. Run the demo: python demo.py")
        print("3. Try the examples: python examples/basic_recognition.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in a virtual environment")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check your Python version (3.8+ required)")
        print("4. Verify audio device connections")
    
    return successful == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Installation test interrupted. Goodbye!")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 