#!/usr/bin/env python3
"""
Test script to verify configuration system is working.
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_speech import Config, get_config

def test_config():
    """Test the configuration system."""
    print("🔧 Testing Configuration System")
    print("=" * 40)
    
    # Test basic config access
    print("📋 Configuration Values:")
    print(f"   API Key: {'✅ Set' if Config.validate_api_key() else '❌ Not set'}")
    print(f"   API Base URL: {Config.GROQ_API_BASE_URL}")
    print(f"   Default Language: {Config.DEFAULT_LANGUAGE}")
    print(f"   Sample Rate: {Config.DEFAULT_SAMPLE_RATE}")
    print(f"   Channels: {Config.DEFAULT_CHANNELS}")
    print(f"   Chunk Size: {Config.DEFAULT_CHUNK_SIZE}")
    print(f"   Device Index: {Config.get_device_index()}")
    print(f"   Timeout: {Config.DEFAULT_TIMEOUT}")
    print(f"   Semantic Segmentation: {Config.ENABLE_SEMANTIC_SEGMENTATION}")
    print(f"   Language Identification: {Config.ENABLE_LANGUAGE_IDENTIFICATION}")
    
    # Test API key validation
    print("\n🔑 API Key Validation:")
    try:
        api_key = Config.get_api_key()
        print(f"   ✅ API key is valid: {api_key[:10]}...")
    except ValueError as e:
        print(f"   ❌ API key error: {e}")
    
    # Test convenience function
    print("\n⚙️  Convenience Function Test:")
    config = get_config()
    print(f"   Config object: {type(config).__name__}")
    print(f"   Same as Config class: {config is Config}")
    
    print("\n✅ Configuration test completed!")

if __name__ == "__main__":
    test_config() 