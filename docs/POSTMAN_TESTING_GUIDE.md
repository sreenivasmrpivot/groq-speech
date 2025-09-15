# Postman API Testing Guide

This guide provides comprehensive instructions for testing all API endpoints using Postman, replicating the CLI functionality.

## Prerequisites

1. **API Server Running**: Ensure the API server is running on `http://localhost:8000`
2. **Postman Installed**: Download and install Postman
3. **Test Audio File**: Use `test1.wav` from the examples directory
4. **Environment Variables**: Ensure `GROQ_API_KEY` and `HF_TOKEN` are configured

## Base Configuration

### Environment Variables in Postman
Create a new environment in Postman with these variables:
```
API_BASE_URL = http://localhost:8000
WS_BASE_URL = ws://localhost:8000
```

## REST API Endpoints

### 1. Health Check
**Endpoint**: `GET {{API_BASE_URL}}/health`

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-10T20:51:17.989Z",
  "version": "1.0.0",
  "api_key_configured": true
}
```

### 2. Basic File Recognition
**Endpoint**: `POST {{API_BASE_URL}}/api/v1/recognize`

**Headers**:
```
Content-Type: application/json
```

**Body** (raw JSON):
```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "model": "whisper-large-v3-turbo",
  "enable_timestamps": false,
  "enable_language_detection": true,
  "enable_diarization": false
}
```

**Expected Response**:
```json
{
  "success": true,
  "text": "Thank you for seeing me. Doctor, today I'm excited to introduce you to a new treatment for moderate to severe eczema, gaboderm ointment...",
  "confidence": 0.95,
  "language": "en",
  "timestamps": null,
  "segments": null,
  "num_speakers": null,
  "error": null
}
```

### 3. File Recognition with Diarization
**Endpoint**: `POST {{API_BASE_URL}}/api/v1/recognize`

**Body** (raw JSON):
```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "model": "whisper-large-v3-turbo",
  "enable_timestamps": false,
  "enable_language_detection": true,
  "enable_diarization": true
}
```

**Expected Response**:
```json
{
  "success": true,
  "text": "Thank you for seeing me. Doctor, today I'm excited to introduce you to a new treatment for moderate to severe eczema, gaboderm ointment...",
  "confidence": 0.95,
  "language": "en",
  "timestamps": null,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "Thank you for seeing me. Doctor, today I'm excited to introduce you to a new treatment for moderate to severe eczema, gaboderm ointment...",
      "start_time": 0.0,
      "end_time": 16.23375,
      "confidence": 0.95
    },
    {
      "speaker": "SPEAKER_01",
      "text": "Yes, of course.",
      "start_time": 16.23375,
      "end_time": 17.17875,
      "confidence": 0.95
    }
  ],
  "num_speakers": 2,
  "error": null
}
```

### 4. Basic File Translation
**Endpoint**: `POST {{API_BASE_URL}}/api/v1/translate`

**Body** (raw JSON):
```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "model": "whisper-large-v3-turbo",
  "target_language": "es",
  "enable_timestamps": false,
  "enable_language_detection": true,
  "enable_diarization": false
}
```

**Expected Response**:
```json
{
  "success": true,
  "text": "Gracias por verme. Doctor, hoy estoy emocionado de presentarle un nuevo tratamiento para el eccema moderado a severo, ungüento gaboderm...",
  "confidence": 0.95,
  "language": "es",
  "timestamps": null,
  "segments": null,
  "num_speakers": null,
  "error": null
}
```

### 5. File Translation with Diarization
**Endpoint**: `POST {{API_BASE_URL}}/api/v1/translate`

**Body** (raw JSON):
```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "model": "whisper-large-v3-turbo",
  "target_language": "fr",
  "enable_timestamps": false,
  "enable_language_detection": true,
  "enable_diarization": true
}
```

### 6. Available Models
**Endpoint**: `GET {{API_BASE_URL}}/api/v1/models`

**Expected Response**:
```json
{
  "models": [
    "whisper-large-v3-turbo",
    "whisper-large-v3"
  ]
}
```

## WebSocket Testing

### WebSocket URL
```
{{WS_BASE_URL}}/ws/recognize
```

### 1. Basic Recognition WebSocket

**Connection**: Connect to WebSocket URL

**Message 1 - Start Recognition**:
```json
{
  "type": "start_recognition",
  "data": {
    "model": "whisper-large-v3-turbo",
    "is_translation": false,
    "target_language": "en",
    "enable_timestamps": false,
    "enable_language_detection": true,
    "enable_diarization": false
  }
}
```

**Message 2 - Send Audio Data**:
```json
{
  "type": "audio_data",
  "data": {
    "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
    "is_final": true
  }
}
```

**Expected Response**:
```json
{
  "type": "recognition_result",
  "data": {
    "success": true,
    "text": "Thank you for seeing me. Doctor, today I'm excited to introduce you to a new treatment for moderate to severe eczema, gaboderm ointment...",
    "confidence": 0.95,
    "language": "en",
    "timestamps": null,
    "segments": null,
    "num_speakers": null,
    "error": null
  }
}
```

### 2. Recognition with Diarization WebSocket

**Message 1 - Start Recognition with Diarization**:
```json
{
  "type": "start_recognition",
  "data": {
    "model": "whisper-large-v3-turbo",
    "is_translation": false,
    "target_language": "en",
    "enable_timestamps": false,
    "enable_language_detection": true,
    "enable_diarization": true
  }
}
```

**Message 2 - Send Audio Data**:
```json
{
  "type": "audio_data",
  "data": {
    "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
    "is_final": true
  }
}
```

### 3. Translation WebSocket

**Message 1 - Start Translation**:
```json
{
  "type": "start_recognition",
  "data": {
    "model": "whisper-large-v3-turbo",
    "is_translation": true,
    "target_language": "es",
    "enable_timestamps": false,
    "enable_language_detection": true,
    "enable_diarization": false
  }
}
```

### 4. Translation with Diarization WebSocket

**Message 1 - Start Translation with Diarization**:
```json
{
  "type": "start_recognition",
  "data": {
    "model": "whisper-large-v3-turbo",
    "is_translation": true,
    "target_language": "fr",
    "enable_timestamps": false,
    "enable_language_detection": true,
    "enable_diarization": true
  }
}
```

## Frontend Logging Endpoint

### Log Endpoint
**Endpoint**: `POST {{API_BASE_URL}}/api/log`

**Headers**:
```
Content-Type: application/json
```

**Body** (raw JSON):
```json
{
  "component": "TEST",
  "level": "INFO",
  "message": "Test log message",
  "data": {
    "test": true,
    "timestamp": "2025-01-10T20:51:17.989Z"
  },
  "timestamp": "2025-01-10T20:51:17.989Z"
}
```

**Expected Response**:
```json
{
  "status": "logged"
}
```

## Testing Checklist

### REST API Tests
- [ ] Health check returns 200 with correct structure
- [ ] Basic recognition returns text with confidence > 0.9
- [ ] Recognition with diarization returns segments and speaker count
- [ ] Basic translation returns text in target language
- [ ] Translation with diarization returns segments in target language
- [ ] Models endpoint returns available models
- [ ] Error handling for invalid audio data

### WebSocket Tests
- [ ] Basic recognition WebSocket works
- [ ] Recognition with diarization WebSocket works
- [ ] Translation WebSocket works
- [ ] Translation with diarization WebSocket works
- [ ] WebSocket connection handling
- [ ] WebSocket error handling

### Frontend Integration Tests
- [ ] Log endpoint accepts frontend logs
- [ ] CORS headers are properly set
- [ ] All endpoints return proper JSON responses

## Expected CLI Equivalents

| CLI Command | API Equivalent |
|-------------|----------------|
| `python speech_demo.py --file test1.wav` | `POST /api/v1/recognize` (basic) |
| `python speech_demo.py --file test1.wav --diarize` | `POST /api/v1/recognize` (with diarization) |
| `python speech_demo.py --file test1.wav --operation translation` | `POST /api/v1/translate` (basic) |
| `python speech_demo.py --file test1.wav --operation translation --diarize` | `POST /api/v1/translate` (with diarization) |
| `python speech_demo.py --microphone-mode single` | WebSocket `/ws/recognize` (basic) |
| `python speech_demo.py --microphone-mode single --diarize` | WebSocket `/ws/recognize` (with diarization) |
| `python speech_demo.py --microphone-mode single --operation translation` | WebSocket `/ws/recognize` (translation) |
| `python speech_demo.py --microphone-mode single --operation translation --diarize` | WebSocket `/ws/recognize` (translation with diarization) |

## Troubleshooting

### Common Issues
1. **Connection Refused**: Ensure API server is running on port 8000
2. **Timeout Errors**: Increase timeout in Postman settings
3. **Invalid Audio Data**: Ensure audio file is properly base64 encoded
4. **Missing Environment Variables**: Check GROQ_API_KEY and HF_TOKEN are set
5. **WebSocket Connection Failed**: Ensure WebSocket URL is correct

### Debug Steps
1. Check API server logs for errors
2. Verify environment variables are loaded
3. Test with smaller audio files first
4. Check network connectivity
5. Verify audio file format and encoding

## Performance Expectations

- **Basic Recognition**: 2-5 seconds for 3-minute audio
- **Recognition with Diarization**: 10-30 seconds for 3-minute audio
- **Translation**: 3-6 seconds for 3-minute audio
- **WebSocket Response**: 1-3 seconds for real-time processing

## Success Criteria

All tests should pass with:
- HTTP 200 status codes for successful requests
- Confidence scores > 0.9 for accurate recognition
- Proper JSON response structure
- WebSocket connections establish and receive responses
- Diarization returns multiple speakers and segments
- Translation returns text in target language
