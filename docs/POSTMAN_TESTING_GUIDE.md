# Groq Speech SDK - Postman Testing Guide

## 🚀 **Quick Start**

### **Prerequisites**
- Postman installed
- Groq Speech API server running (http://localhost:8000)
- Audio files for testing (WAV format recommended)

### **Setup**
1. **Import Collection**: Import the provided Postman collection
2. **Set Environment**: Configure environment variables
3. **Test Health**: Verify API server is running

## 📋 **Postman Collection**

### **Environment Variables**
```json
{
  "base_url": "http://localhost:8000",
  "api_key": "your_groq_api_key",
  "hf_token": "your_huggingface_token",
  "test_audio_file": "path/to/test.wav"
}
```

### **Collection Structure**
```
Groq Speech SDK API
├── Health Check
│   ├── GET Health Check
│   └── GET Models
├── File Processing
│   ├── POST File Transcription
│   ├── POST File Translation
│   ├── POST File Transcription + Diarization
│   └── POST File Translation + Diarization
├── Microphone Processing
│   ├── POST Single Microphone
│   ├── POST Continuous Microphone
│   ├── POST Single Microphone + Diarization
│   └── POST Continuous Microphone + Diarization
└── Utility Endpoints
    ├── POST Frontend Log
    ├── GET Languages
    └── VAD Endpoints (Legacy)
```

## 🔍 **API Endpoints Testing**

### **1. Health Check**

#### **GET Health Check**
```http
GET {{base_url}}/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "groq_api": "connected",
    "hf_api": "connected",
    "gpu": "available"
  }
}
```

**Test Script:**
```javascript
pm.test("Health check returns 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response contains status", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("status");
    pm.expect(jsonData.status).to.eql("healthy");
});
```

#### **GET Models**
```http
GET {{base_url}}/api/v1/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "id": "whisper-large-v3",
      "name": "Whisper Large V3",
      "description": "High accuracy, slower processing"
    },
    {
      "id": "whisper-large-v3-turbo",
      "name": "Whisper Large V3 Turbo",
      "description": "Fast processing, good accuracy"
    }
  ]
}
```

### **2. File Processing**

#### **POST File Transcription**
```http
POST {{base_url}}/api/v1/recognize
Content-Type: application/json

{
  "audio_data": "{{base64_audio_data}}",
  "model": "whisper-large-v3-turbo",
  "enable_timestamps": false,
  "target_language": "en",
  "enable_language_detection": true,
  "enable_diarization": false
}
```

**Pre-request Script:**
```javascript
// Convert audio file to base64
const fs = require('fs');
const path = pm.environment.get('test_audio_file');

if (path && fs.existsSync(path)) {
    const audioData = fs.readFileSync(path);
    const base64Data = audioData.toString('base64');
    pm.environment.set('base64_audio_data', base64Data);
} else {
    console.log('Audio file not found, using sample data');
    pm.environment.set('base64_audio_data', 'sample_base64_data');
}
```

**Test Script:**
```javascript
pm.test("File transcription returns 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response contains transcription", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("success");
    pm.expect(jsonData.success).to.be.true;
    pm.expect(jsonData).to.have.property("text");
    pm.expect(jsonData.text).to.be.a('string');
    pm.expect(jsonData.text.length).to.be.greaterThan(0);
});
```

#### **POST File Translation**
```http
POST {{base_url}}/api/v1/translate
Content-Type: application/json

{
  "audio_data": "{{base64_audio_data}}",
  "model": "whisper-large-v3-turbo",
  "enable_timestamps": false,
  "target_language": "en",
  "enable_language_detection": true,
  "enable_diarization": false
}
```

#### **POST File Transcription + Diarization**
```http
POST {{base_url}}/api/v1/recognize
Content-Type: application/json

{
  "audio_data": "{{base64_audio_data}}",
  "model": "whisper-large-v3-turbo",
  "enable_timestamps": true,
  "target_language": "en",
  "enable_language_detection": true,
  "enable_diarization": true
}
```

**Test Script:**
```javascript
pm.test("Diarization returns 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response contains diarization data", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("success");
    pm.expect(jsonData.success).to.be.true;
    pm.expect(jsonData).to.have.property("enable_diarization");
    pm.expect(jsonData.enable_diarization).to.be.true;
});
```

### **3. Microphone Processing**

#### **POST Single Microphone**
```http
POST {{base_url}}/api/v1/recognize-microphone
Content-Type: application/json

{
  "audio_data": [0.1, 0.2, 0.3, 0.4, 0.5],
  "sample_rate": 16000,
  "enable_diarization": false,
  "is_translation": false,
  "target_language": "en"
}
```

**Pre-request Script:**
```javascript
// Generate sample audio data (Float32Array as JSON array)
const sampleRate = 16000;
const duration = 3; // 3 seconds
const frequency = 440; // A4 note
const audioData = [];

for (let i = 0; i < sampleRate * duration; i++) {
    const sample = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.1;
    audioData.push(sample);
}

pm.environment.set('sample_audio_data', JSON.stringify(audioData));
```

**Test Script:**
```javascript
pm.test("Microphone recognition returns 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response contains recognition result", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("success");
    pm.expect(jsonData.success).to.be.true;
});
```

#### **POST Continuous Microphone**
```http
POST {{base_url}}/api/v1/recognize-microphone-continuous
Content-Type: application/json

{
  "audio_data": {{sample_audio_data}},
  "sample_rate": 16000,
  "enable_diarization": false,
  "is_translation": false,
  "target_language": "en"
}
```

### **4. Utility Endpoints**

#### **POST Frontend Log**
```http
POST {{base_url}}/api/log
Content-Type: application/json

{
  "component": "POSTMAN_TEST",
  "level": "INFO",
  "message": "Test log message",
  "data": {
    "test": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### **GET Languages**
```http
GET {{base_url}}/api/v1/languages
```

### **5. VAD Endpoints (Legacy)**

#### **POST VAD Should Create Chunk**
```http
POST {{base_url}}/api/v1/vad/should-create-chunk
Content-Type: application/json

{
  "audio_data": {{sample_audio_data}},
  "sample_rate": 16000,
  "max_duration_seconds": 390
}
```

#### **POST VAD Audio Level**
```http
POST {{base_url}}/api/v1/vad/audio-level
Content-Type: application/json

{
  "audio_data": {{sample_audio_data}},
  "sample_rate": 16000
}
```

## 🧪 **Test Scenarios**

### **Scenario 1: Basic File Transcription**
1. **Setup**: Ensure API server is running
2. **Test**: Upload a WAV file for transcription
3. **Verify**: Check response contains transcribed text
4. **Validate**: Confirm confidence score > 0.8

### **Scenario 2: File Translation**
1. **Setup**: Prepare non-English audio file
2. **Test**: Send translation request with target language
3. **Verify**: Check response contains translated text
4. **Validate**: Confirm translation is in target language

### **Scenario 3: Speaker Diarization**
1. **Setup**: Prepare multi-speaker audio file
2. **Test**: Send diarization request
3. **Verify**: Check response contains speaker segments
4. **Validate**: Confirm multiple speakers detected

### **Scenario 4: Microphone Processing**
1. **Setup**: Generate sample audio data
2. **Test**: Send microphone recognition request
3. **Verify**: Check response contains recognition result
4. **Validate**: Confirm processing completed successfully

### **Scenario 5: Error Handling**
1. **Test**: Send invalid audio data
2. **Verify**: Check error response format
3. **Validate**: Confirm appropriate HTTP status code
4. **Check**: Verify error message is descriptive

## 📊 **Performance Testing**

### **Load Testing Script**
```javascript
// Add to Pre-request Script
const startTime = Date.now();
pm.environment.set('request_start_time', startTime);

// Add to Test Script
const endTime = Date.now();
const startTime = pm.environment.get('request_start_time');
const responseTime = endTime - startTime;

pm.test(`Response time is less than 5000ms`, function () {
    pm.expect(responseTime).to.be.below(5000);
});

console.log(`Response time: ${responseTime}ms`);
```

### **Concurrent Testing**
1. **Setup**: Create multiple Postman instances
2. **Test**: Send simultaneous requests
3. **Monitor**: Check response times and success rates
4. **Validate**: Confirm system handles concurrency

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Connection Refused**
```bash
# Check if API server is running
curl http://localhost:8000/health

# Start API server if needed
cd api && python server.py
```

#### **2. Invalid Audio Data**
```javascript
// Verify audio data format
const audioData = pm.environment.get('base64_audio_data');
pm.test("Audio data is valid base64", function () {
    pm.expect(audioData).to.match(/^[A-Za-z0-9+/]*={0,2}$/);
});
```

#### **3. Authentication Errors**
```javascript
// Check API key configuration
const apiKey = pm.environment.get('api_key');
pm.test("API key is configured", function () {
    pm.expect(apiKey).to.not.be.undefined;
    pm.expect(apiKey).to.not.be.empty;
});
```

### **Debug Scripts**

#### **Response Debugging**
```javascript
// Add to Test Script
console.log("Response Status:", pm.response.status);
console.log("Response Headers:", pm.response.headers);
console.log("Response Body:", pm.response.text());
```

#### **Request Debugging**
```javascript
// Add to Pre-request Script
console.log("Request URL:", pm.request.url);
console.log("Request Headers:", pm.request.headers);
console.log("Request Body:", pm.request.body);
```

## 📈 **Monitoring & Metrics**

### **Response Time Monitoring**
```javascript
// Add to Test Script
const responseTime = pm.response.responseTime;
pm.test(`Response time: ${responseTime}ms`, function () {
    pm.expect(responseTime).to.be.below(10000);
});
```

### **Success Rate Tracking**
```javascript
// Add to Test Script
const success = pm.response.json().success;
if (success) {
    pm.environment.set('success_count', 
        (pm.environment.get('success_count') || 0) + 1);
} else {
    pm.environment.set('failure_count', 
        (pm.environment.get('failure_count') || 0) + 1);
}
```

### **Error Rate Calculation**
```javascript
// Add to Test Script
const successCount = pm.environment.get('success_count') || 0;
const failureCount = pm.environment.get('failure_count') || 0;
const totalCount = successCount + failureCount;
const errorRate = totalCount > 0 ? (failureCount / totalCount) * 100 : 0;

console.log(`Error Rate: ${errorRate.toFixed(2)}%`);
```

## 📋 **Test Checklist**

### **Pre-Testing**
- [ ] API server is running
- [ ] Environment variables are set
- [ ] Test audio files are available
- [ ] Postman collection is imported

### **Basic Functionality**
- [ ] Health check returns 200
- [ ] File transcription works
- [ ] File translation works
- [ ] Microphone processing works
- [ ] Diarization works

### **Error Handling**
- [ ] Invalid audio data returns error
- [ ] Missing parameters return error
- [ ] Server errors are handled gracefully
- [ ] Error messages are descriptive

### **Performance**
- [ ] Response times are acceptable
- [ ] Concurrent requests work
- [ ] Memory usage is reasonable
- [ ] No memory leaks detected

### **Security**
- [ ] API keys are not exposed
- [ ] Sensitive data is not logged
- [ ] CORS is configured correctly
- [ ] Input validation works

---

**This guide provides comprehensive testing instructions for the Groq Speech SDK API using Postman.**