# Groq Speech SDK - Detailed Code Analysis

## 🎯 **Critical Files Analysis**

### **1. `examples/speech_demo.py` (CLI Reference)**

**Purpose**: Command-line interface that demonstrates all SDK capabilities

**Key Functions**:
- `process_audio_file()` - Handles file-based processing
- `process_microphone_single()` - Single-shot microphone recording
- `process_microphone_continuous()` - Continuous microphone with VAD
- `validate_environment()` - Checks API keys and dependencies

**Critical Patterns**:
```python
# File processing (simple)
result = recognizer.process_file(audio_file, enable_diarization=enable_diarization, is_translation=is_translation)

# Audio data processing (direct)
result = recognizer.recognize_audio_data(audio_data, RATE, is_translation=is_translation)

# Translation configuration
if args.operation == "translation":
    speech_config.enable_translation = True
    speech_config.set_translation_target_language("en")
```

### **2. `api/server.py` (Backend API)**

**Purpose**: FastAPI server that exposes CLI functionality via REST and WebSocket

**Key Endpoints**:
- `POST /api/v1/recognize` - File transcription (REST)
- `POST /api/v1/translate` - File translation (REST)
- `WebSocket /ws/recognize` - Real-time microphone processing

**Critical Functions**:
```python
async def recognize_speech(request: RecognitionRequest):
    # Setup config - EXACTLY like CLI
    speech_config = get_speech_config(
        model=request.model,
        is_translation=False,
        target_language=request.target_language,
    )
    
    # Create recognizer - EXACTLY like CLI
    recognizer = SpeechRecognizer(speech_config)
    
    # Process based on diarization requirement
    if request.enable_diarization:
        # Save to temp file and use process_file (like CLI)
        result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=False)
    else:
        # Direct audio data processing (like CLI)
        result = recognizer.recognize_audio_data(audio_array_float, is_translation=False)
```

**WebSocket Handler**:
```python
async def websocket_endpoint(websocket: WebSocket):
    # Handles real-time audio streaming
    # Processes audio chunks similar to CLI continuous mode
    # Sends results back via WebSocket messages
```

### **3. `examples/groq-speech-ui/src/components/EnhancedSpeechDemo.tsx` (Frontend UI)**

**Purpose**: Main React component that provides web interface for all speech features

**Key State Management**:
```typescript
const [selectedCommand, setSelectedCommand] = useState<CommandType>('file_transcription');
const [isProcessing, setIsProcessing] = useState(false);
const [isRecording, setIsRecording] = useState(false);
const [results, setResults] = useState<RecognitionResult[]>([]);
const [diarizationResults, setDiarizationResults] = useState<DiarizationResult[]>([]);
```

**Command Configuration**:
```typescript
const COMMAND_CONFIGS: CommandConfig[] = [
    // File-based commands (REST API)
    { id: 'file_transcription', endpoint: 'rest', diarization: false },
    { id: 'file_transcription_diarize', endpoint: 'rest', diarization: true },
    { id: 'file_translation', endpoint: 'rest', diarization: false },
    { id: 'file_translation_diarize', endpoint: 'rest', diarization: true },
    
    // Microphone commands (WebSocket)
    { id: 'microphone_single', endpoint: 'websocket', diarization: false },
    { id: 'microphone_single_diarize', endpoint: 'websocket', diarization: true },
    { id: 'microphone_continuous', endpoint: 'websocket', diarization: false },
    // ... etc
];
```

**Key Processing Logic**:
```typescript
const handleCommand = async (command: CommandType) => {
    const config = COMMAND_CONFIGS.find(c => c.id === command);
    
    if (config?.category === 'file') {
        // File processing via REST API
        const result = await apiClient.processAudio(audioData, isTranslation, targetLanguage, enableDiarization);
    } else if (config?.category === 'microphone') {
        // Microphone processing via WebSocket
        const ws = await apiClient.processAudioWithWebSocket(onResult, onError, selectedLanguage, isTranslation, mode);
    }
};
```

### **4. `examples/groq-speech-ui/src/lib/groq-api.ts` (API Client)**

**Purpose**: Handles communication between frontend and backend

**Key Methods**:
```typescript
async processAudio(
    audioData: ArrayBuffer,
    isTranslation: boolean = false,
    targetLanguage: string = 'en',
    enableDiarization: boolean = false
): Promise<RecognitionResult> {
    // Convert WebM/Opus to raw PCM (16kHz)
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(audioData);
    
    // Convert to int16 array
    const int16Array = new Int16Array(channelData.length);
    for (let i = 0; i < channelData.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, Math.round(channelData[i] * 32767)));
    }
    
    // Convert to base64
    const base64Audio = btoa(binaryString);
    
    // Send to API
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        body: JSON.stringify({
            audio_data: base64Audio,
            enable_timestamps: true,
            enable_language_detection: true,
            target_language: targetLanguage,
            enable_diarization: enableDiarization,
        }),
    });
}
```

**WebSocket Method**:
```typescript
async processAudioWithWebSocket(
    onResult: (result: RecognitionResult) => void,
    onError: (error: string) => void,
    selectedLanguage: string = 'en-US',
    isTranslation: boolean = false,
    mode: string = 'continuous'
): Promise<WebSocket> {
    const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws/recognize';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        ws.send(JSON.stringify({
            type: 'start_recognition',
            data: {
                model: 'whisper-large-v3-turbo',
                is_translation: isTranslation,
                target_language: selectedLanguage,
                mode: mode,
            },
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'recognition_result') {
            onResult(createRecognitionResult(data.data));
        } else if (data.type === 'diarization_result') {
            onResult(createDiarizationResult(data.data));
        }
    };
}
```

## 🔍 **Critical Implementation Details**

### **Audio Processing Pipeline**

1. **Frontend (Web Audio API)**:
   ```typescript
   // Record audio
   const mediaRecorder = new MediaRecorder(stream);
   const audioChunks: Blob[] = [];
   
   // Convert to ArrayBuffer
   const audioData = await audioBlob.arrayBuffer();
   
   // Convert to PCM via Web Audio API
   const audioContext = new AudioContext({ sampleRate: 16000 });
   const audioBuffer = await audioContext.decodeAudioData(audioData);
   ```

2. **Backend (Python Processing)**:
   ```python
   # Decode base64 audio
   audio_bytes = base64.b64decode(request.audio_data)
   audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
   audio_array_float = audio_array.astype(np.float32) / 32768.0
   
   # Process with recognizer
   result = recognizer.recognize_audio_data(audio_array_float, is_translation=False)
   ```

### **Diarization Handling**

**CLI Pattern**:
```python
# Save to temp file for diarization
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
    temp_path = temp_file.name
    sf.write(temp_path, audio_data, RATE)

result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=is_translation)
```

**API Pattern**:
```python
# Same pattern in API server
if request.enable_diarization:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_path, audio_array_float, 16000)
    
    result = recognizer.process_file(temp_path, enable_diarization=True, is_translation=False)
```

### **Translation Configuration**

**CLI Pattern**:
```python
if args.operation == "translation":
    speech_config.enable_translation = True
    speech_config.set_translation_target_language("en")
```

**API Pattern**:
```python
speech_config = get_speech_config(
    model=request.model,
    is_translation=True,  # Key difference
    target_language=request.target_language,
)
speech_config.enable_translation = True
```

## 🚨 **Potential Issues & Debugging Points**

### **1. Audio Format Issues**
- **Check**: Sample rate consistency (16kHz)
- **Check**: Audio encoding/decoding
- **Check**: Base64 conversion accuracy

### **2. WebSocket State Management**
- **Check**: Connection state handling
- **Check**: Message parsing
- **Check**: Error handling

### **3. Diarization Temp Files**
- **Check**: File cleanup in API server
- **Check**: File permissions
- **Check**: Memory usage

### **4. Translation Mode**
- **Check**: Configuration propagation
- **Check**: API endpoint routing
- **Check**: Response parsing

### **5. Error Handling**
- **Check**: Consistent error responses
- **Check**: Frontend error display
- **Check**: Backend error logging

## 🔧 **Quick Debugging Commands**

```bash
# Test CLI functionality
python examples/speech_demo.py --file test1.wav
python examples/speech_demo.py --file test1.wav --diarize
python examples/speech_demo.py --microphone-mode single

# Test API server
curl -X POST http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/recognize -H "Content-Type: application/json" -d '{"audio_data":"..."}'

# Test frontend
cd examples/groq-speech-ui
npm run dev
```

This analysis should help you identify any issues in the implementation and understand how each component works together.
