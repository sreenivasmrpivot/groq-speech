# Groq Speech SDK - Complete Architecture Analysis

## 🏗️ **Overall Architecture Overview**

The Groq Speech SDK has **3 main components**:

1. **`groq_speech/`** - Core Python SDK (speech processing engine)
2. **`api/`** - FastAPI backend server (REST API only)
3. **`examples/groq-speech-ui/`** - Next.js frontend (React UI)

## 📁 **File Structure & Purpose**

### **Core SDK (`groq_speech/`)**
- **`speech_recognizer.py`** - Main orchestrator class, handles all speech processing
- **`speech_config.py`** - Configuration management with factory methods
- **`speaker_diarization.py`** - Speaker diarization using Pyannote.audio with GPU support
- **`vad_service.py`** - Voice Activity Detection service
- **`audio_utils.py`** - Audio format utilities and conversion
- **`exceptions.py`** - Custom exception classes
- **`result_reason.py`** - Result status enums
- **`logging_utils.py`** - Structured logging utilities

### **API Server (`api/`)**
- **`server.py`** - FastAPI server with REST API endpoints only
- **`models/requests.py`** - Pydantic request models
- **`models/responses.py`** - Pydantic response models

### **Frontend (`examples/groq-speech-ui/`)**
- **`src/app/page.tsx`** - Main page component with configuration checks
- **`src/components/EnhancedSpeechDemo.tsx`** - Main UI component with all features
- **`src/components/PerformanceMetrics.tsx`** - Performance metrics visualization
- **`src/components/DebugPanel.tsx`** - Debug information panel
- **`src/lib/groq-api.ts`** - REST API client for backend communication
- **`src/lib/audio-recorder.ts`** - Unified audio recording (standard + optimized)
- **`src/lib/continuous-audio-recorder.ts`** - VAD-based continuous recording
- **`src/lib/client-vad-service.ts`** - Client-side Voice Activity Detection
- **`src/lib/audio-converter.ts`** - Unified audio conversion (standard + optimized)
- **`src/lib/frontend-logger.ts`** - Frontend logging utilities
- **`src/types/index.ts`** - TypeScript type definitions

## 🔄 **Flow Comparison: CLI vs Web UI**

#### **CLI Pattern (Direct Access):**
- **Layer 2a → Layer 1**: Direct function calls
- **No network boundaries**: All processing happens in the same process
- **Simplest path**: CLI calls SDK methods directly

#### **Web UI Pattern (Network Access):**
- **Layer 3 → Layer 2b**: HTTP REST requests only
- **Layer 2b → Layer 1**: Direct function calls (same as CLI)
- **Network boundaries**: UI and API run in separate processes
- **More complex**: Requires serialization/deserialization of data

#### **Key Boundary Crossings:**
1. **UI → API**: 
   - File processing: Audio converted to base64, sent via HTTP REST
   - Microphone processing: Audio converted to Float32Array (as JSON array), sent via HTTP REST
2. **API → SDK**: Same direct calls as CLI, but with decoded audio data
3. **SDK → API**: Results returned as Python objects
4. **API → UI**: Results serialized to JSON, sent via HTTP REST

## **Architecture Diagrams**

### **Overall System Architecture**
```mermaid
graph TB
    subgraph "Layer 3: UI Client"
        UI[groq-speech-ui/<br/>EnhancedSpeechDemo.tsx<br/>PerformanceMetrics.tsx]
    end
    
    subgraph "Layer 2b: API Client"
        API[api/server.py<br/>FastAPI REST API only]
    end
    
    subgraph "Layer 2a: CLI Client"
        CLI[speech_demo.py<br/>Command Line Interface]
    end
    
    subgraph "Layer 1: SDK"
        SDK[groq_speech/<br/>speech_recognizer.py<br/>speaker_diarization.py<br/>vad_service.py<br/>audio_utils.py]
    end
    
    UI -->|HTTP REST| API
    CLI -->|Direct Calls| SDK
    API -->|Direct Calls| SDK
    
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style CLI fill:#f3e5f5
    style SDK fill:#e8f5e8
```

### **CLI Flow Pattern**
```mermaid
sequenceDiagram
    participant User
    participant CLI as Layer 2a: CLI Client
    participant SDK as Layer 1: SDK
    
    User->>CLI: Run speech_demo.py
    CLI->>CLI: Process audio input
    CLI->>SDK: Direct function call
    SDK->>SDK: Process audio
    SDK->>CLI: Return result
    CLI->>User: Display result
```

### **Web UI Flow Pattern**
```mermaid
sequenceDiagram
    participant User
    participant UI as Layer 3: UI Client
    participant API as Layer 2b: API Client
    participant SDK as Layer 1: SDK
    
    User->>UI: Select operation
    UI->>UI: Process audio input
    UI->>API: HTTP REST request
    API->>API: Decode audio data
    API->>SDK: Direct function call
    SDK->>SDK: Process audio
    SDK->>API: Return result
    API->>UI: HTTP REST response
    UI->>User: Display result
```

### **Data Flow Transformations**
```mermaid
graph LR
    subgraph "CLI Path"
        A1[Audio File/Mic] --> A2[numpy array] --> A3[SDK Processing] --> A4[Console Output]
    end
    
    subgraph "Web UI Path"
        B1[Audio File/Mic] --> B2{File or Mic?}
        B2 -->|File| B3[base64] --> B4[HTTP REST] --> B5[base64 decode] --> B6[numpy array] --> B7[SDK Processing] --> B8[JSON] --> B9[HTTP REST] --> B10[UI Display]
        B2 -->|Mic| B11[Float32Array] --> B12[HTTP REST] --> B13[Array conversion] --> B6[numpy array] --> B7[SDK Processing] --> B8[JSON] --> B9[HTTP REST] --> B10[UI Display]
    end
    
    style A1 fill:#e8f5e8
    style A4 fill:#e8f5e8
    style B1 fill:#e1f5fe
    style B9 fill:#e1f5fe
```

## 🎯 **Current API Endpoints**

The API server provides the following REST endpoints:

### **Core Endpoints:**
- `POST /api/v1/recognize` - File transcription with base64 audio data
- `POST /api/v1/translate` - File translation with base64 audio data
- `POST /api/v1/recognize-microphone` - Single microphone processing with Float32Array
- `POST /api/v1/recognize-microphone-continuous` - Continuous microphone processing with Float32Array

### **Utility Endpoints:**
- `GET /health` - Health check endpoint
- `GET /api/v1/models` - Available Groq models
- `GET /api/v1/languages` - Supported languages
- `POST /api/log` - Frontend logging

### **VAD Endpoints (Legacy - Not Used by Frontend):**
- `POST /api/v1/vad/should-create-chunk` - VAD chunk detection
- `POST /api/v1/vad/audio-level` - Audio level analysis

### **Data Formats:**
- **File Processing**: Base64-encoded audio data (WAV format)
- **Microphone Processing**: Float32Array as JSON array (raw PCM data)
- **All Responses**: JSON with success/error status and results

## 🔄 **Detailed Flow Examples**

### **1. File Transcription (`--file test1.wav`)**

#### **CLI Flow (Direct Layer Access):**
```mermaid
sequenceDiagram
    participant User
    participant CLI as Layer 2a: CLI Client
    participant SDK as Layer 1: SDK
    
    User->>CLI: python speech_demo.py --file test1.wav
    CLI->>CLI: process_audio_file()
    CLI->>SDK: recognizer.process_file(audio_file, enable_diarization=False, is_translation=False)
    SDK->>SDK: AudioFormatUtils.decode_audio()
    SDK->>SDK: GroqAPIClient.transcribe()
    SDK->>SDK: ResponseParser.parse_response()
    SDK->>CLI: SpeechRecognitionResult
    CLI->>User: Display transcription result
```

#### **Web UI Flow (Multi-Layer with Network Boundaries):**
```mermaid
sequenceDiagram
    participant User
    participant UI as Layer 3: UI Client
    participant API as Layer 2b: API Client
    participant SDK as Layer 1: SDK
    
    User->>UI: Select "File Transcription" → Upload file
    UI->>UI: AudioConverter.convertToPCM() → base64
    UI->>API: POST /api/v1/recognize → base64 audio
    API->>API: AudioFormatUtils.decode_base64_audio()
    API->>SDK: recognizer.process_file() → Same as CLI
    SDK->>SDK: AudioFormatUtils.decode_audio()
    SDK->>SDK: GroqAPIClient.transcribe()
    SDK->>API: SpeechRecognitionResult
    API->>UI: JSON response with transcription
    UI->>User: Display transcription result
```

### **2. Continuous Microphone with VAD**

#### **CLI Flow (Direct Layer Access):**
```mermaid
sequenceDiagram
    participant User
    participant CLI as Layer 2a: CLI Client
    participant SDK as Layer 1: SDK
    
    User->>CLI: python speech_demo.py --microphone-mode continuous
    CLI->>CLI: process_microphone_continuous()
    loop For each audio chunk
        CLI->>CLI: PyAudio recording → numpy array
        CLI->>SDK: recognizer.should_create_chunk() → VAD analysis
        CLI->>SDK: recognizer.get_audio_level() → Visual feedback
        alt Chunk should be created
            CLI->>SDK: recognizer.recognize_audio_data_chunked()
            SDK->>CLI: SpeechRecognitionResult
            CLI->>User: Display transcription
        end
    end
```

#### **Web UI Flow (Client-Side VAD):**
```mermaid
sequenceDiagram
    participant User
    participant UI as Layer 3: UI Client
    participant API as Layer 2b: API Client
    participant SDK as Layer 1: SDK
    
    User->>UI: Select "Continuous Microphone" → Start recording
    UI->>UI: ContinuousAudioRecorder.startRecording() → Web Audio API
    loop For each audio chunk
        UI->>UI: ClientVADService.shouldCreateChunk() → Client-side VAD
        UI->>UI: ClientVADService.getAudioLevel() → Visual feedback
        alt Chunk should be created
            UI->>API: POST /api/v1/recognize-microphone-continuous → Float32Array
            API->>API: AudioFormatUtils.convert_list_to_numpy()
            API->>SDK: recognizer.recognize_audio_data_chunked()
            SDK->>API: SpeechRecognitionResult
            API->>UI: JSON response with transcription
            UI->>User: Display transcription
        end
    end
```

## 🔧 **Key Technical Differences**

### **Audio Processing:**
- **CLI**: Uses PyAudio for microphone input, direct numpy array processing
- **Web UI**: 
  - File processing: Uses Web Audio API, converts to base64 for transmission
  - Microphone processing: Uses Web Audio API, converts to Float32Array (JSON array) for transmission

### **VAD Processing:**
- **CLI**: Server-side VAD using `recognizer.should_create_chunk()`
- **Web UI**: Client-side VAD using `ClientVADService` for real-time performance

### **API Communication:**
- **CLI**: Direct function calls to `groq_speech` module
- **Web UI**: HTTP REST API for all communication

### **Diarization Handling:**
- **CLI**: Direct file processing with `process_file()`
- **Web UI**: Same logic but through API endpoints

### **Translation Mode:**
- **CLI**: Sets `speech_config.enable_translation = True`
- **Web UI**: Passes `is_translation` parameter through API

## 🚀 **Performance Optimizations**

### **Unified Components:**
- **AudioRecorder**: Single class with both standard and optimized modes
- **AudioConverter**: Single class with both standard and optimized modes
- **ClientVADService**: Client-side VAD for real-time processing

### **Memory Management:**
- **Chunked Processing**: Handles large files without memory issues
- **Client-Side VAD**: No network latency for real-time decisions
- **Efficient Audio Conversion**: Optimized base64 and Float32Array handling

### **GPU Support:**
- **Pyannote.audio**: Automatic GPU detection and usage
- **CUDA Support**: Available in Docker and Cloud Run deployments

## 🔌 **Current API Endpoints**

The API server provides the following REST endpoints:

### **Core Endpoints:**
- `POST /api/v1/recognize` - File transcription with base64 audio data
- `POST /api/v1/translate` - File translation with base64 audio data
- `POST /api/v1/recognize-microphone` - Single microphone processing with Float32Array
- `POST /api/v1/recognize-microphone-continuous` - Continuous microphone processing with Float32Array

### **Utility Endpoints:**
- `GET /health` - Health check endpoint
- `GET /api/v1/models` - Available Groq models
- `GET /api/v1/languages` - Supported languages
- `POST /api/log` - Frontend logging

### **VAD Endpoints (Legacy - Not Used by Frontend):**
- `POST /api/v1/vad/should-create-chunk` - VAD chunk detection
- `POST /api/v1/vad/audio-level` - Audio level analysis

### **Data Formats:**
- **File Processing**: Base64-encoded audio data (WAV format)
- **Microphone Processing**: Float32Array as JSON array (raw PCM data)
- **All Responses**: JSON with success/error status and results

## 🎯 **Key Insights**

1. **Same Core Logic**: Both CLI and Web UI use identical SDK processing
2. **Different Interfaces**: CLI uses direct calls, Web UI uses HTTP REST protocols
3. **Client-Side VAD**: Web UI uses client-side VAD for real-time performance
4. **Unified Components**: Single classes handle both standard and optimized modes
5. **Performance**: CLI is faster (no network overhead), Web UI is more accessible

This architecture ensures the Web UI provides the same functionality as the CLI but through a web interface with proper separation of concerns and optimized performance.