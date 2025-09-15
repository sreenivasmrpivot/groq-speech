# Groq Speech SDK - Complete Code Analysis

## 🏗️ **System Overview**

The Groq Speech SDK is a comprehensive speech recognition and translation system with three main components:

1. **Core SDK (`groq_speech/`)** - Python-based speech processing engine
2. **API Server (`api/`)** - FastAPI REST API server
3. **Frontend (`examples/groq-speech-ui/`)** - Next.js React web interface

## 📁 **Component Analysis**

### **Core SDK (`groq_speech/`)**

#### **`speech_recognizer.py`** - Main Orchestrator
```python
class SpeechRecognizer:
    """Main orchestrator class for speech processing"""
    
    def __init__(self, speech_config: SpeechConfig, translation_target_language: str = "en"):
        self.speech_config = speech_config
        self.translation_target_language = translation_target_language
        self.vad_service = VADService(VADConfig())
        # ... other services
    
    async def process_file(self, audio_file: str, enable_diarization: bool = False, is_translation: bool = False):
        """Process audio file with optional diarization and translation"""
        
    def recognize_audio_data_chunked(self, audio_data: np.ndarray, sample_rate: int, is_translation: bool = False):
        """Process audio data in chunks for continuous recognition"""
        
    def should_create_chunk(self, audio_data: np.ndarray, sample_rate: int, max_duration: float) -> Tuple[bool, str]:
        """VAD method to determine if chunk should be created"""
        
    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """Get audio level for visualization"""
```

**Key Features:**
- **Unified Interface**: Single class handles all speech processing
- **VAD Integration**: Built-in voice activity detection
- **Chunked Processing**: Handles continuous audio streams
- **Diarization Support**: Optional speaker diarization
- **Translation Support**: Optional translation capabilities

#### **`speech_config.py`** - Configuration Management
```python
class SpeechConfig:
    """Configuration management with factory methods"""
    
    @staticmethod
    def create_for_recognition(model: str = "whisper-large-v3-turbo", 
                             enable_timestamps: bool = False) -> 'SpeechConfig':
        """Factory method for recognition configuration"""
        
    @staticmethod
    def create_for_translation(model: str = "whisper-large-v3-turbo",
                              target_language: str = "en") -> 'SpeechConfig':
        """Factory method for translation configuration"""
        
    @staticmethod
    def create_for_diarization(model: str = "whisper-large-v3-turbo",
                              enable_timestamps: bool = True) -> 'SpeechConfig':
        """Factory method for diarization configuration"""
```

**Key Features:**
- **Factory Methods**: Centralized configuration creation
- **GPU Support**: Automatic GPU detection and usage
- **Environment Management**: Handles API keys and tokens
- **Model Configuration**: Supports different Groq models

#### **`audio_utils.py`** - Audio Format Utilities
```python
class AudioFormatUtils:
    """Centralized audio format utilities"""
    
    @staticmethod
    def decode_base64_audio(base64_data: str) -> Tuple[np.ndarray, int]:
        """Decode base64 audio data to numpy array"""
        
    @staticmethod
    def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode raw audio bytes to numpy array"""
        
    @staticmethod
    def save_audio_to_temp_file(audio_data: np.ndarray, sample_rate: int) -> str:
        """Save audio data to temporary file"""
        
    @staticmethod
    def get_audio_info(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Get comprehensive audio information"""
```

**Key Features:**
- **Format Conversion**: Handles multiple audio formats
- **Base64 Support**: Web-compatible audio encoding
- **Temp File Management**: Efficient temporary file handling
- **Audio Analysis**: Comprehensive audio information extraction

#### **`speaker_diarization.py`** - Speaker Diarization
```python
class SpeakerDiarizationService:
    """Speaker diarization using Pyannote.audio"""
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline = self._load_pipeline()
        
    def _load_pipeline(self) -> Pipeline:
        """Load Pyannote.audio pipeline with GPU support"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.config.hf_token)
        return pipeline.to(device)
        
    async def diarize_file(self, audio_file: str, mode: str, verbose: bool = False) -> DiarizationResult:
        """Perform speaker diarization on audio file"""
```

**Key Features:**
- **GPU Support**: Automatic CUDA detection and usage
- **Pyannote.audio Integration**: State-of-the-art speaker diarization
- **Async Processing**: Non-blocking diarization operations
- **Result Parsing**: Comprehensive diarization results

### **API Server (`api/`)**

#### **`server.py`** - FastAPI REST Server
```python
@app.post("/api/v1/recognize", response_model=RecognitionResponse)
async def recognize_speech(request: RecognitionRequest):
    """Recognize speech from audio data - EXACTLY like CLI file recognition"""
    # Decode audio data using SDK utilities
    audio_array_float, sample_rate = AudioFormatUtils.decode_base64_audio(request.audio_data)
    
    # Create recognizer - EXACTLY like CLI
    recognizer = SpeechRecognizer(speech_config=speech_config, translation_target_language=request.target_language)
    
    # Process with groq_speech - EXACTLY like CLI
    result = recognizer.process_file(temp_path, enable_diarization=request.enable_diarization, is_translation=False)

@app.post("/api/v1/recognize-microphone-continuous", response_model=RecognitionResponse)
async def recognize_microphone_continuous(request: dict):
    """Continuous microphone recognition - EXACTLY like speech_demo.py process_microphone_continuous"""
    # Convert list to numpy array - EXACTLY like speech_demo.py
    audio_array = np.array(request.audio_data, dtype=np.float32)
    
    # Process with groq_speech - EXACTLY like speech_demo.py process_microphone_continuous
    result = recognizer.recognize_audio_data_chunked(audio_array, sample_rate, is_translation=is_translation)
```

**Key Features:**
- **REST API Only**: No WebSocket endpoints
- **SDK Integration**: Direct calls to groq_speech SDK
- **Audio Format Handling**: Supports both base64 and Float32Array
- **Error Handling**: Comprehensive error handling and logging

### **Frontend (`examples/groq-speech-ui/`)**

#### **`EnhancedSpeechDemo.tsx`** - Main UI Component
```typescript
export const EnhancedSpeechDemo: React.FC<EnhancedSpeechDemoProps> = () => {
    const [selectedCommand, setSelectedCommand] = useState<CommandType>('file_transcription');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [results, setResults] = useState<RecognitionResult[]>([]);
    const [diarizationResults, setDiarizationResults] = useState<DiarizationResult[]>([]);
    
    // 10 different command configurations
    const COMMAND_CONFIGS: CommandConfig[] = [
        // File-based commands
        { id: 'file_transcription', name: 'File Transcription', ... },
        { id: 'file_transcription_diarize', name: 'File Transcription + Diarization', ... },
        { id: 'file_translation', name: 'File Translation', ... },
        { id: 'file_translation_diarize', name: 'File Translation + Diarization', ... },
        // Microphone single commands
        { id: 'microphone_single', name: 'Single Microphone', ... },
        { id: 'microphone_single_diarize', name: 'Single Microphone + Diarization', ... },
        { id: 'microphone_single_translation', name: 'Single Microphone Translation', ... },
        { id: 'microphone_single_translation_diarize', name: 'Single Microphone Translation + Diarization', ... },
        // Microphone continuous commands
        { id: 'microphone_continuous', name: 'Continuous Microphone', ... },
        { id: 'microphone_continuous_diarize', name: 'Continuous Microphone + Diarization', ... },
        { id: 'microphone_continuous_translation', name: 'Continuous Microphone Translation', ... },
        { id: 'microphone_continuous_translation_diarize', name: 'Continuous Microphone Translation + Diarization', ... },
    ];
```

**Key Features:**
- **10 Command Types**: Complete coverage of all speech operations
- **Unified Interface**: Single component handles all functionality
- **Real-time Feedback**: Live audio level and status updates
- **Performance Metrics**: Built-in performance monitoring

#### **`audio-recorder.ts`** - Unified Audio Recording
```typescript
export class AudioRecorder {
    private config: Required<AudioRecorderConfig>;
    private isOptimizedMode = false;
    
    async startRecording(): Promise<void> {
        this.isOptimizedMode = false;
        await this._startRecording();
    }
    
    async startOptimizedRecording(): Promise<void> {
        this.isOptimizedMode = true;
        await this._startRecording();
    }
    
    private async _startRecording(): Promise<void> {
        // Unified recording logic for both modes
    }
}
```

**Key Features:**
- **Unified Interface**: Single class handles both standard and optimized modes
- **Mode Selection**: `startRecording()` vs `startOptimizedRecording()`
- **Configuration**: Flexible configuration for different use cases
- **Error Handling**: Comprehensive error handling and recovery

#### **`continuous-audio-recorder.ts`** - VAD-Based Continuous Recording
```typescript
export class ContinuousAudioRecorder {
    private config: ContinuousAudioRecorderConfig;
    private apiClient: GroqAPIClient;
    private clientVADService: ClientVADService;
    
    private processAudio(combinedAudio: Float32Array, duration: number, sizeMB: number): void {
        // Visual feedback (synchronous for real-time performance)
        const audioLevel = clientVADService.getAudioLevel(combinedAudio.slice(-this.config.sampleRate));
        
        // Check if we should create a chunk (synchronous for real-time performance)
        const [shouldCreate, reason] = clientVADService.shouldCreateChunk(
            combinedAudio, this.config.sampleRate, this.config.maxDurationSeconds
        );
        
        if (shouldCreate) {
            this.processChunk(combinedAudio, reason);
        }
    }
}
```

**Key Features:**
- **Client-Side VAD**: Real-time voice activity detection
- **Synchronous Processing**: No network latency for VAD decisions
- **Visual Feedback**: Live audio level and status updates
- **Chunk Processing**: Automatic chunk creation on silence detection

#### **`client-vad-service.ts`** - Client-Side VAD
```typescript
export class ClientVADService {
    private silenceStartTime: number | null = null;
    private lastAudioTime: number | null = null;
    private readonly silenceThreshold: number = 0.003;
    private readonly requiredSilenceSeconds: number = 15.0;
    
    shouldCreateChunk(audioData: Float32Array, sampleRate: number, maxDurationSeconds?: number): [boolean, string] {
        // Real-time silence detection logic
        const rms = this.calculateRMS(recentAudio);
        const max = Math.max(...recentAudio.map(Math.abs));
        
        if (rms < this.silenceThreshold && max < this.maxSilence) {
            // Silence detected - check duration
            if (silenceDuration >= this.requiredSilenceSeconds) {
                return [true, `Silence detected (${silenceDuration.toFixed(1)}s)`];
            }
        }
        
        return [false, `Audio detected (RMS: ${rms.toFixed(4)})`];
    }
}
```

**Key Features:**
- **Real-Time Processing**: No network latency
- **Same Logic as Python SDK**: Mirrors `groq_speech/vad_service.py`
- **RMS-Based Detection**: Conservative silence detection
- **15-Second Threshold**: Matches CLI behavior

## 🔄 **Audio Format Handling Across Layers**

### **File Processing Flow:**
```
Audio File → Web Audio API → Float32Array → base64 → HTTP REST → base64 decode → numpy array → SDK Processing
```

### **Microphone Processing Flow:**
```
Microphone → Web Audio API → Float32Array → HTTP REST → array conversion → numpy array → SDK Processing
```

### **VAD Processing Flow:**
```
Audio Data → ClientVADService → Real-time decisions → Chunk creation → API processing
```

## 🚀 **Performance Optimizations**

### **Unified Components:**
- **Single Classes**: `AudioRecorder` and `AudioConverter` handle both standard and optimized modes
- **Mode Selection**: Runtime selection between standard and optimized processing
- **Memory Efficiency**: Chunked processing for large files

### **Client-Side VAD:**
- **Real-Time Processing**: No network latency for VAD decisions
- **Synchronous Operations**: Immediate responses for better UX
- **Reduced Server Load**: No VAD API calls from frontend

### **GPU Support:**
- **Automatic Detection**: CUDA detection in `speech_config.py`
- **Pyannote.audio**: GPU acceleration for diarization
- **Docker Support**: GPU-enabled containers available

## 🔧 **Key Technical Decisions**

### **1. Client-Side VAD**
**Decision**: Move VAD from server-side to client-side
**Rationale**: Real-time processing requires immediate responses without network latency
**Implementation**: `ClientVADService` with same logic as Python SDK

### **2. Unified Components**
**Decision**: Merge separate optimized classes into single unified classes
**Rationale**: Reduces code duplication and improves maintainability
**Implementation**: Mode-based methods in `AudioRecorder` and `AudioConverter`

### **3. REST API Only**
**Decision**: Remove WebSocket endpoints, use REST API only
**Rationale**: Simpler architecture, easier to maintain and debug
**Implementation**: All communication via HTTP REST endpoints

### **4. SDK Factory Methods**
**Decision**: Add factory methods to `SpeechConfig`
**Rationale**: Centralized configuration creation, reduces duplication
**Implementation**: `create_for_recognition()`, `create_for_translation()`, `create_for_diarization()`

## 🎯 **Current State Summary**

### **Working Features:**
- ✅ **File Transcription**: Base64 → API → SDK → JSON response
- ✅ **File Translation**: Same flow with translation enabled
- ✅ **File Diarization**: Pyannote.audio integration with GPU support
- ✅ **Single Microphone**: Float32Array → API → SDK → JSON response
- ✅ **Continuous Microphone**: Client-side VAD → chunk processing → API
- ✅ **Real-time VAD**: 15-second silence detection, audio level visualization
- ✅ **Performance Metrics**: Built-in monitoring and analytics
- ✅ **Error Handling**: Comprehensive error handling across all layers

### **Architecture Benefits:**
- **Maintainable**: Clear separation of concerns
- **Scalable**: REST API can be deployed independently
- **Performant**: Client-side VAD for real-time processing
- **Unified**: Single components handle multiple modes
- **Extensible**: Easy to add new features and endpoints

This architecture provides a robust, performant, and maintainable speech processing system that works identically across CLI and web interfaces.