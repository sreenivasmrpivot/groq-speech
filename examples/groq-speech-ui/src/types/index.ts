export interface TimingMetrics {
  microphone_capture?: number;
  api_call?: number;
  response_processing?: number;
  total_time?: number;
}

export interface RecognitionResult {
  text: string;
  confidence: number;
  language: string;
  timestamps?: Array<{
    start: number;
    end: number;
    text: string;
    avg_logprob: number;
    compression_ratio: number;
    no_speech_prob: number;
  }>;
  timing_metrics?: TimingMetrics;
  timestamp: string;
}

export interface PerformanceMetrics {
  total_requests: number;
  successful_recognitions: number;
  failed_recognitions: number;
  avg_response_time: number;
  audio_processing: {
    avg_processing_time: number;
    total_chunks: number;
    buffer_size: number;
  };
}

export interface RecognitionMode {
  type: 'single' | 'continuous';
  operation: 'transcription' | 'translation';
}

export interface AudioConfig {
  sample_rate: number;
  channels: number;
  chunk_duration: number;
  buffer_size: number;
  silence_threshold: number;
  vad_enabled: boolean;
  enable_compression: boolean;
}

export interface ModelConfig {
  model_id: string;
  response_format: string;
  temperature: number;
  enable_word_timestamps: boolean;
  enable_segment_timestamps: boolean;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
  error?: string;
}

export interface RecognitionSession {
  id: string;
  mode: RecognitionMode;
  isActive: boolean;
  startTime: string;
  results: RecognitionResult[];
  performanceMetrics: PerformanceMetrics;
} 