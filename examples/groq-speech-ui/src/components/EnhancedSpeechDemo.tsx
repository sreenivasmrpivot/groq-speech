'use client';

import { AudioRecorder } from '@/lib/audio-recorder';
import { GroqAPIClient } from '@/lib/groq-api';
import { PerformanceMetrics, RecognitionResult, DiarizationResult } from '@/types';
import { uiLogger, audioLogger, apiLogger } from '@/lib/frontend-logger';
import { audioConverter } from '@/lib/audio-converter';
import { OptimizedAudioRecorder } from '@/lib/optimized-audio-recorder';
import { OptimizedAudioConverter } from '@/lib/optimized-audio-converter';
import { ContinuousAudioRecorder } from '@/lib/continuous-audio-recorder';
import {
    Download,
    FileText,
    Mic,
    RotateCcw,
    Settings,
    Square,
    Users,
    Languages
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { PerformanceMetricsComponent } from './PerformanceMetrics';

interface EnhancedSpeechDemoProps {
    // All functionality is self-contained - no props needed
    [key: string]: never;
}

type CommandType = 
    | 'file_transcription'
    | 'file_transcription_diarize'
    | 'file_translation'
    | 'file_translation_diarize'
    | 'microphone_single'
    | 'microphone_single_diarize'
    | 'microphone_single_translation'
    | 'microphone_single_translation_diarize'
    | 'microphone_continuous'
    | 'microphone_continuous_diarize'
    | 'microphone_continuous_translation'
    | 'microphone_continuous_translation_diarize';

interface CommandConfig {
    id: CommandType;
    name: string;
    description: string;
    icon: React.ReactNode;
    category: 'file' | 'microphone';
    mode: 'single' | 'continuous';
    operation: 'transcription' | 'translation';
    diarization: boolean;
}

const COMMAND_CONFIGS: CommandConfig[] = [
    // File-based commands
    {
        id: 'file_transcription',
        name: 'File Transcription',
        description: 'Transcribe audio file without diarization',
        icon: <FileText className="w-5 h-5" />,
        category: 'file',
        mode: 'single',
        operation: 'transcription',
        diarization: false,
    },
    {
        id: 'file_transcription_diarize',
        name: 'File Transcription + Diarization',
        description: 'Transcribe audio file with speaker diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'file',
        mode: 'single',
        operation: 'transcription',
        diarization: true,
    },
    {
        id: 'file_translation',
        name: 'File Translation',
        description: 'Translate audio file without diarization',
        icon: <Languages className="w-5 h-5" />,
        category: 'file',
        mode: 'single',
        operation: 'translation',
        diarization: false,
    },
    {
        id: 'file_translation_diarize',
        name: 'File Translation + Diarization',
        description: 'Translate audio file with speaker diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'file',
        mode: 'single',
        operation: 'translation',
        diarization: true,
    },
    // Microphone single commands
    {
        id: 'microphone_single',
        name: 'Single Microphone',
        description: 'Record once, then transcribe',
        icon: <Mic className="w-5 h-5" />,
        category: 'microphone',
        mode: 'single',
        operation: 'transcription',
        diarization: false,
    },
    {
        id: 'microphone_single_diarize',
        name: 'Single Microphone + Diarization',
        description: 'Record once, then transcribe with diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'microphone',
        mode: 'single',
        operation: 'transcription',
        diarization: true,
    },
    {
        id: 'microphone_single_translation',
        name: 'Single Microphone Translation',
        description: 'Record once, then translate',
        icon: <Languages className="w-5 h-5" />,
        category: 'microphone',
        mode: 'single',
        operation: 'translation',
        diarization: false,
    },
    {
        id: 'microphone_single_translation_diarize',
        name: 'Single Microphone Translation + Diarization',
        description: 'Record once, then translate with diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'microphone',
        mode: 'single',
        operation: 'translation',
        diarization: true,
    },
    // Microphone continuous commands
    {
        id: 'microphone_continuous',
        name: 'Continuous Microphone',
        description: 'Real-time transcription',
        icon: <Mic className="w-5 h-5" />,
        category: 'microphone',
        mode: 'continuous',
        operation: 'transcription',
        diarization: false,
    },
    {
        id: 'microphone_continuous_diarize',
        name: 'Continuous Microphone + Diarization',
        description: 'Real-time transcription with diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'microphone',
        mode: 'continuous',
        operation: 'transcription',
        diarization: true,
    },
    {
        id: 'microphone_continuous_translation',
        name: 'Continuous Microphone Translation',
        description: 'Real-time translation',
        icon: <Languages className="w-5 h-5" />,
        category: 'microphone',
        mode: 'continuous',
        operation: 'translation',
        diarization: false,
    },
    {
        id: 'microphone_continuous_translation_diarize',
        name: 'Continuous Microphone Translation + Diarization',
        description: 'Real-time translation with diarization',
        icon: <Users className="w-5 h-5" />,
        category: 'microphone',
        mode: 'continuous',
        operation: 'translation',
        diarization: true,
    }
];

export const EnhancedSpeechDemo: React.FC<EnhancedSpeechDemoProps> = () => {
    const [selectedCommand, setSelectedCommand] = useState<CommandType>('file_transcription');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [results, setResults] = useState<RecognitionResult[]>([]);
    const [diarizationResults, setDiarizationResults] = useState<DiarizationResult[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const [showMetrics, setShowMetrics] = useState(false);
    const [useOptimizedRecorder, setUseOptimizedRecorder] = useState(true); // Use optimized recorder by default
    const [isDiarizationProcessing, setIsDiarizationProcessing] = useState(false);
    const [audioLevel, setAudioLevel] = useState(0);
    const [recordingStatus, setRecordingStatus] = useState('');
    const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
        total_requests: 0,
        successful_recognitions: 0,
        failed_recognitions: 0,
        avg_response_time: 0,
        audio_processing: {
            avg_processing_time: 0,
            total_chunks: 0,
            buffer_size: 0
        }
    });

    const audioRecorderRef = useRef<AudioRecorder | null>(null);
    const optimizedAudioRecorderRef = useRef<OptimizedAudioRecorder | null>(null);
    const optimizedAudioConverterRef = useRef<OptimizedAudioConverter | null>(null);
    const continuousAudioRecorderRef = useRef<ContinuousAudioRecorder | null>(null);
    const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const apiClientRef = useRef<GroqAPIClient | null>(null);
    const recordingStartTimeRef = useRef<number | null>(null);

    const currentConfig = COMMAND_CONFIGS.find(cmd => cmd.id === selectedCommand);

    useEffect(() => {
        apiClientRef.current = new GroqAPIClient();
        
        // Cleanup on unmount
        return () => {
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
            }
        };
    }, []);

    const startRecording = useCallback(async () => {
        if (!currentConfig || currentConfig.category !== 'microphone') return;

        console.log(`[UI] 🎤 Starting recording: ${currentConfig.name} (${currentConfig.mode} mode)`);
        uiLogger.info('🎤 Starting recording process', {
            command: currentConfig.id,
            mode: currentConfig.mode,
            operation: currentConfig.operation,
            diarization: currentConfig.diarization
        });
        
        // Log to terminal for debugging
        uiLogger.info('🎤 UI Action: Start Recording', {
            action: 'start_recording',
            command: currentConfig.name,
            mode: currentConfig.mode,
            operation: currentConfig.operation,
            diarization: currentConfig.diarization,
            timestamp: new Date().toISOString()
        });

        try {
            setError(null);
            setIsRecording(true);
            setRecordingDuration(0);
            
            // Record the start time for accurate duration calculation
            recordingStartTimeRef.current = Date.now();

            if (useOptimizedRecorder) {
                // Use optimized recorder for large files
                console.log('🎤 Using optimized recorder for single mode');
                
                if (!optimizedAudioRecorderRef.current) {
                    optimizedAudioRecorderRef.current = new OptimizedAudioRecorder({
                        chunkInterval: 1000, // 1 second chunks
                        onChunk: async (chunk: Blob, chunkIndex: number) => {
                            console.log('📦 Optimized audio chunk received', {
                                chunkIndex: chunkIndex,
                                chunkSize: chunk.size,
                                chunkType: chunk.type,
                                recordingDuration: optimizedAudioRecorderRef.current?.getStatus().duration || 0
                            });
                            
                            // Note: Continuous mode now uses ContinuousAudioRecorder with VAD
                            // This onChunk callback is only used for single mode
                        },
                        onComplete: async (audioBlob: Blob, duration: number) => {
                            console.log('✅ Optimized recording completed', {
                                audioSize: audioBlob.size,
                                duration: duration,
                                audioSizeMB: (audioBlob.size / (1024 * 1024)).toFixed(2) + ' MB'
                            });
                            
                            // Stop the duration timer
                            if (durationIntervalRef.current) {
                                clearInterval(durationIntervalRef.current);
                                durationIntervalRef.current = null;
                            }
                            
                            // Process the audio using optimized converter
                            await processOptimizedAudio(audioBlob, duration);
                        },
                        onError: (error: Error) => {
                            console.error('❌ Optimized recording error:', error);
                            setError(`Recording error: ${error.message}`);
                            setIsRecording(false);
                            
                            // Stop the duration timer
                            if (durationIntervalRef.current) {
                                clearInterval(durationIntervalRef.current);
                                durationIntervalRef.current = null;
                            }
                        }
                    });
                }

                audioLogger.info('OptimizedAudioRecorder initialized', {
                    recorderType: 'OptimizedAudioRecorder',
                    mode: currentConfig.mode
                });
            } else {
                // Use legacy recorder for backward compatibility
                console.log('🎤 Using legacy recorder for single mode');
                
                const audioRecorder = new AudioRecorder();
                audioRecorderRef.current = audioRecorder;

                audioLogger.info('AudioRecorder initialized', {
                    recorderType: 'AudioRecorder',
                    mode: currentConfig.mode
                });
            }

            // Start duration timer
            durationIntervalRef.current = setInterval(() => {
                setRecordingDuration(prev => prev + 0.1);
            }, 100);

            if (currentConfig.mode === 'continuous') {
                uiLogger.info('Starting continuous recognition mode with VAD-based silence detection');
                
                // Initialize continuous audio recorder with VAD
                if (!continuousAudioRecorderRef.current) {
                    continuousAudioRecorderRef.current = new ContinuousAudioRecorder({
                        sampleRate: 16000,
                        chunkSize: 8192,
                        maxDurationSeconds: 390, // 6.5 minutes
                        onChunkProcessed: async (audioData: Float32Array, sampleRate: number, reason: string) => {
                            console.log(`🔄 Processing continuous chunk: ${reason}`);
                            await processContinuousRecognitionMicrophone(audioData, sampleRate);
                        },
                        onVisualUpdate: (audioLevel: number, duration: number, sizeMB: number, status: string) => {
                            setAudioLevel(audioLevel);
                            setRecordingStatus(status);
                            setRecordingDuration(duration);
                        },
                        onError: (error: Error) => {
                            console.error('❌ Continuous recording error:', error);
                            setError(`Continuous recording error: ${error.message}`);
                            setIsRecording(false);
                        }
                    });
                }

                await continuousAudioRecorderRef.current.startRecording();
                console.log('🎤 Continuous recording started with VAD-based silence detection');
            } else {
                // For single mode, start recording with the appropriate recorder
                if (useOptimizedRecorder && optimizedAudioRecorderRef.current) {
                    // Use optimized recorder
                    await optimizedAudioRecorderRef.current.startRecording();
                    console.log('🎤 Optimized recording started for single mode');
                } else if (audioRecorderRef.current) {
                    // Use legacy recorder
                    audioRecorderRef.current.setOnDataAvailable((chunk: Blob) => {
                        audioLogger.debug('Audio chunk received for single mode', {
                            chunkSize: chunk.size,
                            chunkType: chunk.type,
                            timestamp: new Date().toISOString()
                        });
                    });

                    await audioRecorderRef.current.startRecording();
                    audioLogger.success('Audio recording started for single mode', {
                        mode: 'single',
                        operation: currentConfig.operation,
                        diarization: currentConfig.diarization
                    });
                }
            }
        } catch (err) {
            uiLogger.error('Recording failed', { error: err, command: currentConfig.id });
            setError(`Recording failed: ${err}`);
            setIsRecording(false);
        }
    }, [currentConfig, useOptimizedRecorder]);

    const stopRecording = useCallback(async () => {
        // Calculate actual duration from start time
        const actualDuration = recordingStartTimeRef.current 
            ? (Date.now() - recordingStartTimeRef.current) / 1000 
            : recordingDuration;
            
        console.log(`[UI] 🛑 Stopping recording: ${currentConfig?.name} (${actualDuration.toFixed(1)}s)`);
        uiLogger.info('🛑 Stopping recording process', {
            mode: currentConfig?.mode,
            stateDuration: recordingDuration,
            actualDuration: actualDuration,
            isRecording
        });
        
        // Log to terminal for debugging
        uiLogger.info('🛑 UI Action: Stop Recording', {
            action: 'stop_recording',
            command: currentConfig?.name,
            mode: currentConfig?.mode,
            operation: currentConfig?.operation,
            diarization: currentConfig?.diarization,
            duration: actualDuration,
            timestamp: new Date().toISOString()
        });

        if (useOptimizedRecorder && optimizedAudioRecorderRef.current) {
            // Use optimized recorder
            console.log('🛑 Stopping optimized recording...');
            optimizedAudioRecorderRef.current.stopRecording();
        } else if (audioRecorderRef.current) {
            // Use legacy recorder
            console.log('🛑 Stopping legacy recording...');
            
            // For single recognition, we need to get the audio data and send it
            if (currentConfig?.mode === 'single') {
                try {
                    // Stop the audio recording first
                    audioRecorderRef.current.stopRecording();
                    audioLogger.success('Audio recording stopped for single mode');
                    
                    // Get the recorded audio data
                    const audioBlob = await audioRecorderRef.current.getAudioBlob();
                    
                    // Calculate actual duration from start time
                    const actualDuration = recordingStartTimeRef.current 
                        ? (Date.now() - recordingStartTimeRef.current) / 1000 
                        : recordingDuration;
                    
                    audioLogger.info('Retrieved audio blob for single recognition', {
                        blobSize: audioBlob.size,
                        blobType: audioBlob.type,
                        stateDuration: recordingDuration,
                        actualDuration: actualDuration,
                        startTime: recordingStartTimeRef.current,
                        endTime: Date.now()
                    });
                    
                    if (audioBlob.size === 0) {
                        uiLogger.error('No audio data captured', { blobSize: audioBlob.size });
                        setError('No audio data captured. Please try recording again.');
                        setIsRecording(false);
                        return;
                    }
                    
                    // Check if we have enough audio data (at least 0.5 seconds)
                    // Use the actual calculated duration instead of state duration
                    if (actualDuration < 0.5) {
                        uiLogger.warning('Recording too short', { 
                            stateDuration: recordingDuration,
                            actualDuration: actualDuration,
                            minimumRequired: 0.5 
                        });
                        setError('Recording too short. Please record for at least 0.5 seconds.');
                        setIsRecording(false);
                        return;
                    }
                    
                    // Convert to base64 using chunked approach to avoid stack overflow
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);
                    
                    audioLogger.debug('Audio blob conversion details', {
                        originalSize: audioBlob.size,
                        blobType: audioBlob.type,
                        arrayBufferLength: arrayBuffer.byteLength,
                        uint8ArrayLength: uint8Array.length,
                        stateDuration: recordingDuration,
                        actualDuration: actualDuration
                    });
                    
                    // Convert WebM/Opus audio to raw PCM data
                    console.log('🔄 Converting WebM/Opus audio to PCM format');
                    uiLogger.info('🔄 Audio Conversion: WebM to PCM', {
                        action: 'audio_conversion',
                        inputSize: audioBlob.size,
                        inputType: audioBlob.type,
                        step: 'webm_to_pcm',
                        timestamp: new Date().toISOString()
                    });
                    
                    const conversionResult = await audioConverter.convertToPCM(audioBlob);
                    
                    // Convert PCM data to base64 for transmission
                    console.log('🔄 Converting PCM to base64 for transmission');
                    uiLogger.info('🔄 Audio Conversion: PCM to Base64', {
                        action: 'audio_conversion',
                        pcmSize: conversionResult.pcmData.length,
                        sampleRate: conversionResult.sampleRate,
                        step: 'pcm_to_base64',
                        timestamp: new Date().toISOString()
                    });
                    
                    // Use optimized base64 conversion for WebSocket transmission
                    const base64Audio = audioConverter.convertPCMToBase64(conversionResult.pcmData);
                    
                    audioLogger.success('Audio converted to PCM and base64', {
                        originalSize: conversionResult.originalSize,
                        originalSizeMB: (conversionResult.originalSize / (1024 * 1024)).toFixed(2) + ' MB',
                        pcmLength: conversionResult.pcmData.length,
                        pcmDuration: (conversionResult.pcmData.length / conversionResult.sampleRate).toFixed(2) + 's',
                        base64Length: base64Audio.length,
                        base64LengthMB: (base64Audio.length / (1024 * 1024)).toFixed(2) + ' MB',
                        sampleRate: conversionResult.sampleRate,
                        duration: conversionResult.duration,
                        durationMinutes: (conversionResult.duration / 60).toFixed(2) + ' min',
                        compressionRatio: (base64Audio.length / conversionResult.originalSize).toFixed(2),
                        expectedPCMLength: Math.floor(conversionResult.duration * conversionResult.sampleRate),
                        pcmLengthMatch: conversionResult.pcmData.length === Math.floor(conversionResult.duration * conversionResult.sampleRate) ? '✅ MATCH' : '❌ MISMATCH'
                    });
                    
                    // Now set up WebSocket connection and send the audio data
                    uiLogger.dataFlow('UI', 'WebSocket', { 
                        audioLength: base64Audio.length,
                        operation: currentConfig.operation,
                        diarization: currentConfig.diarization,
                        actualDuration: actualDuration
                    }, 'Single recognition audio data');
                    
                    // Log to terminal for debugging
                    uiLogger.info('📤 WebSocket Data Transmission', {
                        action: 'websocket_transmission',
                        audioLength: base64Audio.length,
                        audioLengthMB: (base64Audio.length / (1024 * 1024)).toFixed(2) + ' MB',
                        operation: currentConfig.operation,
                        diarization: currentConfig.diarization,
                        actualDuration: actualDuration,
                        actualDurationMinutes: (actualDuration / 60).toFixed(2) + ' min',
                        pcmLength: conversionResult.pcmData.length,
                        pcmDuration: (conversionResult.pcmData.length / conversionResult.sampleRate).toFixed(2) + 's',
                        sampleRate: conversionResult.sampleRate,
                        timestamp: new Date().toISOString()
                    });
                    
                    await processSingleRecognitionMicrophone(conversionResult.pcmData, conversionResult.sampleRate);
                    
                } catch (error) {
                    console.error('❌ Error processing audio data:', error);
                    console.error('❌ Error stack:', error instanceof Error ? error.stack : 'No stack trace');
                    console.error('❌ Error details:', {
                        name: error instanceof Error ? error.name : 'Unknown',
                        message: error instanceof Error ? error.message : String(error),
                        cause: error instanceof Error ? error.cause : undefined
                    });
                    
                    uiLogger.error('Error processing audio data', { 
                        error: error instanceof Error ? {
                            name: error.name,
                            message: error.message,
                            stack: error.stack,
                            cause: error.cause
                        } : {
                            type: typeof error,
                            value: String(error)
                        },
                        mode: 'single',
                        stateDuration: recordingDuration,
                        actualDuration: actualDuration
                    });
                    
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    setError(`Failed to process audio: ${errorMessage}`);
                    setIsProcessing(false);
                    setIsRecording(false);
                }
            } else if (currentConfig?.mode === 'continuous' && continuousAudioRecorderRef.current) {
                // For continuous mode, stop the continuous audio recorder
                audioLogger.info('Stopping continuous recording mode with VAD');
                await continuousAudioRecorderRef.current.stopRecording();
            } else {
                // For continuous mode, just stop recording
                audioLogger.info('Stopping continuous recording mode');
                if (audioRecorderRef.current) {
                    audioRecorderRef.current.stopRecording();
                }
            }
        }
        
        if (durationIntervalRef.current) {
            clearInterval(durationIntervalRef.current);
        }
        
        // WebSocket cleanup is no longer needed for continuous mode
        // as we're using REST API for all microphone processing
        
        // Only set recording to false for continuous mode
        // For single mode, keep recording state until we get the result
        if (currentConfig?.mode === 'continuous') {
            setIsRecording(false);
        }
    }, [currentConfig]);

    const processOptimizedAudio = async (audioBlob: Blob, duration: number) => {
        try {
            console.log('🔄 Processing optimized audio', {
                audioSize: audioBlob.size,
                duration: duration,
                audioSizeMB: (audioBlob.size / (1024 * 1024)).toFixed(2) + ' MB'
            });

            // Initialize optimized converter if needed
            if (!optimizedAudioConverterRef.current) {
                optimizedAudioConverterRef.current = new OptimizedAudioConverter();
            }

            // Convert audio to PCM and base64
            const conversionResult = await optimizedAudioConverterRef.current.convertToPCM(audioBlob);
            
            console.log('✅ Optimized audio conversion completed', {
                pcmLength: conversionResult.pcmData.length,
                sampleRate: conversionResult.sampleRate,
                duration: conversionResult.duration,
                base64Length: conversionResult.base64Data.length,
                base64LengthMB: (conversionResult.base64Data.length / (1024 * 1024)).toFixed(2) + ' MB'
            });

            // Log the data being sent to backend
            console.log('📤 Frontend sending data to backend', {
              base64Length: conversionResult.base64Data.length,
              base64LengthMB: (conversionResult.base64Data.length / (1024 * 1024)).toFixed(2) + ' MB',
              pcmLength: conversionResult.pcmData.length,
              duration: conversionResult.duration,
              durationMinutes: (conversionResult.duration / 60).toFixed(2) + ' min',
              sampleRate: conversionResult.sampleRate,
              originalSize: conversionResult.originalSize,
              originalSizeMB: (conversionResult.originalSize / (1024 * 1024)).toFixed(2) + ' MB'
            });

            // Log to terminal for debugging
            uiLogger.info('📤 Frontend Data Transmission', {
              action: 'frontend_data_transmission',
              base64Length: conversionResult.base64Data.length,
              base64LengthMB: (conversionResult.base64Data.length / (1024 * 1024)).toFixed(2) + ' MB',
              pcmLength: conversionResult.pcmData.length,
              duration: conversionResult.duration,
              durationMinutes: (conversionResult.duration / 60).toFixed(2) + ' min',
              sampleRate: conversionResult.sampleRate,
              originalSize: conversionResult.originalSize,
              originalSizeMB: (conversionResult.originalSize / (1024 * 1024)).toFixed(2) + ' MB',
              timestamp: new Date().toISOString()
            });

            // Process the audio using the microphone REST API (like speech_demo.py)
            await processSingleRecognitionMicrophone(conversionResult.pcmData, conversionResult.sampleRate);

        } catch (error) {
            console.error('❌ Optimized audio processing failed:', error);
            setError(`Audio processing failed: ${error instanceof Error ? error.message : String(error)}`);
            setIsProcessing(false);
        }
    };

    const processSingleRecognitionMicrophone = async (pcmData: Float32Array, sampleRate: number) => {
        try {
            // Calculate actual duration for logging
            const actualDuration = recordingStartTimeRef.current 
                ? (Date.now() - recordingStartTimeRef.current) / 1000 
                : recordingDuration;
                
            uiLogger.processing('Processing single microphone audio (like speech_demo.py)', {
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                duration: actualDuration,
                operation: currentConfig?.operation,
                diarization: currentConfig?.diarization
            });
            
            setError('🔄 Processing audio... Please wait.');
            setIsProcessing(true);
            
            console.log('📤 Starting single microphone processing (like speech_demo.py)', {
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                duration: actualDuration,
                operation: currentConfig?.operation,
                diarization: currentConfig?.diarization
            });

            // Convert Float32Array to regular array for JSON serialization - EXACTLY like speech_demo.py
            const audioArray = Array.from(pcmData);

            // Send to single microphone endpoint - EXACTLY like speech_demo.py process_microphone_single
            const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            console.log('🔍 API Base URL:', apiBaseUrl);
            console.log('🔍 Full URL:', `${apiBaseUrl}/api/v1/recognize-microphone`);
            
            const response = await fetch(`${apiBaseUrl}/api/v1/recognize-microphone`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    audio_data: audioArray,
                    sample_rate: sampleRate,
                    enable_diarization: currentConfig?.diarization || false,
                    is_translation: currentConfig?.operation === 'translation',
                    target_language: 'en'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            console.log('📥 Received single microphone response:', result);
            
            if (result.success) {
                // Check if this is a diarization result
                if (result.segments && result.segments.length > 0) {
                    // Handle diarization result
                    const diarizationResult: DiarizationResult = {
                        segments: result.segments.map((segment: { speaker_id: string; text: string; start_time?: number; end_time?: number }) => ({
                            speaker_id: segment.speaker_id,
                            text: segment.text,
                            start_time: segment.start_time || 0.0,
                            end_time: segment.end_time || 0.0
                        })),
                        num_speakers: result.num_speakers || 0,
                        is_translation: currentConfig?.operation === 'translation',
                        enable_diarization: true,
                        timestamp: new Date().toISOString()
                    };
                    
                    setDiarizationResults(prev => [...prev, diarizationResult]);
                    console.log('✅ Diarization result processed:', {
                        numSpeakers: diarizationResult.num_speakers,
                        segmentsCount: diarizationResult.segments.length,
                        isTranslation: diarizationResult.is_translation
                    });
                } else {
                    // Handle regular recognition result
                    const recognitionResult: RecognitionResult = {
                        text: result.text || 'No text returned',
                        confidence: result.confidence || 0.0,
                        language: result.language || 'Unknown',
                        timestamp: new Date().toISOString(),
                        duration: actualDuration,
                        hasError: false
                    };
                    
                    setResults(prev => [...prev, recognitionResult]);
                }
                
                setError('');
            } else {
                setError(`Recognition failed: ${result.error || 'Unknown error'}`);
            }
            
            setIsProcessing(false);
            setIsRecording(false);
            
        } catch (error) {
            console.error('Error processing single microphone audio:', error);
            uiLogger.error('Failed to process single microphone audio', {
                error: error instanceof Error ? error.message : String(error),
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                operation: currentConfig?.operation,
                actualDuration: recordingStartTimeRef.current 
                    ? (Date.now() - recordingStartTimeRef.current) / 1000 
                    : recordingDuration
            });
            setError(`Failed to process audio: ${error instanceof Error ? error.message : String(error)}`);
            setIsProcessing(false);
            setIsRecording(false);
        }
    };


    const processContinuousRecognitionMicrophone = async (pcmData: Float32Array, sampleRate: number) => {
        try {
            // Calculate actual duration for logging
            const actualDuration = recordingStartTimeRef.current 
                ? (Date.now() - recordingStartTimeRef.current) / 1000 
                : recordingDuration;
                
            uiLogger.processing('Processing continuous microphone audio chunk (like speech_demo.py)', {
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                duration: actualDuration,
                operation: currentConfig?.operation,
                diarization: currentConfig?.diarization
            });
            
            console.log('📤 Starting continuous microphone processing (like speech_demo.py)', {
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                duration: actualDuration,
                operation: currentConfig?.operation,
                diarization: currentConfig?.diarization
            });

            // Convert Float32Array to regular array for JSON serialization - EXACTLY like speech_demo.py
            const audioArray = Array.from(pcmData);

            // Send to continuous microphone endpoint - EXACTLY like speech_demo.py process_microphone_continuous
            const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            console.log('🔍 API Base URL:', apiBaseUrl);
            console.log('🔍 Full URL:', `${apiBaseUrl}/api/v1/recognize-microphone-continuous`);
            
            const response = await fetch(`${apiBaseUrl}/api/v1/recognize-microphone-continuous`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    audio_data: audioArray,
                    sample_rate: sampleRate,
                    enable_diarization: currentConfig?.diarization || false,
                    is_translation: currentConfig?.operation === 'translation',
                    target_language: 'en'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            console.log('📥 Received continuous microphone response:', result);
            
            if (result.success) {
                // Check if this is a diarization result
                if (result.segments && result.segments.length > 0) {
                    // Handle diarization result
                    const diarizationResult: DiarizationResult = {
                        segments: result.segments.map((segment: { speaker_id: string; text: string; start_time?: number; end_time?: number }) => ({
                            speaker_id: segment.speaker_id,
                            text: segment.text,
                            start_time: segment.start_time || 0.0,
                            end_time: segment.end_time || 0.0
                        })),
                        num_speakers: result.num_speakers || 0,
                        is_translation: currentConfig?.operation === 'translation',
                        enable_diarization: true,
                        timestamp: new Date().toISOString()
                    };
                    
                    setDiarizationResults(prev => [...prev, diarizationResult]);
                    console.log('✅ Continuous diarization result processed:', {
                        numSpeakers: diarizationResult.num_speakers,
                        segmentsCount: diarizationResult.segments.length,
                        isTranslation: diarizationResult.is_translation
                    });
                } else {
                    // Handle regular recognition result
                    const recognitionResult: RecognitionResult = {
                        text: result.text || 'No text returned',
                        confidence: result.confidence || 0.0,
                        language: result.language || 'Unknown',
                        timestamp: new Date().toISOString(),
                        duration: actualDuration,
                        hasError: false
                    };
                    
                    setResults(prev => [...prev, recognitionResult]);
                }
                
                setError('');
            } else {
                console.warn(`Continuous recognition warning: ${result.error || 'Unknown error'}`);
            }
            
        } catch (error) {
            console.error('Error processing continuous microphone audio:', error);
            uiLogger.error('Failed to process continuous microphone audio', {
                error: error instanceof Error ? error.message : String(error),
                audioSamples: pcmData.length,
                sampleRate: sampleRate,
                operation: currentConfig?.operation,
                actualDuration: recordingStartTimeRef.current 
                    ? (Date.now() - recordingStartTimeRef.current) / 1000 
                    : recordingDuration
            });
            // Don't set error for continuous mode, just log it
        }
    };


    const handleFileUpload = useCallback(async (file: File) => {
        if (!currentConfig || currentConfig.category !== 'file') return;

        uiLogger.info('Starting file upload process', {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            command: currentConfig.id,
            operation: currentConfig.operation,
            diarization: currentConfig.diarization
        });

        try {
            setError(null);
            setIsProcessing(true);

            const startTime = performance.now();
            
            // Convert file to ArrayBuffer
            const arrayBuffer = await file.arrayBuffer();
            
            uiLogger.dataFlow('UI', 'API', {
                fileName: file.name,
                fileSize: file.size,
                arrayBufferSize: arrayBuffer.byteLength,
                operation: currentConfig.operation,
                diarization: currentConfig.diarization
            }, 'File upload data');

            // Show processing message based on operation type
            if (currentConfig.diarization) {
                setIsDiarizationProcessing(true);
                setError('🔄 Processing file with diarization... This may take 30-60 seconds for long audio files. Please wait while we analyze speaker segments...');
                uiLogger.info('File processing with diarization enabled', {
                    fileName: file.name,
                    estimatedTime: '30-60 seconds'
                });
            } else {
                setError('🔄 Processing file... Please wait.');
                uiLogger.info('File processing without diarization', {
                    fileName: file.name
                });
            }

            // Call appropriate API endpoint
            apiLogger.info('Calling processAudio API', {
                endpoint: 'processAudio',
                operation: currentConfig.operation,
                diarization: currentConfig.diarization,
                fileSize: arrayBuffer.byteLength
            });
            
            const response = await apiClientRef.current!.processAudio(
                arrayBuffer,
                currentConfig.operation === 'translation',
                'en',
                currentConfig.diarization
            );

            const endTime = performance.now();
            const processingTime = endTime - startTime;

            apiLogger.success('API response received', {
                processingTime: processingTime,
                responseType: response.segments ? 'diarization' : 'recognition',
                hasText: !!response.text,
                hasSegments: !!(response.segments && response.segments.length > 0),
                segmentsCount: response.segments?.length || 0,
                numSpeakers: response.num_speakers || 0
            });

            // Update performance metrics
            setPerformanceMetrics(prev => ({
                ...prev,
                total_requests: prev.total_requests + 1,
                successful_recognitions: prev.successful_recognitions + 1,
                avg_response_time: (prev.avg_response_time + processingTime) / 2,
                audio_processing: {
                    ...prev.audio_processing,
                    avg_processing_time: (prev.audio_processing.avg_processing_time + processingTime) / 2,
                    total_chunks: prev.audio_processing.total_chunks + 1
                }
            }));

            // Handle response - check if it's a diarization result or regular result
            if (response.segments && response.segments.length > 0) {
                // This is a diarization result
                const diarizationResult: DiarizationResult = {
                    segments: response.segments,
                    num_speakers: response.num_speakers || 0,
                    is_translation: currentConfig.operation === 'translation',
                    enable_diarization: true,
                    timestamp: new Date().toISOString()
                };
                
                uiLogger.success('Diarization result processed', {
                    numSpeakers: diarizationResult.num_speakers,
                    segmentsCount: diarizationResult.segments.length,
                    isTranslation: diarizationResult.is_translation
                });
                
                setDiarizationResults(prev => [...prev, diarizationResult]);
            } else if (response.text) {
                // Regular recognition result
                const recognitionResult: RecognitionResult = {
                    text: response.text,
                    confidence: response.confidence || 0,
                    language: response.language || 'unknown',
                    timestamp: new Date().toISOString(),
                    is_translation: currentConfig.operation === 'translation',
                    enable_diarization: currentConfig.diarization
                };
                
                uiLogger.success('Recognition result processed', {
                    text: recognitionResult.text,
                    confidence: recognitionResult.confidence,
                    language: recognitionResult.language,
                    isTranslation: recognitionResult.is_translation
                });
                
                setResults(prev => [...prev, recognitionResult]);
            } else {
                uiLogger.error('No valid response data received', { response: response });
                setError('No text or segments received from API');
            }
        } catch (err) {
            uiLogger.error('File processing failed', { 
                error: err,
                fileName: file.name,
                fileSize: file.size,
                command: currentConfig.id
            });
            setError(`File processing failed: ${err}`);
        } finally {
            setIsProcessing(false);
            setIsDiarizationProcessing(false);
        }
    }, [currentConfig]);

    const clearResults = useCallback(() => {
        console.log(`[UI] 🧹 Clear Results clicked (${results.length} results, ${diarizationResults.length} diarization)`);
        uiLogger.info('🧹 Clear Results clicked', {
            currentResultsCount: results.length,
            currentDiarizationResultsCount: diarizationResults.length,
            hasError: !!error
        });
        setResults([]);
        setDiarizationResults([]);
        setError(null);
    }, [results.length, diarizationResults.length, error]);

    const downloadResults = useCallback(() => {
        const allResults = [
            ...results.map(r => ({ type: 'recognition', ...r })),
            ...diarizationResults.map(d => ({ type: 'diarization', ...d }))
        ];
        
        console.log(`[UI] 📥 Download Results clicked (${allResults.length} total results)`);
        uiLogger.info('📥 Download Results clicked', {
            totalResults: allResults.length,
            recognitionResults: results.length,
            diarizationResults: diarizationResults.length,
            fileName: `speech_results_${Date.now()}.json`
        });
        
        const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `speech_results_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }, [results, diarizationResults]);

    return (
        <div className="min-h-screen bg-gray-50 p-6">
            <div className="max-w-7xl mx-auto">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">
                        🎤 Enhanced Speech Demo
                    </h1>
                    <p className="text-gray-600">
                        Test all 8 speech_demo.py commands through an intuitive web interface
                    </p>
                </div>

                {/* Command Selection */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                    <h2 className="text-xl font-semibold mb-4">Select Command</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {COMMAND_CONFIGS.map((config) => (
                            <button
                                key={config.id}
                                onClick={() => {
                                    console.log(`[UI] 🎯 Command selected: ${config.name} (${config.id})`);
                                    uiLogger.info('🎯 Command selected', {
                                        commandId: config.id,
                                        commandName: config.name,
                                        category: config.category,
                                        mode: config.mode,
                                        operation: config.operation,
                                        diarization: config.diarization
                                    });
                                    setSelectedCommand(config.id);
                                }}
                                className={`p-4 rounded-lg border-2 transition-all ${
                                    selectedCommand === config.id
                                        ? 'border-blue-500 bg-blue-50'
                                        : 'border-gray-200 hover:border-gray-300'
                                }`}
                            >
                                <div className="flex items-center space-x-3 mb-2">
                                    {config.icon}
                                    <span className="font-medium">{config.name}</span>
                                </div>
                                <p className="text-sm text-gray-600 text-left">
                                    {config.description}
                                </p>
                                <div className="mt-2 flex flex-wrap gap-1">
                                    <span className={`px-2 py-1 text-xs rounded ${
                                        config.category === 'file' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                                    }`}>
                                        {config.category}
                                    </span>
                                    <span className={`px-2 py-1 text-xs rounded ${
                                        config.operation === 'transcription' ? 'bg-purple-100 text-purple-800' : 'bg-orange-100 text-orange-800'
                                    }`}>
                                        {config.operation}
                                    </span>
                                    {config.diarization && (
                                        <span className="px-2 py-1 text-xs rounded bg-pink-100 text-pink-800">
                                            diarization
                                        </span>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Action Panel */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                    <h2 className="text-xl font-semibold mb-4">Actions</h2>
                    
                    {currentConfig?.category === 'file' ? (
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Upload Audio File
                                </label>
                                <input
                                    type="file"
                                    accept=".wav"
                                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                                />
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="flex items-center space-x-4">
                                <button
                                    onClick={isRecording ? stopRecording : startRecording}
                                    disabled={isProcessing}
                                    className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                                        isRecording
                                            ? 'bg-red-500 text-white hover:bg-red-600'
                                            : 'bg-green-500 text-white hover:bg-green-600'
                                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                                >
                                    {isRecording ? <Square className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                                    <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
                                </button>
                                
                                {isRecording && (
                                    <div className="text-sm text-gray-600 space-y-2">
                                        <div>Duration: {recordingDuration.toFixed(1)}s</div>
                                        {currentConfig?.mode === 'continuous' && (
                                            <div className="space-y-1">
                                                <div className="flex items-center space-x-2">
                                                    <span>Audio Level:</span>
                                                    <div className="flex space-x-1">
                                                        {Array.from({ length: 20 }, (_, i) => (
                                                            <div
                                                                key={i}
                                                                className={`w-1 h-3 ${
                                                                    i < audioLevel * 20 
                                                                        ? 'bg-green-500' 
                                                                        : 'bg-gray-300'
                                                                }`}
                                                            />
                                                        ))}
                                                    </div>
                                                    <span className="text-xs">{(audioLevel * 100).toFixed(0)}%</span>
                                                </div>
                                                <div className="text-xs text-gray-500">
                                                    Status: {recordingStatus}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    <div className="flex items-center space-x-4 mt-4">
                        <button
                            onClick={clearResults}
                            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                        >
                            <RotateCcw className="w-4 h-4" />
                            <span>Clear Results</span>
                        </button>
                        
                        <button
                            onClick={downloadResults}
                            disabled={results.length === 0 && diarizationResults.length === 0}
                            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Download className="w-4 h-4" />
                            <span>Download Results</span>
                        </button>

                        <button
                            onClick={() => {
                                uiLogger.info('📊 Performance Metrics toggle clicked', {
                                    currentState: showMetrics,
                                    newState: !showMetrics,
                                    metrics: performanceMetrics
                                });
                                setShowMetrics(!showMetrics);
                            }}
                            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                        >
                            <Settings className="w-4 h-4" />
                            <span>Performance Metrics</span>
                        </button>
                    </div>
                    
                    {/* Optimized Recorder Toggle */}
                    <div className="flex items-center space-x-2 p-3 bg-gray-50 rounded-lg border">
                        <input
                            type="checkbox"
                            id="optimized-recorder"
                            checked={useOptimizedRecorder}
                            onChange={(e) => setUseOptimizedRecorder(e.target.checked)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="optimized-recorder" className="text-sm text-gray-700 cursor-pointer">
                            🚀 Use Optimized Recorder (for 60+ min recordings)
                        </label>
                        <div className="text-xs text-gray-500 ml-2">
                            {useOptimizedRecorder ? '✅ Enabled' : '❌ Disabled'}
                        </div>
                    </div>
                </div>

                {/* Results Display */}
                <div className="space-y-6">
                    {/* Error Display */}
                    {error && (
                        <div className={`border rounded-lg p-4 ${error.includes('🔄') ? 'bg-blue-50 border-blue-200' : 'bg-red-50 border-red-200'}`}>
                            <div className="flex items-center space-x-2">
                                <div className={`w-5 h-5 ${error.includes('🔄') ? 'text-blue-500' : 'text-red-500'}`}>
                                    {error.includes('🔄') ? '🔄' : '⚠️'}
                                </div>
                                <span className={error.includes('🔄') ? 'text-blue-800' : 'text-red-800'}>{error}</span>
                            </div>
                            {isDiarizationProcessing && (
                                <div className="mt-4">
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '100%'}}></div>
                                    </div>
                                    <p className="text-sm text-blue-600 mt-2">Analyzing speaker segments with Pyannote.audio...</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Recognition Results */}
                    {results.length > 0 && (
                        <div className="bg-white rounded-lg shadow-md p-6">
                            <h3 className="text-lg font-semibold mb-4">Recognition Results</h3>
                            <div className="space-y-4">
                                {results.map((result, index) => (
                                    <div key={index} className="border rounded-lg p-4">
                                        <div className="flex justify-between items-start mb-2">
                                            <span className="text-sm text-gray-500">Result #{index + 1}</span>
                                            <span className="text-sm text-gray-500">
                                                {new Date(result.timestamp).toLocaleTimeString()}
                                            </span>
                                        </div>
                                        <p className="text-gray-900 mb-2">{result.text}</p>
                                        <div className="flex space-x-4 text-sm text-gray-600">
                                            <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                                            <span>Language: {result.language}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Diarization Results */}
                    {diarizationResults.length > 0 && (
                        <div className="bg-white rounded-lg shadow-md p-6">
                            <h3 className="text-lg font-semibold mb-4">Diarization Results</h3>
                            <div className="space-y-4">
                                {diarizationResults.map((result, index) => (
                                    <div key={index} className="border rounded-lg p-4">
                                        <div className="flex justify-between items-start mb-4">
                                            <span className="text-sm text-gray-500">Diarization #{index + 1}</span>
                                            <div className="text-sm text-gray-500">
                                                <span className="mr-4">Speakers: {result.num_speakers}</span>
                                                <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            {result.segments.map((segment, segIndex) => (
                                                <div key={segIndex} className="flex items-start space-x-3 p-2 bg-gray-50 rounded">
                                                    <span className="font-medium text-blue-600 min-w-[100px]">
                                                        {segment.speaker_id}:
                                                    </span>
                                                    <span className="text-gray-900">{segment.text}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Performance Metrics */}
                    {showMetrics && (
                        <PerformanceMetricsComponent 
                            timingMetrics={{
                                total_time: performanceMetrics.avg_response_time,
                                api_call: performanceMetrics.avg_response_time * 0.8,
                                response_processing: performanceMetrics.avg_response_time * 0.2
                            }}
                            performanceMetrics={performanceMetrics}
                            recentResults={results.map(r => ({
                                timestamp: r.timestamp,
                                timing_metrics: {
                                    total_time: performanceMetrics.avg_response_time
                                }
                            }))}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};