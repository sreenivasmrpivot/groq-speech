'use client';

import { AudioRecorder } from '@/lib/audio-recorder';
import { GroqAPIClient } from '@/lib/groq-api';
import { PerformanceMetrics, RecognitionResult, DiarizationResult } from '@/types';
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
    useMockApi?: boolean;
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
    endpoint: 'rest' | 'websocket';
}

const COMMAND_CONFIGS: CommandConfig[] = [
    // File-based commands (REST API)
    {
        id: 'file_transcription',
        name: 'File Transcription',
        description: 'Transcribe audio file without diarization',
        icon: <FileText className="w-5 h-5" />,
        category: 'file',
        mode: 'single',
        operation: 'transcription',
        diarization: false,
        endpoint: 'rest'
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
        endpoint: 'rest'
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
        endpoint: 'rest'
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
        endpoint: 'rest'
    },
    // Microphone single commands (WebSocket)
    {
        id: 'microphone_single',
        name: 'Single Microphone',
        description: 'Record once, then transcribe',
        icon: <Mic className="w-5 h-5" />,
        category: 'microphone',
        mode: 'single',
        operation: 'transcription',
        diarization: false,
        endpoint: 'websocket'
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
        endpoint: 'websocket'
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
        endpoint: 'websocket'
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
        endpoint: 'websocket'
    },
    // Microphone continuous commands (WebSocket)
    {
        id: 'microphone_continuous',
        name: 'Continuous Microphone',
        description: 'Real-time transcription',
        icon: <Mic className="w-5 h-5" />,
        category: 'microphone',
        mode: 'continuous',
        operation: 'transcription',
        diarization: false,
        endpoint: 'websocket'
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
        endpoint: 'websocket'
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
        endpoint: 'websocket'
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
        endpoint: 'websocket'
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
    const [isDiarizationProcessing, setIsDiarizationProcessing] = useState(false);
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
    const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const apiClientRef = useRef<GroqAPIClient | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);

    const currentConfig = COMMAND_CONFIGS.find(cmd => cmd.id === selectedCommand);

    useEffect(() => {
        apiClientRef.current = new GroqAPIClient();
        
        // Cleanup on unmount
        return () => {
            if (websocketRef.current) {
                try {
                    if (websocketRef.current.readyState === WebSocket.OPEN) {
                        websocketRef.current.close();
                    }
                } catch (error) {
                    console.warn('Error closing WebSocket on unmount:', error);
                }
                websocketRef.current = null;
            }
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
            }
        };
    }, []);

    const startRecording = useCallback(async () => {
        if (!currentConfig || currentConfig.category !== 'microphone') return;

        try {
            setError(null);
            setIsRecording(true);
            setRecordingDuration(0);

            const audioRecorder = new AudioRecorder();
            audioRecorderRef.current = audioRecorder;

            // Start duration timer
            durationIntervalRef.current = setInterval(() => {
                setRecordingDuration(prev => prev + 0.1);
            }, 100);

            if (currentConfig.mode === 'continuous') {
                await startContinuousRecognition();
            } else {
                await startSingleRecognition();
            }
        } catch (err) {
            setError(`Recording failed: ${err}`);
            setIsRecording(false);
        }
    }, [currentConfig]);

    const stopRecording = useCallback(() => {
        if (audioRecorderRef.current) {
            audioRecorderRef.current.stopRecording();
        }
        if (durationIntervalRef.current) {
            clearInterval(durationIntervalRef.current);
        }
        if (websocketRef.current) {
            try {
                if (currentConfig?.mode === 'continuous' && websocketRef.current.readyState === WebSocket.OPEN) {
                    // Send stop message for continuous mode
                    apiClientRef.current?.sendStopRecognition(websocketRef.current);
                }
            } catch (error) {
                console.warn('Error sending stop message:', error);
            } finally {
                websocketRef.current.close();
                websocketRef.current = null;
            }
        }
        setIsRecording(false);
    }, [currentConfig]);

    const startSingleRecognition = async () => {
        try {
            console.log('Starting single recognition...');
            
            // Connect to WebSocket and wait for connection to be established
            const ws = await new Promise<WebSocket>((resolve, reject) => {
                const wsUrl = (apiClientRef.current as any).baseUrl.replace('http', 'ws') + '/ws/recognize';
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('🔌 WebSocket connected for single recognition');
                    resolve(ws);
                };
                
                ws.onerror = (error) => {
                    console.error('💥 WebSocket connection error:', error);
                    reject(new Error('WebSocket connection failed'));
                };
                
                ws.onclose = (event) => {
                    if (event.code !== 1000) {
                        reject(new Error(`WebSocket closed unexpectedly: ${event.code}`));
                    }
                };
            });
            
            websocketRef.current = ws;
            
            // Set up message handlers
            ws.onmessage = (event: MessageEvent<string>) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('📨 Single recognition message:', data);
                    
                    if (data.type === 'recognition_result') {
                        const result: RecognitionResult = {
                            text: data.data.text || '',
                            confidence: data.data.confidence || 0.95,
                            language: data.data.language || 'auto-detected',
                            timestamps: data.data.timestamps || [],
                            timestamp: new Date().toISOString(),
                            is_translation: data.data.is_translation || false,
                            enable_diarization: data.data.enable_diarization || false,
                        };
                        
                        // Check if it's a diarization result
                        if (result.segments && result.segments.length > 0) {
                            const diarizationResult: DiarizationResult = {
                                segments: result.segments,
                                num_speakers: result.num_speakers || 0,
                                is_translation: currentConfig?.operation === 'translation',
                                enable_diarization: true,
                                timestamp: new Date().toISOString()
                            };
                            setDiarizationResults(prev => [...prev, diarizationResult]);
                        } else {
                            setResults(prev => [...prev, result]);
                        }
                    } else if (data.type === 'diarization_result') {
                        const diarizationResult: DiarizationResult = {
                            segments: data.data.segments || [],
                            num_speakers: data.data.num_speakers || 0,
                            is_translation: currentConfig?.operation === 'translation',
                            enable_diarization: true,
                            timestamp: new Date().toISOString()
                        };
                        setDiarizationResults(prev => [...prev, diarizationResult]);
                    } else if (data.type === 'error') {
                        setError(data.data?.message || 'Unknown error occurred');
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    setError('Failed to parse server response');
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setError('WebSocket connection error');
            };
            
            ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                if (event.code !== 1000 && event.code !== 1005) {
                    setError(`WebSocket closed unexpectedly: ${event.code} ${event.reason}`);
                }
            };
            
            // Now send the single recognition message
            ws.send(JSON.stringify({
                type: 'single_recognition',
                data: {
                    is_translation: currentConfig?.operation === 'translation',
                    enable_diarization: currentConfig?.diarization || false,
                    target_language: 'en'
                }
            }));
            
        } catch (error) {
            console.error('Error starting single recognition:', error);
            setError(`Failed to start single recognition: ${error}`);
        }
    };

    const startContinuousRecognition = async () => {
        try {
            console.log('Starting continuous recognition...');
            
            // Connect to WebSocket and wait for connection to be established
            const ws = await new Promise<WebSocket>((resolve, reject) => {
                const wsUrl = (apiClientRef.current as any).baseUrl.replace('http', 'ws') + '/ws/recognize';
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('🔌 WebSocket connected for continuous recognition');
                    resolve(ws);
                };
                
                ws.onerror = (error) => {
                    console.error('💥 WebSocket connection error:', error);
                    reject(new Error('WebSocket connection failed'));
                };
                
                ws.onclose = (event) => {
                    if (event.code !== 1000) {
                        reject(new Error(`WebSocket closed unexpectedly: ${event.code}`));
                    }
                };
            });
            
            websocketRef.current = ws;
            
            // Set up message handlers
            ws.onmessage = (event: MessageEvent<string>) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('📨 Continuous recognition message:', data);
                    
                    if (data.type === 'recognition_result') {
                        const result: RecognitionResult = {
                            text: data.data.text || '',
                            confidence: data.data.confidence || 0.95,
                            language: data.data.language || 'auto-detected',
                            timestamps: data.data.timestamps || [],
                            timestamp: new Date().toISOString(),
                            is_translation: data.data.is_translation || false,
                            enable_diarization: data.data.enable_diarization || false,
                        };
                        
                        // Check if it's a diarization result
                        if (result.segments && result.segments.length > 0) {
                            const diarizationResult: DiarizationResult = {
                                segments: result.segments,
                                num_speakers: result.num_speakers || 0,
                                is_translation: currentConfig?.operation === 'translation',
                                enable_diarization: true,
                                timestamp: new Date().toISOString()
                            };
                            setDiarizationResults(prev => [...prev, diarizationResult]);
                        } else {
                            setResults(prev => [...prev, result]);
                        }
                    } else if (data.type === 'diarization_result') {
                        const diarizationResult: DiarizationResult = {
                            segments: data.data.segments || [],
                            num_speakers: data.data.num_speakers || 0,
                            is_translation: currentConfig?.operation === 'translation',
                            enable_diarization: true,
                            timestamp: new Date().toISOString()
                        };
                        setDiarizationResults(prev => [...prev, diarizationResult]);
                    } else if (data.type === 'error') {
                        setError(data.data?.message || 'Unknown error occurred');
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    setError('Failed to parse server response');
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setError('WebSocket connection error');
            };
            
            ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                if (event.code !== 1000 && event.code !== 1005) {
                    setError(`WebSocket closed unexpectedly: ${event.code} ${event.reason}`);
                }
            };
            
            // Now send the start recognition message
            ws.send(JSON.stringify({
                type: 'start_recognition',
                data: {
                    is_translation: currentConfig?.operation === 'translation',
                    enable_diarization: currentConfig?.diarization || false,
                    target_language: 'en',
                    mode: 'continuous'
                }
            }));
            
        } catch (error) {
            console.error('Error starting continuous recognition:', error);
            setError(`Failed to start continuous recognition: ${error}`);
        }
    };

    const handleFileUpload = useCallback(async (file: File) => {
        if (!currentConfig || currentConfig.category !== 'file') return;

        try {
            setError(null);
            setIsProcessing(true);

            const startTime = performance.now();
            
            // Convert file to ArrayBuffer
            const arrayBuffer = await file.arrayBuffer();

            // Show processing message based on operation type
            if (currentConfig.diarization) {
                setIsDiarizationProcessing(true);
                setError('🔄 Processing file with diarization... This may take 30-60 seconds for long audio files. Please wait while we analyze speaker segments...');
            } else {
                setError('🔄 Processing file... Please wait.');
            }

            // Call appropriate API endpoint
            const response = await apiClientRef.current!.transcribeAudio(
                arrayBuffer,
                currentConfig.operation === 'translation',
                'en',
                currentConfig.diarization
            );

            const endTime = performance.now();
            const processingTime = endTime - startTime;

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
                setResults(prev => [...prev, recognitionResult]);
            } else {
                setError('No text or segments received from API');
            }
        } catch (err) {
            setError(`File processing failed: ${err}`);
        } finally {
            setIsProcessing(false);
            setIsDiarizationProcessing(false);
        }
    }, [currentConfig]);

    const clearResults = useCallback(() => {
        setResults([]);
        setDiarizationResults([]);
        setError(null);
    }, []);

    const downloadResults = useCallback(() => {
        const allResults = [
            ...results.map(r => ({ type: 'recognition', ...r })),
            ...diarizationResults.map(d => ({ type: 'diarization', ...d }))
        ];
        
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
                                onClick={() => setSelectedCommand(config.id)}
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
                                    <span className={`px-2 py-1 text-xs rounded ${
                                        config.endpoint === 'rest' ? 'bg-gray-100 text-gray-800' : 'bg-yellow-100 text-yellow-800'
                                    }`}>
                                        {config.endpoint}
                                    </span>
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
                                    <div className="text-sm text-gray-600">
                                        Duration: {recordingDuration.toFixed(1)}s
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
                            onClick={() => setShowMetrics(!showMetrics)}
                            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                        >
                            <Settings className="w-4 h-4" />
                            <span>Performance Metrics</span>
                        </button>
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
