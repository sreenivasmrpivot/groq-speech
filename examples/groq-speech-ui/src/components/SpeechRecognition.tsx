'use client';

import { AudioRecorder } from '@/lib/audio-recorder';
import { GroqAPIClient, MockGroqAPIClient } from '@/lib/groq-api';
import { PerformanceMetrics, RecognitionMode, RecognitionResult } from '@/types';
import {
    Download,
    FileText,
    Globe,
    Mic,
    Play,
    RotateCcw,
    Settings,
    Square
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { PerformanceMetricsComponent } from './PerformanceMetrics';

interface SpeechRecognitionProps {
    useMockApi?: boolean;
}

export const SpeechRecognitionComponent: React.FC<SpeechRecognitionProps> = ({
    useMockApi = false,
}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>({
        type: 'single',
        operation: 'transcription',
    });
    const [results, setResults] = useState<RecognitionResult[]>([]);
    const [currentResult, setCurrentResult] = useState<RecognitionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [targetLanguage, setTargetLanguage] = useState('en');
    const [selectedLanguage, setSelectedLanguage] = useState('en-US');
    const [recordingDuration, setRecordingDuration] = useState(0);
    const [showMetrics, setShowMetrics] = useState(false);

    const audioRecorderRef = useRef<AudioRecorder | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);
    const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const apiClientRef = useRef<GroqAPIClient | MockGroqAPIClient | null>(null);

    // Performance metrics
    const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
        total_requests: 0,
        successful_recognitions: 0,
        failed_recognitions: 0,
        avg_response_time: 0,
        audio_processing: {
            avg_processing_time: 0,
            total_chunks: 0,
            buffer_size: 0,
        },
    });

    // Initialize API client
    useEffect(() => {
        apiClientRef.current = useMockApi
            ? new MockGroqAPIClient()
            : new GroqAPIClient();
    }, [useMockApi]);

    // Initialize audio recorder
    useEffect(() => {
        audioRecorderRef.current = new AudioRecorder();

        audioRecorderRef.current.setOnRecordingComplete(async (audioBlob, duration) => {
            setIsRecording(false);
            setIsProcessing(true);
            setRecordingDuration(0);

            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
            }

            try {
                // For single mode, send audio data to backend via REST API
                if (recognitionMode.type === 'single' && apiClientRef.current) {
                    console.log('Processing single mode audio:', audioBlob.size, 'bytes');

                    // Convert Blob to ArrayBuffer for the API call
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    console.log('Converted to ArrayBuffer:', arrayBuffer.byteLength, 'bytes');

                    const result = await apiClientRef.current.transcribeAudio(
                        arrayBuffer,
                        recognitionMode.operation === 'translation',
                        targetLanguage
                    );

                    console.log('Single mode result received:', result);

                    // Update performance metrics
                    setPerformanceMetrics(prev => ({
                        ...prev,
                        total_requests: prev.total_requests + 1,
                        successful_recognitions: prev.successful_recognitions + 1,
                        avg_response_time: (prev.avg_response_time * prev.total_requests + (result.timing_metrics?.total_time || 0)) / (prev.total_requests + 1),
                    }));

                    setResults(prev => [...prev, result]);
                    setCurrentResult(result);
                }
            } catch (err) {
                console.error('Recognition error:', err);
                setError(err instanceof Error ? err.message : 'Recognition failed');

                setPerformanceMetrics(prev => ({
                    ...prev,
                    total_requests: prev.total_requests + 1,
                    failed_recognitions: prev.failed_recognitions + 1,
                }));
            } finally {
                setIsProcessing(false);
            }
        });

        return () => {
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
            }
        };
    }, [targetLanguage, recognitionMode]);

    const startRecording = useCallback(async () => {
        if (!audioRecorderRef.current) return;

        try {
            setError(null);
            setIsRecording(true);
            setCurrentResult(null);
            setRecordingDuration(0);

            // Check microphone availability first
            const isMicrophoneAvailable = await AudioRecorder.checkMicrophoneAvailability();
            if (!isMicrophoneAvailable) {
                throw new Error('Microphone is not available. Please check your microphone connection and permissions.');
            }

            console.log('Starting single mode recording...');
            await audioRecorderRef.current.startRecording();

            // Start duration timer
            durationIntervalRef.current = setInterval(() => {
                if (audioRecorderRef.current) {
                    setRecordingDuration(audioRecorderRef.current.getRecordingDuration());
                }
            }, 100);

        } catch (err) {
            console.error('Failed to start recording:', err);
            setError(err instanceof Error ? err.message : 'Failed to access microphone. Please check permissions.');
            setIsRecording(false);
        }
    }, []);

    const stopRecording = useCallback(() => {
        if (audioRecorderRef.current && isRecording) {
            audioRecorderRef.current.stopRecording();
        }
    }, [isRecording]);

    const startContinuousRecognition = useCallback(async () => {
        if (!apiClientRef.current) return;

        try {
            setError(null);
            setIsRecording(true);
            setCurrentResult(null);

            websocketRef.current = await apiClientRef.current.transcribeWithWebSocket(
                (result) => {
                    console.log('Received recognition result:', result);
                    setResults(prev => [...prev, result]);
                    setCurrentResult(result);

                    setPerformanceMetrics(prev => ({
                        ...prev,
                        total_requests: prev.total_requests + 1,
                        successful_recognitions: prev.successful_recognitions + 1,
                        avg_response_time: (prev.avg_response_time * prev.total_requests + (result.timing_metrics?.total_time || 0)) / (prev.total_requests + 1),
                    }));
                },
                (error) => {
                    console.error('WebSocket error received:', error);
                    setError(error);
                    setPerformanceMetrics(prev => ({
                        ...prev,
                        total_requests: prev.total_requests + 1,
                        failed_recognitions: prev.failed_recognitions + 1,
                    }));
                },
                selectedLanguage,
                recognitionMode.operation === 'translation'
            );

            console.log('WebSocket connection established:', websocketRef.current.readyState);

            // Wait for WebSocket to be ready
            if (websocketRef.current.readyState !== WebSocket.OPEN) {
                console.log('Waiting for WebSocket to be ready...');
                await new Promise<void>((resolve, reject) => {
                    const checkState = () => {
                        if (websocketRef.current?.readyState === WebSocket.OPEN) {
                            console.log('WebSocket is now ready');
                            resolve();
                        } else if (websocketRef.current?.readyState === WebSocket.CLOSED) {
                            reject(new Error('WebSocket connection closed'));
                        } else {
                            setTimeout(checkState, 100);
                        }
                    };
                    checkState();
                });
            }

            // Set up audio recording for continuous recognition
            if (audioRecorderRef.current) {
                // Set up data available callback BEFORE starting recording
                console.log('Setting up data available callback for continuous mode...');
                audioRecorderRef.current.setOnDataAvailable(async (chunk) => {
                    console.log('Audio chunk received in continuous mode:', chunk.size, 'bytes');
                    // Send audio chunk to WebSocket for real-time processing
                    if (websocketRef.current && apiClientRef.current) {
                        try {
                            console.log('Sending audio chunk to WebSocket...');
                            await (apiClientRef.current as any).sendAudioData(websocketRef.current, chunk);
                        } catch (error) {
                            console.error('Error sending audio chunk:', error);
                            setError('Failed to send audio chunk: ' + (error as Error).message);
                        }
                    } else {
                        console.warn('WebSocket or API client not available');
                    }
                });

                // Start recording for continuous mode AFTER setting up callback
                console.log('Starting continuous mode recording...');
                await audioRecorderRef.current.startRecording();
            }

        } catch (err) {
            console.error('Failed to start continuous recognition:', err);
            setError('Failed to start continuous recognition');
            setIsRecording(false);
        }
    }, [selectedLanguage, recognitionMode]);

    const stopContinuousRecognition = useCallback(() => {
        if (websocketRef.current) {
            websocketRef.current.close();
            websocketRef.current = null;
        }

        // Stop the audio recording
        if (audioRecorderRef.current && isRecording) {
            audioRecorderRef.current.stopRecording();
        }

        setIsRecording(false);
    }, [isRecording]);

    const handleStart = useCallback(() => {
        if (recognitionMode.type === 'continuous') {
            startContinuousRecognition();
        } else {
            startRecording();
        }
    }, [recognitionMode.type, startRecording, startContinuousRecognition]);

    const handleStop = useCallback(() => {
        if (recognitionMode.type === 'continuous') {
            stopContinuousRecognition();
        } else {
            stopRecording();
        }
    }, [recognitionMode.type, stopRecording, stopContinuousRecognition]);

    const clearResults = useCallback(() => {
        setResults([]);
        setCurrentResult(null);
        setError(null);
    }, []);

    const exportResults = useCallback(() => {
        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `groq-speech-results-${new Date().toISOString()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }, [results]);

    const formatDuration = (ms: number) => {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    const getCurrentTimingMetrics = () => {
        if (currentResult?.timing_metrics) {
            return currentResult.timing_metrics;
        }
        return {
            total_time: 0,
            api_call: 0,
            response_processing: 0,
        };
    };

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-6">
            {/* Header */}
            <div className="bg-white rounded-lg shadow-md border p-6">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900">
                            Groq Speech Recognition
                        </h1>
                        <p className="text-gray-600">
                            Real-time transcription and translation with performance metrics
                            {useMockApi && <span className="text-blue-600"> (Mock Mode)</span>}
                        </p>
                    </div>
                    <div className="flex items-center space-x-2">
                        <button
                            onClick={() => setShowMetrics(!showMetrics)}
                            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        >
                            <Settings className="h-4 w-4 mr-2" />
                            {showMetrics ? 'Hide' : 'Show'} Metrics
                        </button>
                    </div>
                </div>

                {/* Mode Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Recognition Mode
                        </label>
                        <div className="flex space-x-2">
                            <button
                                onClick={() => setRecognitionMode(prev => ({ ...prev, type: 'single' }))}
                                className={`px-4 py-2 rounded-lg border transition-colors ${recognitionMode.type === 'single'
                                    ? 'bg-blue-600 text-white border-blue-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                                    }`}
                            >
                                <Mic className="h-4 w-4 inline mr-2" />
                                Single Shot
                            </button>
                            <button
                                onClick={() => setRecognitionMode(prev => ({ ...prev, type: 'continuous' }))}
                                className={`px-4 py-2 rounded-lg border transition-colors ${recognitionMode.type === 'continuous'
                                    ? 'bg-blue-600 text-white border-blue-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                                    }`}
                            >
                                <Play className="h-4 w-4 inline mr-2" />
                                Continuous
                            </button>
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Operation Type
                        </label>
                        <div className="flex space-x-2">
                            <button
                                onClick={() => setRecognitionMode(prev => ({ ...prev, operation: 'transcription' }))}
                                className={`px-4 py-2 rounded-lg border transition-colors ${recognitionMode.operation === 'transcription'
                                    ? 'bg-green-600 text-white border-green-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                                    }`}
                            >
                                <FileText className="h-4 w-4 inline mr-2" />
                                Transcription
                            </button>
                            <button
                                onClick={() => setRecognitionMode(prev => ({ ...prev, operation: 'translation' }))}
                                className={`px-4 py-2 rounded-lg border transition-colors ${recognitionMode.operation === 'translation'
                                    ? 'bg-green-600 text-white border-green-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                                    }`}
                            >
                                <Globe className="h-4 w-4 inline mr-2" />
                                Translation
                            </button>
                        </div>
                    </div>
                </div>

                {/* Language Selection */}
                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Language
                    </label>
                    <select
                        value={selectedLanguage}
                        onChange={(e) => setSelectedLanguage(e.target.value)}
                        className="w-full md:w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                        <option value="en-US">English (US)</option>
                        <option value="de-DE">German</option>
                        <option value="fr-FR">French</option>
                        <option value="es-ES">Spanish</option>
                        <option value="it-IT">Italian</option>
                        <option value="pt-BR">Portuguese (Brazil)</option>
                        <option value="ja-JP">Japanese</option>
                        <option value="ko-KR">Korean</option>
                        <option value="zh-CN">Chinese (Simplified)</option>
                        <option value="ru-RU">Russian</option>
                    </select>
                </div>

                {/* Recording Controls */}
                <div className="flex items-center justify-center space-x-4">
                    {!isRecording ? (
                        <button
                            onClick={handleStart}
                            disabled={isProcessing}
                            className="flex items-center px-6 py-3 bg-red-600 text-white rounded-full hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <Mic className="h-5 w-5 mr-2" />
                            Start Recording
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            className="flex items-center px-6 py-3 bg-gray-600 text-white rounded-full hover:bg-gray-700 transition-colors"
                        >
                            <Square className="h-5 w-5 mr-2" />
                            Stop Recording
                        </button>
                    )}

                    {recordingDuration > 0 && (
                        <div className="text-lg font-mono text-gray-700">
                            {formatDuration(recordingDuration)}
                        </div>
                    )}

                    <button
                        onClick={clearResults}
                        className="flex items-center px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                    >
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Clear
                    </button>

                    {results.length > 0 && (
                        <button
                            onClick={exportResults}
                            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                        >
                            <Download className="h-4 w-4 mr-2" />
                            Export
                        </button>
                    )}
                </div>

                {/* Error Display */}
                {error && (
                    <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                        {error}
                    </div>
                )}

                {/* Processing Indicator */}
                {isProcessing && (
                    <div className="mt-4 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <span className="ml-2 text-gray-600">Processing...</span>
                    </div>
                )}
            </div>

            {/* Current Result */}
            {currentResult && (
                <div className="bg-white rounded-lg shadow-md border p-6">
                    <h2 className="text-xl font-semibold mb-4">Latest Result</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                {recognitionMode.operation === 'translation' ? 'Translated Text' : 'Transcribed Text'}
                            </label>
                            <div className="p-4 bg-gray-50 rounded-lg border">
                                <p className="text-lg">{currentResult.text}</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Confidence
                                </label>
                                <p className="text-lg font-semibold">
                                    {(currentResult.confidence * 100).toFixed(1)}%
                                </p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Language
                                </label>
                                <p className="text-lg font-semibold">{currentResult.language}</p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Timestamp
                                </label>
                                <p className="text-sm text-gray-600">
                                    {new Date(currentResult.timestamp).toLocaleTimeString()}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Performance Metrics */}
            {showMetrics && (
                <PerformanceMetricsComponent
                    timingMetrics={getCurrentTimingMetrics()}
                    performanceMetrics={performanceMetrics}
                    recentResults={results}
                />
            )}

            {/* Results History */}
            {results.length > 0 && (
                <div className="bg-white rounded-lg shadow-md border p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold">Results History</h2>
                        <span className="text-sm text-gray-500">
                            {results.length} result{results.length !== 1 ? 's' : ''}
                        </span>
                    </div>

                    <div className="space-y-4 max-h-96 overflow-y-auto">
                        {results.map((result, index) => (
                            <div key={index} className="p-4 border rounded-lg hover:bg-gray-50">
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <p className="text-lg mb-2">{result.text}</p>
                                        <div className="flex items-center space-x-4 text-sm text-gray-600">
                                            <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                                            <span>Language: {result.language}</span>
                                            <span>
                                                {new Date(result.timestamp).toLocaleTimeString()}
                                            </span>
                                        </div>
                                    </div>
                                    {result.timing_metrics && (
                                        <div className="text-right text-sm text-gray-500">
                                            <div>Total: {(result.timing_metrics.total_time || 0).toFixed(0)}ms</div>
                                            <div>API: {(result.timing_metrics.api_call || 0).toFixed(0)}ms</div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}; 