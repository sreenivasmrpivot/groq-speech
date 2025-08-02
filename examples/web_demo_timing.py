#!/usr/bin/env python3
"""
Enhanced Web Demo with Timing Metrics
Demonstrates speech recognition with detailed performance tracking.
"""

import os
import sys
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import webbrowser

# Add the current directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, Config


class TimingWebDemo:
    """Web-based speech recognition demo with timing metrics."""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "groq-speech-timing-demo"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.speech_config = SpeechConfig()
        self.recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.active_sessions = {}

        self.setup_routes()
        self.setup_socket_events()

    def setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main page."""
            return render_template("timing_index.html")

        @self.app.route("/api/performance")
        def performance():
            """Performance metrics endpoint."""
            return jsonify(self.get_performance_metrics())

    def setup_socket_events(self):
        """Setup WebSocket events."""

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            session_id = request.sid
            self.active_sessions[session_id] = {
                "connected": True,
                "start_time": datetime.now(),
                "transcripts": [],
                "timing_metrics": [],
                "is_recognizing": False,
            }
            self.socketio.emit("connected", {"session_id": session_id}, room=session_id)
            print(f"Client connected: {session_id}")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            session_id = request.sid
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            print(f"Client disconnected: {session_id}")

        @self.socketio.on("start_recognition")
        def handle_start_recognition(data):
            """Start speech recognition."""
            session_id = request.sid
            mode = data.get("mode", "once")

            if session_id not in self.active_sessions:
                self.socketio.emit(
                    "recognition_error", {"error": "Session not found"}, room=session_id
                )
                return

            session_data = self.active_sessions[session_id]
            if session_data.get("is_recognizing"):
                self.socketio.emit(
                    "recognition_error",
                    {"error": "Already processing"},
                    room=session_id,
                )
                return

            session_data["is_recognizing"] = True
            session_data["mode"] = mode

            self.socketio.emit(
                "recognition_started",
                {"status": "listening", "mode": mode},
                room=session_id,
            )

            # Start recognition in background
            def recognize():
                try:
                    print(
                        f"Starting recognition for session {session_id}, mode: {mode}"
                    )

                    if mode == "continuous":
                        # Continuous recognition
                        def on_recognized(result):
                            if session_id not in self.active_sessions:
                                return

                            if result.reason == ResultReason.RecognizedSpeech:
                                # Extract timing metrics
                                timing_data = {}
                                if result.timing_metrics:
                                    timing_data = result.timing_metrics.get_metrics()

                                transcript = {
                                    "text": result.text,
                                    "confidence": result.confidence,
                                    "language": result.language,
                                    "timestamp": datetime.now().isoformat(),
                                    "timing_metrics": timing_data,
                                    "segment_id": len(session_data["transcripts"]) + 1,
                                }

                                if session_id in self.active_sessions:
                                    self.active_sessions[session_id][
                                        "transcripts"
                                    ].append(transcript)
                                    self.active_sessions[session_id][
                                        "timing_metrics"
                                    ].append(timing_data)

                                print(f"Emitting result: {transcript}")
                                self.socketio.emit(
                                    "recognition_result", transcript, room=session_id
                                )

                        # Connect event handlers
                        self.recognizer.connect("recognized", on_recognized)

                        # Start continuous recognition
                        self.recognizer.start_continuous_recognition()

                        # Keep running until stopped
                        while (
                            session_id in self.active_sessions
                            and self.active_sessions[session_id].get("is_recognizing")
                        ):
                            time.sleep(0.1)

                        # Stop continuous recognition
                        self.recognizer.stop_continuous_recognition()
                    else:
                        # Single recognition
                        print("Starting single recognition...")
                        result = self.recognizer.recognize_once_async()
                        print(f"Recognition result: {result}")

                        if result.reason == ResultReason.RecognizedSpeech:
                            # Extract timing metrics
                            timing_data = {}
                            if result.timing_metrics:
                                timing_data = result.timing_metrics.get_metrics()

                            transcript = {
                                "text": result.text,
                                "confidence": result.confidence,
                                "language": result.language,
                                "timestamp": datetime.now().isoformat(),
                                "timing_metrics": timing_data,
                                "segment_id": 1,
                            }

                            if session_id in self.active_sessions:
                                self.active_sessions[session_id]["transcripts"].append(
                                    transcript
                                )
                                self.active_sessions[session_id][
                                    "timing_metrics"
                                ].append(timing_data)

                            print(f"Emitting result: {transcript}")
                            self.socketio.emit(
                                "recognition_result", transcript, room=session_id
                            )
                        else:
                            print(f"No speech detected: {result.reason}")
                            self.socketio.emit(
                                "recognition_error",
                                {"error": "No speech detected"},
                                room=session_id,
                            )

                except Exception as e:
                    print(f"Recognition error: {e}")
                    self.socketio.emit(
                        "recognition_error", {"error": str(e)}, room=session_id
                    )
                finally:
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["is_recognizing"] = False

            thread = threading.Thread(target=recognize)
            thread.daemon = True
            thread.start()

        @self.socketio.on("stop_recognition")
        def handle_stop_recognition():
            """Stop speech recognition."""
            session_id = request.sid
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["is_recognizing"] = False
            self.socketio.emit(
                "recognition_stopped", {"status": "stopped"}, room=session_id
            )

    def get_performance_metrics(self):
        """Get performance metrics for all sessions."""
        metrics = {
            "total_sessions": len(self.active_sessions),
            "total_transcripts": 0,
            "avg_confidence": 0.0,
            "timing_breakdown": {
                "microphone_capture": [],
                "api_call": [],
                "response_processing": [],
                "total_time": [],
            },
        }

        total_confidence = 0.0
        confidence_count = 0

        for session_data in self.active_sessions.values():
            metrics["total_transcripts"] += len(session_data["transcripts"])

            for transcript in session_data["transcripts"]:
                if "confidence" in transcript:
                    total_confidence += transcript["confidence"]
                    confidence_count += 1

                if "timing_metrics" in transcript:
                    timing = transcript["timing_metrics"]
                    for key in metrics["timing_breakdown"]:
                        if key in timing:
                            metrics["timing_breakdown"][key].append(timing[key])

        if confidence_count > 0:
            metrics["avg_confidence"] = total_confidence / confidence_count

        # Calculate averages for timing metrics
        for key in metrics["timing_breakdown"]:
            values = metrics["timing_breakdown"][key]
            if values:
                metrics["timing_breakdown"][key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return metrics

    def create_templates(self):
        """Create HTML templates."""
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Create timing index.html
        timing_index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groq Speech - Timing Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: white; font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .card h2 { color: #333; margin-bottom: 20px; font-size: 1.5rem; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .controls { text-align: center; }
        .btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 1.1rem; cursor: pointer; margin: 10px; transition: all 0.3s ease; }
        .btn:hover { transform: translateY(-2px); }
        .btn.recording { background: linear-gradient(45deg, #e74c3c, #c0392b); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        .status { margin: 20px 0; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; }
        .status.ready { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.listening { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .transcript-area { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; height: 400px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 0.9rem; }
        .transcript-entry { margin-bottom: 15px; padding: 10px; background: white; border-radius: 8px; border-left: 4px solid #667eea; }
        .transcript-text { font-weight: bold; margin-bottom: 5px; }
        .transcript-meta { font-size: 0.8rem; color: #666; }
        .timing-metrics { background: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.8rem; }
        .timing-bar { width: 100%; height: 6px; background: #e9ecef; border-radius: 3px; overflow: hidden; margin: 5px 0; }
        .timing-fill { height: 100%; transition: width 0.3s ease; }
        .timing-fill.microphone { background: linear-gradient(90deg, #4caf50, #8bc34a); }
        .timing-fill.api { background: linear-gradient(90deg, #2196f3, #03a9f4); }
        .timing-fill.processing { background: linear-gradient(90deg, #ff9800, #ffc107); }
        .charts-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .chart-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .chart-title { font-size: 1.1rem; font-weight: bold; margin-bottom: 15px; color: #333; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
        .stat-item { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px; border: 1px solid #dee2e6; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 0.9rem; color: #666; margin-top: 5px; }
        .footer { text-align: center; color: rgba(255,255,255,0.8); margin-top: 30px; }
        @media (max-width: 1200px) { .main-content { grid-template-columns: 1fr; } .charts-container { grid-template-columns: 1fr; } .header h1 { font-size: 2rem; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Groq Speech - Timing Analysis</h1>
            <p>Real-time speech recognition with detailed performance tracking</p>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>🎙️ Speech Recognition</h2>
                <div class="controls">
                    <select id="recognitionMode">
                        <option value="once">Single Recognition</option>
                        <option value="continuous">Continuous Recognition</option>
                    </select>
                    <button id="recordBtn" class="btn">🎤 Start Transcription</button>
                    <button id="clearBtn" class="btn">🗑️ Clear</button>
                </div>
                <div id="status" class="status ready">Ready to record</div>
            </div>
            
            <div class="card">
                <h2>📊 Performance Metrics</h2>
                <div class="stats">
                    <div class="stat-item">
                        <div id="wordCount" class="stat-value">0</div>
                        <div class="stat-label">Words</div>
                    </div>
                    <div class="stat-item">
                        <div id="segmentCount" class="stat-value">0</div>
                        <div class="stat-label">Segments</div>
                    </div>
                    <div class="stat-item">
                        <div id="avgConfidence" class="stat-value">0%</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat-item">
                        <div id="avgTotalTime" class="stat-value">0ms</div>
                        <div class="stat-label">Avg Total Time</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 id="transcriptTitle">📝 Live Transcript with Timing</h2>
            <div id="transcriptArea" class="transcript-area">
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    Start recording to see transcriptions with timing metrics...
                </div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-card">
                <div class="chart-title">⏱️ Timing Breakdown</div>
                <canvas id="timingChart" width="400" height="300"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-title">📈 Performance Trends</div>
                <canvas id="performanceChart" width="400" height="300"></canvas>
            </div>
        </div>
        
        <div class="footer">
            <p>Built with Groq Speech SDK • Real-time timing analysis</p>
        </div>
    </div>

    <script>
        const socket = io();
        const recordBtn = document.getElementById('recordBtn');
        const clearBtn = document.getElementById('clearBtn');
        const recognitionMode = document.getElementById('recognitionMode');
        const transcriptTitle = document.getElementById('transcriptTitle');
        const status = document.getElementById('status');
        const transcriptArea = document.getElementById('transcriptArea');
        const wordCount = document.getElementById('wordCount');
        const segmentCount = document.getElementById('segmentCount');
        const avgConfidence = document.getElementById('avgConfidence');
        const avgTotalTime = document.getElementById('avgTotalTime');
        
        let isRecording = false;
        let transcripts = [];
        let timingChart = null;
        let performanceChart = null;
        
        function initializeCharts() {
            const timingCtx = document.getElementById('timingChart').getContext('2d');
            timingChart = new Chart(timingCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Microphone Capture', 'API Call', 'Response Processing'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: ['#4caf50', '#2196f3', '#ff9800'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.parsed;
                                    return label + ': ' + value.toFixed(2) + 'ms';
                                }
                            }
                        }
                    }
                }
            });
            
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Total Time (ms)',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Confidence (%)',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Segment' } },
                        y: { title: { display: true, text: 'Time (ms)' } },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Confidence (%)' },
                            grid: { drawOnChartArea: false },
                        }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });
        }
        
        socket.on('connected', (data) => {
            console.log('Connected to server:', data.session_id);
            updateStatus('Connected to server', 'ready');
            initializeCharts();
        });
        
        socket.on('recognition_started', (data) => {
            updateStatus('Listening... Speak now!', 'listening');
        });
        
        socket.on('recognition_result', (data) => {
            console.log('Received recognition_result:', data);
            addTranscript(data);
            updateStats();
            updateCharts(data);
        });
        
        socket.on('recognition_stopped', (data) => {
            updateStatus('Recording stopped', 'ready');
        });
        
        socket.on('recognition_error', (data) => {
            updateStatus('Error: ' + data.error, 'error');
        });
        
        recordBtn.addEventListener('click', () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        });
        
        clearBtn.addEventListener('click', () => {
            clearTranscripts();
        });
        
        function startRecording() {
            isRecording = true;
            recordBtn.textContent = '⏹️ Stop Transcription';
            recordBtn.classList.add('recording');
            transcriptTitle.textContent = '🎤 Live Transcript with Timing';
            const mode = recognitionMode.value;
            socket.emit('start_recognition', { mode: mode });
        }
        
        function stopRecording() {
            isRecording = false;
            recordBtn.textContent = '🎤 Start Transcription';
            recordBtn.classList.remove('recording');
            transcriptTitle.textContent = '📝 Live Transcript with Timing';
            socket.emit('stop_recognition');
        }
        
        function updateStatus(message, type) {
            status.textContent = message;
            status.className = `status ${type}`;
        }
        
        function addTranscript(data) {
            const entry = document.createElement('div');
            entry.className = 'transcript-entry';
            
            const confidencePercent = Math.round(data.confidence * 100);
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            
            let timingHtml = '';
            if (data.timing_metrics) {
                const timing = data.timing_metrics;
                const totalTime = timing.total_time || 0;
                const micTime = timing.microphone_capture || 0;
                const apiTime = timing.api_call || 0;
                const procTime = timing.response_processing || 0;
                
                timingHtml = `
                    <div class="timing-metrics">
                        <strong>Timing Breakdown:</strong><br>
                        <div>Microphone: ${(micTime * 1000).toFixed(1)}ms</div>
                        <div class="timing-bar">
                            <div class="timing-fill microphone" style="width: ${(micTime / totalTime * 100)}%"></div>
                        </div>
                        <div>API Call: ${(apiTime * 1000).toFixed(1)}ms</div>
                        <div class="timing-bar">
                            <div class="timing-fill api" style="width: ${(apiTime / totalTime * 100)}%"></div>
                        </div>
                        <div>Processing: ${(procTime * 1000).toFixed(1)}ms</div>
                        <div class="timing-bar">
                            <div class="timing-fill processing" style="width: ${(procTime / totalTime * 100)}%"></div>
                        </div>
                        <div><strong>Total: ${(totalTime * 1000).toFixed(1)}ms</strong></div>
                    </div>
                `;
            }
            
            entry.innerHTML = `
                <div class="transcript-text">
                    🎤 ${data.text}
                </div>
                <div class="transcript-meta">
                    Recognition • 
                    Confidence: ${confidencePercent}% • 
                    Language: ${data.language || 'Unknown'} • 
                    Time: ${timestamp} •
                    Segment: ${data.segment_id || 1}
                </div>
                ${timingHtml}
            `;
            
            transcriptArea.appendChild(entry);
            transcriptArea.scrollTop = transcriptArea.scrollHeight;
            
            transcripts.push(data);
        }
        
        function clearTranscripts() {
            transcriptArea.innerHTML = `
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    Start recording to see transcriptions with timing metrics...
                </div>
            `;
            transcripts = [];
            updateStats();
            updateCharts();
        }
        
        function updateStats() {
            const totalWords = transcripts.reduce((sum, t) => sum + t.text.split(' ').length, 0);
            const avgConf = transcripts.length > 0 
                ? transcripts.reduce((sum, t) => sum + t.confidence, 0) / transcripts.length 
                : 0;
            
            let avgTotal = 0;
            if (transcripts.length > 0) {
                const totalTimes = transcripts
                    .filter(t => t.timing_metrics && t.timing_metrics.total_time)
                    .map(t => t.timing_metrics.total_time * 1000);
                avgTotal = totalTimes.length > 0 ? totalTimes.reduce((a, b) => a + b) / totalTimes.length : 0;
            }
            
            wordCount.textContent = totalWords;
            segmentCount.textContent = transcripts.length;
            avgConfidence.textContent = Math.round(avgConf * 100) + '%';
            avgTotalTime.textContent = Math.round(avgTotal) + 'ms';
        }
        
        function updateCharts(data = null) {
            if (data && data.timing_metrics) {
                const timing = data.timing_metrics;
                const micTime = (timing.microphone_capture || 0) * 1000;
                const apiTime = (timing.api_call || 0) * 1000;
                const procTime = (timing.response_processing || 0) * 1000;
                
                // Update timing breakdown chart
                timingChart.data.datasets[0].data = [micTime, apiTime, procTime];
                timingChart.update();
                
                // Update performance trends chart
                const segmentNum = transcripts.length;
                const totalTime = (timing.total_time || 0) * 1000;
                const confidence = data.confidence * 100;
                
                performanceChart.data.labels.push(`S${segmentNum}`);
                performanceChart.data.datasets[0].data.push(totalTime);
                performanceChart.data.datasets[1].data.push(confidence);
                
                // Keep only last 10 segments
                if (performanceChart.data.labels.length > 10) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                    performanceChart.data.datasets[1].data.shift();
                }
                
                performanceChart.update();
            }
        }
    </script>
</body>
</html>"""

        with open(os.path.join(templates_dir, "timing_index.html"), "w") as f:
            f.write(timing_index_html)

    def run(self, host="localhost", port=5000, debug=False):
        """Run the timing web demo."""
        print(f"🌐 Starting Timing Web Demo on http://{host}:{port}")
        print("Opening browser automatically...")

        # Create templates
        self.create_templates()

        # Open browser
        webbrowser.open(f"http://{host}:{port}")

        # Run Flask app
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def main():
    """Main function."""
    print("🌐 Starting Groq Speech Timing Demo...")
    print("This demo showcases timing metrics and performance analysis.")
    print("Make sure your GROQ_API_KEY is set in the .env file.")
    print()

    try:
        # Validate API key
        Config.get_api_key()
        print("✅ API key found and validated.")
    except ValueError as e:
        print(f"❌ API key error: {e}")
        print("Please set your GROQ_API_KEY in the .env file.")
        return

    try:
        demo = TimingWebDemo()
        demo.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
