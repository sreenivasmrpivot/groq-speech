#!/usr/bin/env python3
"""
Web Demo - Modern web interface for speech recognition
Demonstrates a web-based speech recognition application using Flask.
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


class WebSpeechDemo:
    """Web-based speech recognition demo."""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "groq-speech-demo-secret"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.speech_config = SpeechConfig()
        self.recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.translator = SpeechRecognizer(
            speech_config=self.speech_config
        )  # Separate instance for translation
        self.active_sessions = {}

        self.setup_routes()
        self.setup_socket_events()

    def setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main page."""
            return render_template("index.html")

        @self.app.route("/api/health")
        def health():
            """Health check endpoint."""
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                }
            )

        @self.app.route("/api/recognize", methods=["POST"])
        def recognize_audio():
            """REST API endpoint for audio recognition."""
            try:
                # This would handle audio data from POST request
                # For demo purposes, we'll simulate recognition
                return jsonify(
                    {
                        "success": True,
                        "text": "Demo recognition result",
                        "confidence": 0.95,
                        "language": "en-US",
                    }
                )
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

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
                "is_recognizing": False,
                "is_translating": False,
            }
            self.socketio.emit("connected", {"session_id": session_id}, room=session_id)
            print(f"Client connected: {session_id}")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            session_id = request.sid
            if session_id in self.active_sessions:
                # Stop any ongoing recognition/translation
                session_data = self.active_sessions[session_id]
                if session_data.get("is_recognizing") or session_data.get(
                    "is_translating"
                ):
                    self._stop_session_processing(session_id)
                del self.active_sessions[session_id]
            print(f"Client disconnected: {session_id}")

        @self.socketio.on("start_recognition")
        def handle_start_recognition(data):
            """Start speech recognition."""
            session_id = request.sid
            mode = data.get("mode", "once")  # "once" or "continuous"

            if session_id not in self.active_sessions:
                self.socketio.emit(
                    "recognition_error", {"error": "Session not found"}, room=session_id
                )
                return

            session_data = self.active_sessions[session_id]
            if session_data.get("is_recognizing") or session_data.get("is_translating"):
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
                                transcript = {
                                    "text": result.text,
                                    "confidence": result.confidence,
                                    "language": result.language,
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "recognition",
                                }

                                if session_id in self.active_sessions:
                                    self.active_sessions[session_id][
                                        "transcripts"
                                    ].append(transcript)

                                print(
                                    f"Emitting continuous recognition_result: {transcript}"
                                )
                                # Use socketio.emit to work outside request context
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
                        result = self.recognizer.recognize_once_async()
                        print(f"Recognition result: {result}")

                        if result.reason == ResultReason.RecognizedSpeech:
                            transcript = {
                                "text": result.text,
                                "confidence": result.confidence,
                                "language": result.language,
                                "timestamp": datetime.now().isoformat(),
                                "type": "recognition",
                            }

                            if session_id in self.active_sessions:
                                self.active_sessions[session_id]["transcripts"].append(
                                    transcript
                                )

                            print(f"Emitting recognition_result: {transcript}")
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

        @self.socketio.on("start_translation")
        def handle_start_translation(data):
            """Start speech translation."""
            session_id = request.sid
            mode = data.get("mode", "once")  # "once" or "continuous"

            if session_id not in self.active_sessions:
                self.socketio.emit(
                    "translation_error", {"error": "Session not found"}, room=session_id
                )
                return

            session_data = self.active_sessions[session_id]
            if session_data.get("is_recognizing") or session_data.get("is_translating"):
                self.socketio.emit(
                    "translation_error",
                    {"error": "Already processing"},
                    room=session_id,
                )
                return

            session_data["is_translating"] = True
            session_data["mode"] = mode

            self.socketio.emit(
                "translation_started",
                {"status": "translating", "mode": mode},
                room=session_id,
            )

            # Start translation in background
            def translate():
                try:
                    print(
                        f"Starting translation for session {session_id}, mode: {mode}"
                    )

                    if mode == "continuous":
                        # Continuous translation
                        def on_translation_recognized(result):
                            if session_id not in self.active_sessions:
                                return

                            if result.reason == ResultReason.RecognizedSpeech:
                                transcript = {
                                    "text": result.text,
                                    "confidence": result.confidence,
                                    "language": result.language,
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "translation",
                                }

                                if session_id in self.active_sessions:
                                    self.active_sessions[session_id][
                                        "transcripts"
                                    ].append(transcript)

                                print(
                                    f"Emitting continuous translation_result: {transcript}"
                                )
                                # Use socketio.emit to work outside request context
                                self.socketio.emit(
                                    "translation_result", transcript, room=session_id
                                )

                        # Connect event handlers
                        self.translator.connect("recognized", on_translation_recognized)

                        # Start continuous recognition
                        self.translator.start_continuous_recognition()

                        # Keep running until stopped
                        while (
                            session_id in self.active_sessions
                            and self.active_sessions[session_id].get("is_translating")
                        ):
                            time.sleep(0.1)

                        # Stop continuous recognition
                        self.translator.stop_continuous_recognition()
                    else:
                        # Single translation
                        result = self.translator.recognize_once_async()
                        print(f"Translation result: {result}")

                        if result.reason == ResultReason.RecognizedSpeech:
                            transcript = {
                                "text": result.text,
                                "confidence": result.confidence,
                                "language": result.language,
                                "timestamp": datetime.now().isoformat(),
                                "type": "translation",
                            }

                            if session_id in self.active_sessions:
                                self.active_sessions[session_id]["transcripts"].append(
                                    transcript
                                )

                            print(f"Emitting translation_result: {transcript}")
                            self.socketio.emit(
                                "translation_result", transcript, room=session_id
                            )
                        else:
                            print(
                                f"No speech detected for translation: {result.reason}"
                            )
                            self.socketio.emit(
                                "translation_error",
                                {"error": "No speech detected"},
                                room=session_id,
                            )

                except Exception as e:
                    print(f"Translation error: {e}")
                    self.socketio.emit(
                        "translation_error", {"error": str(e)}, room=session_id
                    )
                finally:
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["is_translating"] = False

            thread = threading.Thread(target=translate)
            thread.daemon = True
            thread.start()

        @self.socketio.on("stop_translation")
        def handle_stop_translation():
            """Stop speech translation."""
            session_id = request.sid
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["is_translating"] = False
            self.socketio.emit(
                "translation_stopped", {"status": "stopped"}, room=session_id
            )

    def _stop_session_processing(self, session_id):
        """Stop any ongoing recognition or translation for a session."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            session_data["is_recognizing"] = False
            session_data["is_translating"] = False

    def create_templates(self):
        """Create HTML templates."""
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Create index.html
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groq Speech Web Demo</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .controls {
            text-align: center;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1rem;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn.recording {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            animation: pulse 1.5s infinite;
        }
        
        .mode-selector {
            margin-bottom: 15px;
            text-align: center;
        }
        
        .mode-selector label {
            color: #333;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .mode-selector select {
            padding: 8px 12px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            color: #333;
            font-size: 0.9rem;
            cursor: pointer;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .status.ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.listening {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .transcript-area {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .transcript-entry {
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .transcript-text {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .transcript-meta {
            font-size: 0.8rem;
            color: #666;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }
        
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 30px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Groq Speech Web Demo</h1>
            <p>Real-time speech recognition powered by Groq AI</p>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>🎙️ Speech Recognition</h2>
                <div class="controls">
                    <div class="mode-selector">
                        <label>Mode:</label>
                        <select id="recognitionMode">
                            <option value="once">Single Recognition</option>
                            <option value="continuous">Continuous Recognition</option>
                        </select>
                    </div>
                    <button id="recordBtn" class="btn">🎤 Start Transcription</button>
                    <button id="translateBtn" class="btn">🌐 Start Translation</button>
                    <button id="clearBtn" class="btn">🗑️ Clear</button>
                </div>
                <div id="status" class="status ready">Ready to record</div>
            </div>
            
            <div class="card">
                <h2>📊 Statistics</h2>
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
                        <div id="sessionTime" class="stat-value">0s</div>
                        <div class="stat-label">Session Time</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 id="transcriptTitle">📝 Live Transcript</h2>
            <div id="transcriptArea" class="transcript-area">
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    Start recording to see transcriptions here...
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Built with Groq Speech SDK • Real-time AI-powered transcription</p>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // DOM elements
        const recordBtn = document.getElementById('recordBtn');
        const translateBtn = document.getElementById('translateBtn');
        const clearBtn = document.getElementById('clearBtn');
        const recognitionMode = document.getElementById('recognitionMode');
        const transcriptTitle = document.getElementById('transcriptTitle');
        const status = document.getElementById('status');
        const transcriptArea = document.getElementById('transcriptArea');
        const wordCount = document.getElementById('wordCount');
        const segmentCount = document.getElementById('segmentCount');
        const avgConfidence = document.getElementById('avgConfidence');
        const sessionTime = document.getElementById('sessionTime');
        
        // State
        let isRecording = false;
        let isTranslating = false;
        let sessionStartTime = null;
        let transcripts = [];
        
        // Socket event handlers
        socket.on('connected', (data) => {
            console.log('Connected to server:', data.session_id);
            updateStatus('Connected to server', 'ready');
        });
        
        socket.on('recognition_started', (data) => {
            updateStatus('Listening... Speak now!', 'listening');
        });
        
        socket.on('recognition_result', (data) => {
            console.log('Received recognition_result:', data);
            addTranscript(data);
            updateStats();
        });
        
        socket.on('recognition_stopped', (data) => {
            updateStatus('Recording stopped', 'ready');
        });
        
        socket.on('recognition_error', (data) => {
            updateStatus('Error: ' + data.error, 'error');
        });
        
        socket.on('translation_started', (data) => {
            updateStatus('Translating... Speak now!', 'listening');
        });
        
        socket.on('translation_result', (data) => {
            console.log('Received translation_result:', data);
            addTranscript(data);
            updateStats();
        });
        
        socket.on('translation_stopped', (data) => {
            updateStatus('Translation stopped', 'ready');
        });
        
        socket.on('translation_error', (data) => {
            updateStatus('Translation Error: ' + data.error, 'error');
        });
        
        // Button event handlers
        recordBtn.addEventListener('click', () => {
            if (!isRecording && !isTranslating) {
                startRecording();
            } else if (isRecording) {
                stopRecording();
            }
        });
        
        translateBtn.addEventListener('click', () => {
            if (!isTranslating && !isRecording) {
                startTranslation();
            } else if (isTranslating) {
                stopTranslation();
            }
        });
        
        clearBtn.addEventListener('click', () => {
            clearTranscripts();
        });
        
        // Functions
        function startRecording() {
            isRecording = true;
            recordBtn.textContent = '⏹️ Stop Transcription';
            recordBtn.classList.add('recording');
            sessionStartTime = Date.now();
            const mode = recognitionMode.value;
            transcriptTitle.textContent = '🎤 Live Transcript';
            socket.emit('start_recognition', { mode: mode });
        }
        
        function stopRecording() {
            isRecording = false;
            recordBtn.textContent = '🎤 Start Transcription';
            recordBtn.classList.remove('recording');
            transcriptTitle.textContent = '📝 Live Transcript';
            socket.emit('stop_recognition');
        }
        
        function startTranslation() {
            isTranslating = true;
            translateBtn.textContent = '⏹️ Stop Translation';
            translateBtn.classList.add('recording');
            sessionStartTime = Date.now();
            const mode = recognitionMode.value;
            transcriptTitle.textContent = '🌐 Live Translation';
            socket.emit('start_translation', { mode: mode });
        }
        
        function stopTranslation() {
            isTranslating = false;
            translateBtn.textContent = '🌐 Start Translation';
            translateBtn.classList.remove('recording');
            transcriptTitle.textContent = '📝 Live Transcript';
            socket.emit('stop_translation');
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
            const isTranslation = data.type === 'translation';
            
            entry.innerHTML = `
                <div class="transcript-text">
                    ${isTranslation ? '🌐 ' : '🎤 '}${data.text}
                </div>
                <div class="transcript-meta">
                    ${isTranslation ? 'Translation' : 'Recognition'} • 
                    Confidence: ${confidencePercent}% • 
                    Language: ${data.language || 'Unknown'} • 
                    Time: ${timestamp}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
            `;
            
            transcriptArea.appendChild(entry);
            transcriptArea.scrollTop = transcriptArea.scrollHeight;
            
            transcripts.push(data);
        }
        
        function clearTranscripts() {
            transcriptArea.innerHTML = `
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    Start recording to see transcriptions here...
                </div>
            `;
            transcripts = [];
            updateStats();
        }
        
        function updateStats() {
            const totalWords = transcripts.reduce((sum, t) => sum + t.text.split(' ').length, 0);
            const avgConf = transcripts.length > 0 
                ? transcripts.reduce((sum, t) => sum + t.confidence, 0) / transcripts.length 
                : 0;
            
            wordCount.textContent = totalWords;
            segmentCount.textContent = transcripts.length;
            avgConfidence.textContent = Math.round(avgConf * 100) + '%';
            
            if (sessionStartTime) {
                const elapsed = Math.round((Date.now() - sessionStartTime) / 1000);
                sessionTime.textContent = elapsed + 's';
            }
        }
        
        // Update session time every second
        setInterval(() => {
            if (sessionStartTime) {
                const elapsed = Math.round((Date.now() - sessionStartTime) / 1000);
                sessionTime.textContent = elapsed + 's';
            }
        }, 1000);
    </script>
</body>
</html>"""

        with open(os.path.join(templates_dir, "index.html"), "w") as f:
            f.write(index_html)

    def run(self, host="localhost", port=5000, debug=False):
        """Run the web demo."""
        print(f"🌐 Starting Web Demo on http://{host}:{port}")
        print("Opening browser automatically...")

        # Create templates
        self.create_templates()

        # Open browser
        webbrowser.open(f"http://{host}:{port}")

        # Run Flask app
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def main():
    """Main function."""
    print("🌐 Starting Groq Speech Web Demo...")
    print("This demo showcases a modern web interface for speech recognition.")
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
        demo = WebSpeechDemo()
        demo.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
