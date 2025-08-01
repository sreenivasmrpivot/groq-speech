#!/usr/bin/env python3
"""
Transcription Workbench Demo - Professional transcription tool
Demonstrates advanced transcription features with file handling, export, and analysis.
"""

import os
import sys
import threading
import json
from datetime import datetime
from typing import List, Dict, Any
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import webbrowser

# Add the current directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import (
    SpeechConfig,
    SpeechRecognizer,
    AudioConfig,
    ResultReason,
    Config,
)


class TranscriptionWorkbench:
    """Professional transcription workbench with advanced features."""

    def __init__(self):
        self.speech_config = SpeechConfig()
        self.recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.is_recording = False
        self.transcripts = []
        self.current_session = {
            "start_time": None,
            "end_time": None,
            "segments": [],
            "total_duration": 0,
            "word_count": 0,
        }
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Transcription Workbench")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Configure style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Title.TLabel", font=("Arial", 16, "bold"), foreground="#ecf0f1"
        )
        style.configure("Status.TLabel", font=("Arial", 12), foreground="#bdc3c7")

        self.create_widgets()
        self.setup_events()

    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            main_frame, text="📝 Transcription Workbench", style="Title.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Control panel
        control_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        # Status and controls
        status_frame = tk.Frame(control_frame, bg="#34495e")
        status_frame.pack(fill=tk.X, padx=20, pady=10)

        self.status_label = ttk.Label(
            status_frame, text="Ready to transcribe...", style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT)

        # Control buttons
        button_frame = tk.Frame(control_frame, bg="#34495e")
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.record_button = tk.Button(
            button_frame,
            text="🎤 Start Recording",
            command=self.toggle_recording,
            bg="#3498db",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))

        self.file_button = tk.Button(
            button_frame,
            text="📁 Load Audio File",
            command=self.load_audio_file,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.file_button.pack(side=tk.LEFT, padx=(0, 10))

        self.export_button = tk.Button(
            button_frame,
            text="💾 Export Transcript",
            command=self.export_transcript,
            bg="#f39c12",
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_all,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.clear_button.pack(side=tk.LEFT)

        # Main content area
        content_frame = tk.Frame(main_frame, bg="#2c3e50")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Live transcription
        left_frame = tk.Frame(content_frame, bg="#34495e", relief=tk.SUNKEN, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        live_label = ttk.Label(
            left_frame, text="Live Transcription", style="Status.TLabel"
        )
        live_label.pack(pady=(10, 5))

        self.live_text = scrolledtext.ScrolledText(
            left_frame,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 10),
            wrap=tk.WORD,
            height=20,
        )
        self.live_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Right panel - Session info and analysis
        right_frame = tk.Frame(content_frame, bg="#34495e", relief=tk.SUNKEN, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Session info
        session_frame = tk.Frame(right_frame, bg="#34495e")
        session_frame.pack(fill=tk.X, padx=10, pady=10)

        session_label = ttk.Label(
            session_frame, text="Session Info", style="Status.TLabel"
        )
        session_label.pack(anchor=tk.W)

        self.session_info = tk.Text(
            session_frame,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 9),
            height=8,
            wrap=tk.WORD,
        )
        self.session_info.pack(fill=tk.X, pady=(5, 0))

        # Analysis frame
        analysis_frame = tk.Frame(right_frame, bg="#34495e")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        analysis_label = ttk.Label(
            analysis_frame, text="Analysis", style="Status.TLabel"
        )
        analysis_label.pack(anchor=tk.W)

        self.analysis_text = tk.Text(
            analysis_frame,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def setup_events(self):
        """Setup event handlers."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Bind keyboard shortcuts
        self.root.bind("<Control-r>", lambda e: self.toggle_recording())
        self.root.bind("<Control-o>", lambda e: self.load_audio_file())
        self.root.bind("<Control-s>", lambda e: self.export_transcript())
        self.root.bind("<Control-c>", lambda e: self.clear_all())

    def toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording for transcription."""
        self.is_recording = True
        self.record_button.config(text="⏹️ Stop Recording", bg="#e74c3c")
        self.status_label.config(text="Recording... Speak now!")

        # Initialize session
        self.current_session = {
            "start_time": datetime.now(),
            "end_time": None,
            "segments": [],
            "total_duration": 0,
            "word_count": 0,
        }

        # Start recognition in a separate thread
        self.recognition_thread = threading.Thread(target=self._record_transcription)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()

        self.update_session_info()

    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording", bg="#3498db")
        self.status_label.config(text="Recording stopped.")

        # Finalize session
        self.current_session["end_time"] = datetime.now()
        if self.current_session["start_time"]:
            duration = (
                self.current_session["end_time"] - self.current_session["start_time"]
            ).total_seconds()
            self.current_session["total_duration"] = duration

        self.update_session_info()
        self.update_analysis()

    def _record_transcription(self):
        """Record transcription in background."""
        try:
            result = self.recognizer.recognize_once_async()
            self.root.after(0, self._handle_transcription_result, result)
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _handle_transcription_result(self, result):
        """Handle transcription result."""
        if result.reason == ResultReason.RecognizedSpeech:
            text = result.text.strip()
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Add to live transcription
            self.live_text.insert(tk.END, f"[{timestamp}] {text}\n")
            self.live_text.see(tk.END)

            # Add to session
            segment = {
                "timestamp": timestamp,
                "text": text,
                "confidence": result.confidence,
                "language": result.language,
            }
            self.current_session["segments"].append(segment)
            self.current_session["word_count"] += len(text.split())

            # Continue recording if still active
            if self.is_recording:
                self.recognition_thread = threading.Thread(
                    target=self._record_transcription
                )
                self.recognition_thread.daemon = True
                self.recognition_thread.start()

            self.update_session_info()

        elif result.reason == ResultReason.NoMatch:
            if self.is_recording:
                # Continue recording even if no match
                self.recognition_thread = threading.Thread(
                    target=self._record_transcription
                )
                self.recognition_thread.daemon = True
                self.recognition_thread.start()

        elif result.reason == ResultReason.Canceled:
            self.stop_recording()

    def _handle_error(self, error_msg):
        """Handle recognition error."""
        self.stop_recording()
        messagebox.showerror("Error", f"Recognition error: {error_msg}")

    def load_audio_file(self):
        """Load and transcribe an audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                self.status_label.config(
                    text=f"Processing {os.path.basename(file_path)}..."
                )

                # Create audio config for file
                audio_config = AudioConfig(filename=file_path)
                file_recognizer = SpeechRecognizer(
                    speech_config=self.speech_config, audio_config=audio_config
                )

                # Process file
                result = file_recognizer.recognize_once_async()

                if result.reason == ResultReason.RecognizedSpeech:
                    text = result.text.strip()
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    self.live_text.insert(tk.END, f"[{timestamp}] File: {text}\n")
                    self.live_text.see(tk.END)

                    # Add to session
                    segment = {
                        "timestamp": timestamp,
                        "text": text,
                        "confidence": result.confidence,
                        "language": result.language,
                        "source": "file",
                    }
                    self.current_session["segments"].append(segment)
                    self.current_session["word_count"] += len(text.split())

                    self.status_label.config(text="File processed successfully.")
                    self.update_session_info()
                    self.update_analysis()

                else:
                    messagebox.showwarning("Warning", "No speech detected in file.")

            except Exception as e:
                messagebox.showerror("Error", f"Error processing file: {e}")
                self.status_label.config(text="File processing failed.")

    def export_transcript(self):
        """Export transcript to file."""
        if not self.current_session["segments"]:
            messagebox.showwarning("Warning", "No transcript to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Transcript",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                if file_path.endswith(".json"):
                    self._export_json(file_path)
                else:
                    self._export_text(file_path)

                messagebox.showinfo("Success", f"Transcript exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    def _export_text(self, file_path: str):
        """Export as plain text."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Transcription Workbench Export\n")
            f.write("=" * 30 + "\n\n")

            if self.current_session["start_time"]:
                f.write(f"Session: {self.current_session['start_time']}\n")
                if self.current_session["end_time"]:
                    f.write(
                        f"Duration: {self.current_session['total_duration']:.1f}s\n"
                    )
                f.write(f"Word count: {self.current_session['word_count']}\n\n")

            for segment in self.current_session["segments"]:
                f.write(f"[{segment['timestamp']}] {segment['text']}\n")

    def _export_json(self, file_path: str):
        """Export as JSON."""
        export_data = {
            "session": self.current_session,
            "export_time": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

    def clear_all(self):
        """Clear all transcriptions and session data."""
        if messagebox.askyesno("Confirm", "Clear all transcriptions?"):
            self.live_text.delete(1.0, tk.END)
            self.analysis_text.delete(1.0, tk.END)
            self.current_session = {
                "start_time": None,
                "end_time": None,
                "segments": [],
                "total_duration": 0,
                "word_count": 0,
            }
            self.update_session_info()
            self.status_label.config(text="All data cleared.")

    def update_session_info(self):
        """Update session information display."""
        self.session_info.delete(1.0, tk.END)

        info = []
        info.append(f"Segments: {len(self.current_session['segments'])}")
        info.append(f"Word count: {self.current_session['word_count']}")

        if self.current_session["start_time"]:
            info.append(
                f"Start: {self.current_session['start_time'].strftime('%H:%M:%S')}"
            )

            if self.current_session["end_time"]:
                info.append(
                    f"End: {self.current_session['end_time'].strftime('%H:%M:%S')}"
                )
                info.append(f"Duration: {self.current_session['total_duration']:.1f}s")

        if self.current_session["segments"]:
            avg_confidence = sum(
                s.get("confidence", 0) for s in self.current_session["segments"]
            ) / len(self.current_session["segments"])
            info.append(f"Avg confidence: {avg_confidence:.2f}")

        self.session_info.insert(tk.END, "\n".join(info))

    def update_analysis(self):
        """Update analysis display."""
        self.analysis_text.delete(1.0, tk.END)

        if not self.current_session["segments"]:
            return

        analysis = []
        analysis.append("TRANSCRIPTION ANALYSIS")
        analysis.append("=" * 25)
        analysis.append("")

        # Basic stats
        total_words = self.current_session["word_count"]
        total_segments = len(self.current_session["segments"])
        avg_words_per_segment = (
            total_words / total_segments if total_segments > 0 else 0
        )

        analysis.append(f"Total segments: {total_segments}")
        analysis.append(f"Total words: {total_words}")
        analysis.append(f"Avg words/segment: {avg_words_per_segment:.1f}")

        if self.current_session["total_duration"] > 0:
            words_per_minute = (
                total_words / self.current_session["total_duration"]
            ) * 60
            analysis.append(f"Words per minute: {words_per_minute:.1f}")

        # Confidence analysis
        confidences = [s.get("confidence", 0) for s in self.current_session["segments"]]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)

            analysis.append("")
            analysis.append("CONFIDENCE ANALYSIS")
            analysis.append(f"Average: {avg_conf:.2f}")
            analysis.append(f"Range: {min_conf:.2f} - {max_conf:.2f}")

        # Language analysis
        languages = [
            s.get("language", "unknown") for s in self.current_session["segments"]
        ]
        if languages:
            lang_counts = {}
            for lang in languages:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            analysis.append("")
            analysis.append("LANGUAGE DETECTION")
            for lang, count in lang_counts.items():
                percentage = (count / len(languages)) * 100
                analysis.append(f"{lang}: {count} ({percentage:.1f}%)")

        self.analysis_text.insert(tk.END, "\n".join(analysis))

    def on_closing(self):
        """Handle window closing."""
        if self.is_recording:
            self.stop_recording()
        self.root.quit()

    def run(self):
        """Run the transcription workbench."""
        self.root.mainloop()


def main():
    """Main function."""
    print("📝 Starting Transcription Workbench Demo...")
    print("This demo showcases professional transcription features.")
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
        workbench = TranscriptionWorkbench()
        workbench.run()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
