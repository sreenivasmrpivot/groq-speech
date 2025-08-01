#!/usr/bin/env python3
"""
Voice Assistant Demo - Real-world application with intuitive UI
Demonstrates a practical voice assistant with visual feedback and command processing.
"""

import os
import sys
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
import webbrowser

# Add the current directory to the path to import the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq_speech import SpeechConfig, SpeechRecognizer, ResultReason, Config


class VoiceAssistant:
    """A practical voice assistant with GUI interface."""

    def __init__(self):
        self.speech_config = SpeechConfig()
        self.recognizer = SpeechRecognizer(speech_config=self.speech_config)
        self.is_listening = False
        self.commands = {
            "hello": self._handle_greeting,
            "time": self._handle_time,
            "date": self._handle_date,
            "weather": self._handle_weather,
            "search": self._handle_search,
            "open": self._handle_open,
            "help": self._handle_help,
            "quit": self._handle_quit,
            "stop": self._handle_stop,
        }
        self.conversation_history = []
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Voice Assistant Demo")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        # Configure style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Title.TLabel", font=("Arial", 16, "bold"), foreground="#ecf0f1"
        )
        style.configure("Status.TLabel", font=("Arial", 12), foreground="#bdc3c7")
        style.configure("Success.TLabel", font=("Arial", 10), foreground="#27ae60")
        style.configure("Error.TLabel", font=("Arial", 10), foreground="#e74c3c")

        self.create_widgets()
        self.setup_events()

    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            main_frame, text="🎤 Voice Assistant", style="Title.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Status frame
        status_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=(0, 20))

        self.status_label = ttk.Label(
            status_frame, text="Ready to listen...", style="Status.TLabel"
        )
        self.status_label.pack(pady=10)

        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg="#2c3e50")
        button_frame.pack(fill=tk.X, pady=(0, 20))

        self.listen_button = tk.Button(
            button_frame,
            text="🎤 Start Listening",
            command=self.toggle_listening,
            bg="#3498db",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.listen_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear History",
            command=self.clear_history,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))

        self.help_button = tk.Button(
            button_frame,
            text="❓ Help",
            command=self.show_help,
            bg="#f39c12",
            fg="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.help_button.pack(side=tk.LEFT)

        # Conversation display
        conversation_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.SUNKEN, bd=2)
        conversation_frame.pack(fill=tk.BOTH, expand=True)

        conversation_label = ttk.Label(
            conversation_frame, text="Conversation History", style="Status.TLabel"
        )
        conversation_label.pack(pady=(10, 5))

        self.conversation_text = scrolledtext.ScrolledText(
            conversation_frame,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Command suggestions
        suggestions_frame = tk.Frame(main_frame, bg="#2c3e50")
        suggestions_frame.pack(fill=tk.X, pady=(10, 0))

        suggestions_label = ttk.Label(
            suggestions_frame, text="Try saying:", style="Status.TLabel"
        )
        suggestions_label.pack(anchor=tk.W)

        suggestions = [
            "hello",
            "what time is it",
            "search for python",
            "open google",
            "help",
        ]
        for suggestion in suggestions:
            suggestion_btn = tk.Button(
                suggestions_frame,
                text=suggestion,
                command=lambda s=suggestion: self.simulate_command(s),
                bg="#34495e",
                fg="#ecf0f1",
                font=("Arial", 9),
                relief=tk.FLAT,
                padx=10,
                pady=2,
            )
            suggestion_btn.pack(side=tk.LEFT, padx=(0, 5))

    def setup_events(self):
        """Setup event handlers."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Bind keyboard shortcuts
        self.root.bind("<Control-l>", lambda e: self.toggle_listening())
        self.root.bind("<Control-h>", lambda e: self.show_help())
        self.root.bind("<Control-c>", lambda e: self.clear_history())

    def toggle_listening(self):
        """Toggle listening state."""
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        """Start listening for voice input."""
        self.is_listening = True
        self.listen_button.config(text="⏹️ Stop Listening", bg="#e74c3c")
        self.status_label.config(text="Listening... Speak now!")

        # Start recognition in a separate thread
        self.recognition_thread = threading.Thread(target=self._listen_for_speech)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()

    def stop_listening(self):
        """Stop listening for voice input."""
        self.is_listening = False
        self.listen_button.config(text="🎤 Start Listening", bg="#3498db")
        self.status_label.config(text="Ready to listen...")

    def _listen_for_speech(self):
        """Listen for speech input."""
        try:
            result = self.recognizer.recognize_once_async()
            self.root.after(0, self._handle_recognition_result, result)
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _handle_recognition_result(self, result):
        """Handle speech recognition result."""
        self.stop_listening()

        if result.reason == ResultReason.RecognizedSpeech:
            text = result.text.lower().strip()
            self.add_to_conversation(f"You: {result.text}", "user")

            # Process command
            response = self.process_command(text)
            self.add_to_conversation(f"Assistant: {response}", "assistant")

        elif result.reason == ResultReason.NoMatch:
            self.add_to_conversation(
                "Assistant: I didn't catch that. Could you please repeat?", "error"
            )
        elif result.reason == ResultReason.Canceled:
            self.add_to_conversation("Assistant: Recognition was canceled.", "error")

    def _handle_error(self, error_msg):
        """Handle recognition error."""
        self.stop_listening()
        self.add_to_conversation(f"Assistant: Error - {error_msg}", "error")

    def process_command(self, text: str) -> str:
        """Process voice commands."""
        text_lower = text.lower()

        # Check for exact command matches
        for command, handler in self.commands.items():
            if command in text_lower:
                return handler(text)

        # Check for partial matches
        if "time" in text_lower:
            return self._handle_time(text)
        elif "date" in text_lower:
            return self._handle_date(text)
        elif "search" in text_lower:
            return self._handle_search(text)
        elif "open" in text_lower:
            return self._handle_open(text)
        elif "help" in text_lower:
            return self._handle_help(text)
        elif "quit" in text_lower or "exit" in text_lower:
            return self._handle_quit(text)
        elif "stop" in text_lower:
            return self._handle_stop(text)

        return "I'm not sure how to help with that. Try saying 'help' for available commands."

    def _handle_greeting(self, text: str) -> str:
        """Handle greeting commands."""
        greetings = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! I'm ready to assist you.",
            "Hello! I'm your voice assistant. How may I help?",
        ]
        return greetings[hash(text) % len(greetings)]

    def _handle_time(self, text: str) -> str:
        """Handle time requests."""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def _handle_date(self, text: str) -> str:
        """Handle date requests."""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}"

    def _handle_weather(self, text: str) -> str:
        """Handle weather requests."""
        return "I'm sorry, I don't have access to weather information yet. This is a demo assistant."

    def _handle_search(self, text: str) -> str:
        """Handle search requests."""
        # Extract search query
        if "search for" in text.lower():
            query = text.lower().replace("search for", "").strip()
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Searching for '{query}' on Google..."
        return "Please say 'search for' followed by what you want to search for."

    def _handle_open(self, text: str) -> str:
        """Handle open requests."""
        if "open" in text.lower():
            site = text.lower().replace("open", "").strip()
            if "google" in site:
                webbrowser.open("https://www.google.com")
                return "Opening Google..."
            elif "youtube" in site:
                webbrowser.open("https://www.youtube.com")
                return "Opening YouTube..."
            elif "github" in site:
                webbrowser.open("https://github.com")
                return "Opening GitHub..."
            else:
                return f"I don't know how to open '{site}'. Try 'open google' or 'open youtube'."
        return "Please say 'open' followed by the website name."

    def _handle_help(self, text: str) -> str:
        """Handle help requests."""
        help_text = """
Available commands:
• "hello" - Greeting
• "what time is it" - Get current time
• "what's the date" - Get current date
• "search for [query]" - Search Google
• "open [website]" - Open websites (google, youtube, github)
• "help" - Show this help
• "quit" or "exit" - Close the assistant
• "stop" - Stop listening

Keyboard shortcuts:
• Ctrl+L - Toggle listening
• Ctrl+H - Show help
• Ctrl+C - Clear history
        """
        return help_text.strip()

    def _handle_quit(self, text: str) -> str:
        """Handle quit requests."""
        self.root.after(2000, self.root.quit)
        return "Goodbye! Closing the assistant in 2 seconds..."

    def _handle_stop(self, text: str) -> str:
        """Handle stop requests."""
        self.stop_listening()
        return "Stopped listening."

    def add_to_conversation(self, message: str, message_type: str = "normal"):
        """Add message to conversation history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.conversation_text.config(state=tk.NORMAL)

        # Color coding based on message type
        if message_type == "user":
            self.conversation_text.insert(tk.END, formatted_message)
        elif message_type == "assistant":
            self.conversation_text.insert(tk.END, formatted_message)
        elif message_type == "error":
            self.conversation_text.insert(tk.END, formatted_message)

        self.conversation_text.config(state=tk.DISABLED)
        self.conversation_text.see(tk.END)

        # Add to history
        self.conversation_history.append(
            {"timestamp": timestamp, "message": message, "type": message_type}
        )

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_text.config(state=tk.NORMAL)
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_text.config(state=tk.DISABLED)
        self.conversation_history.clear()
        self.add_to_conversation("Conversation history cleared.", "normal")

    def show_help(self):
        """Show help dialog."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Voice Assistant Help")
        help_window.geometry("600x400")
        help_window.configure(bg="#2c3e50")

        help_text = scrolledtext.ScrolledText(
            help_window, bg="#34495e", fg="#ecf0f1", font=("Consolas", 10), wrap=tk.WORD
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        help_content = """
🎤 Voice Assistant Demo

This demo showcases a practical voice assistant using the Groq Speech SDK.

FEATURES:
• Real-time speech recognition
• Command processing
• Visual feedback
• Conversation history
• Keyboard shortcuts

AVAILABLE COMMANDS:
• "hello" - Get a greeting
• "what time is it" - Get current time
• "what's the date" - Get current date
• "search for [query]" - Search Google
• "open [website]" - Open websites
• "help" - Show this help
• "quit" or "exit" - Close assistant
• "stop" - Stop listening

KEYBOARD SHORTCUTS:
• Ctrl+L - Toggle listening
• Ctrl+H - Show help
• Ctrl+C - Clear history

TECHNOLOGY:
• Groq Speech SDK for speech recognition
• Tkinter for GUI
• Threading for non-blocking recognition
• Real-time visual feedback

This is a demonstration of how the Groq Speech SDK can be used to build
practical voice-enabled applications with intuitive user interfaces.
        """

        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)

    def simulate_command(self, command: str):
        """Simulate a voice command (for testing)."""
        self.add_to_conversation(f"You: {command}", "user")
        response = self.process_command(command)
        self.add_to_conversation(f"Assistant: {response}", "assistant")

    def on_closing(self):
        """Handle window closing."""
        if self.is_listening:
            self.stop_listening()
        self.root.quit()

    def run(self):
        """Run the voice assistant."""
        self.add_to_conversation(
            "Voice Assistant started. Say 'hello' or 'help' to begin!", "normal"
        )
        self.root.mainloop()


def main():
    """Main function."""
    print("🎤 Starting Voice Assistant Demo...")
    print("This demo showcases a practical voice assistant with GUI interface.")
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
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
