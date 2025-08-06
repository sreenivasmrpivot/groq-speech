"""
Groq Speech SDK API Server Package.

This package provides REST API, WebSocket, and gRPC endpoints for the Groq Speech SDK.
"""

__version__ = "1.0.0"
__author__ = "Groq Speech SDK Contributors"
__email__ = "support@groq.com"

from .server import create_app, get_app

__all__ = ["create_app", "get_app"]
