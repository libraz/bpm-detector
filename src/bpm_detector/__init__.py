"""BPM and Key Detector - A Python tool for automatic detection of BPM and musical key."""

from .cli import main
from .detector import AudioAnalyzer, BPMDetector, KeyDetector

__version__ = "0.1.0"
__author__ = "libraz"
__email__ = "libraz@libraz.net"

__all__ = ["AudioAnalyzer", "BPMDetector", "KeyDetector", "main"]


def hello() -> str:
    """Legacy function for compatibility."""
    return "Hello from bpm-detector!"
