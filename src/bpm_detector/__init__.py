"""BPM and Key Detector - A comprehensive music analysis tool."""

from .cli import main
from .music_analyzer import AudioAnalyzer, BPMDetector, KeyDetector
from .chord_analyzer import ChordProgressionAnalyzer
from .structure_analyzer import StructureAnalyzer
from .rhythm_analyzer import RhythmAnalyzer
from .timbre_analyzer import TimbreAnalyzer
from .melody_harmony_analyzer import MelodyHarmonyAnalyzer
from .dynamics_analyzer import DynamicsAnalyzer
from .similarity_engine import SimilarityEngine

__version__ = "0.2.0"
__author__ = "libraz"
__email__ = "libraz@libraz.net"

__all__ = [
    "AudioAnalyzer",
    "BPMDetector",
    "KeyDetector",
    "ChordProgressionAnalyzer",
    "StructureAnalyzer",
    "RhythmAnalyzer",
    "TimbreAnalyzer",
    "MelodyHarmonyAnalyzer",
    "DynamicsAnalyzer",
    "SimilarityEngine",
    "main"
]


def hello() -> str:
    """Legacy function for compatibility."""
    return "Hello from bpm-detector!"
