"""BPM and Key Detector - A comprehensive music analysis tool."""

from .auto_parallel import AutoParallelConfig, ParallelConfig, SystemMonitor
from .chord_analyzer import ChordProgressionAnalyzer
from .cli import main
from .dynamics_analyzer import DynamicsAnalyzer
from .harmony_analyzer import HarmonyAnalyzer
from .key_detector import KeyDetector as EnhancedKeyDetector
from .melody_analyzer import MelodyAnalyzer
from .melody_harmony_analyzer import MelodyHarmonyAnalyzer
from .music_analyzer import AudioAnalyzer, BPMDetector, KeyDetector
from .parallel_analyzer import SmartParallelAudioAnalyzer
from .progress_manager import ProgressManager, create_progress_display
from .rhythm_analyzer import RhythmAnalyzer
from .similarity_engine import SimilarityEngine
from .structure_analyzer import StructureAnalyzer
from .timbre_analyzer import TimbreAnalyzer

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
    "MelodyAnalyzer",
    "HarmonyAnalyzer",
    "EnhancedKeyDetector",
    "DynamicsAnalyzer",
    "SimilarityEngine",
    "SmartParallelAudioAnalyzer",
    "AutoParallelConfig",
    "SystemMonitor",
    "ParallelConfig",
    "ProgressManager",
    "create_progress_display",
    "main",
]


def hello() -> str:
    """Legacy function for compatibility."""
    return "Hello from bpm-detector!"
