"""BPM and Key Detector - A comprehensive music analysis tool."""

from .cli import main
from .music_analyzer import AudioAnalyzer, BPMDetector, KeyDetector
from .chord_analyzer import ChordProgressionAnalyzer
from .structure_analyzer import StructureAnalyzer
from .rhythm_analyzer import RhythmAnalyzer
from .timbre_analyzer import TimbreAnalyzer
from .melody_harmony_analyzer import MelodyHarmonyAnalyzer
from .melody_analyzer import MelodyAnalyzer
from .harmony_analyzer import HarmonyAnalyzer
from .key_detector import KeyDetector as EnhancedKeyDetector
from .dynamics_analyzer import DynamicsAnalyzer
from .similarity_engine import SimilarityEngine
from .parallel_analyzer import SmartParallelAudioAnalyzer
from .auto_parallel import AutoParallelConfig, SystemMonitor, ParallelConfig
from .progress_manager import ProgressManager, create_progress_display

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
    "main"
]


def hello() -> str:
    """Legacy function for compatibility."""
    return "Hello from bpm-detector!"
