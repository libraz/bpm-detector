"""BPM and Key detection algorithms."""

import warnings
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Configuration constants ---
SR_DEFAULT = 22_050
HOP_DEFAULT = 128
BIN_WIDTH = 0.5
RATIOS = [0.5, 2/3, 0.75, 1.0, 4/3, 1.5, 2.0, 3.0, 4.0]
TOL = 0.05
THRESH_HIGHER = 0.15  # 15% of total votes

# --- Key detection constants ---
# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Note names for display
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class BPMDetector:
    """BPM detection using harmonic clustering."""
    
    def __init__(self, sr: int = SR_DEFAULT, hop_length: int = HOP_DEFAULT):
        self.sr = sr
        self.hop_length = hop_length
    
    def harmonic_cluster(self, bpms: np.ndarray, votes: np.ndarray) -> Dict[float, List[Tuple[float, int]]]:
        """Group BPM candidates into harmonic clusters."""
        clusters = {}
        for bpm, hit in sorted(zip(bpms, votes), key=lambda x: -x[1]):
            for base in list(clusters):
                r = bpm / base
                if any(abs(r - k) < TOL or abs(r - 1/k) < TOL for k in RATIOS):
                    clusters[base].append((bpm, hit))
                    break
            else:
                clusters[bpm] = [(bpm, hit)]
        return clusters
    
    def smart_choice(self, clusters: Dict[float, List[Tuple[float, int]]], total_votes: int) -> Tuple[float, float]:
        """Choose the best BPM from clusters using smart selection."""
        # base cluster = largest votes
        base, base_vals = max(clusters.items(), key=lambda kv: sum(v for _, v in kv[1]))
        base_votes = sum(v for _, v in base_vals)

        higher = [(rep, sum(v for _, v in vals))
                  for rep, vals in clusters.items() if rep > base]
        higher.sort(key=lambda x: -x[1])

        if higher and higher[0][1] / total_votes >= THRESH_HIGHER:
            rep_bpm = max(higher[0][0], higher[0][0])
            conf = 100 * higher[0][1] / total_votes
        else:
            rep_bpm = max(base_vals, key=lambda x: x[1])[0]
            conf = 100 * base_votes / total_votes
        return rep_bpm, conf
    
    def detect(self, y: np.ndarray, sr: int, min_bpm: float = 40.0, 
               max_bpm: float = 300.0, start_bpm: float = 150.0) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Detect BPM from audio signal."""
        # Choose API without FutureWarning
        if hasattr(librosa, 'feature') and hasattr(librosa.feature, 'rhythm') and hasattr(librosa.feature.rhythm, 'tempo'):
            tempo_func = librosa.feature.rhythm.tempo
        else:
            warnings.filterwarnings('ignore', category=FutureWarning, message='.*librosa.beat.tempo.*')
            tempo_func = librosa.beat.tempo

        cands = tempo_func(y=y, sr=sr, aggregate=None,
                          hop_length=self.hop_length,
                          max_tempo=max_bpm,
                          start_bpm=start_bpm)

        bins = np.arange(min_bpm, max_bpm + BIN_WIDTH, BIN_WIDTH)
        hist, edges = np.histogram(cands, bins=bins)
        top_idx = hist.argsort()[::-1][:10]
        top_bpms = edges[top_idx]
        top_hits = hist[top_idx]

        clusters = self.harmonic_cluster(top_bpms, top_hits)
        rep_bpm, conf = self.smart_choice(clusters, hist.sum())

        return rep_bpm, conf, top_bpms, top_hits


class KeyDetector:
    """Musical key detection using chroma features and key profiles."""
    
    def __init__(self, hop_length: int = HOP_DEFAULT):
        self.hop_length = hop_length
    
    def detect(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Detect musical key from audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            tuple: (Key name, Confidence)
        """
        # Calculate chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        
        # Take average over time axis
        chroma_mean = np.mean(chroma, axis=1)
        
        # Calculate correlation with each key
        correlations = []
        
        # 12 major keys
        for i in range(12):
            # Rotate profile to correspond to each key
            rotated_major = np.roll(MAJOR_PROFILE, i)
            correlation = np.corrcoef(chroma_mean, rotated_major)[0, 1]
            correlations.append((NOTE_NAMES[i] + ' Major', correlation))
        
        # 12 minor keys
        for i in range(12):
            # Rotate profile to correspond to each key
            rotated_minor = np.roll(MINOR_PROFILE, i)
            correlation = np.corrcoef(chroma_mean, rotated_minor)[0, 1]
            correlations.append((NOTE_NAMES[i] + ' Minor', correlation))
        
        # Select key with highest correlation
        best_key, best_correlation = max(correlations, key=lambda x: x[1])
        confidence = max(0, best_correlation * 100)  # Clip negative values to 0
        
        return best_key, confidence


class AudioAnalyzer:
    """Main analyzer combining BPM and key detection."""
    
    def __init__(self, sr: int = SR_DEFAULT, hop_length: int = HOP_DEFAULT):
        self.sr = sr
        self.hop_length = hop_length
        self.bpm_detector = BPMDetector(sr, hop_length)
        self.key_detector = KeyDetector(hop_length)
    
    def analyze_file(self, path: str, detect_key: bool = False, 
                    min_bpm: float = 40.0, max_bpm: float = 300.0, 
                    start_bpm: float = 150.0, progress_callback=None) -> Dict[str, Any]:
        """Analyze audio file for BPM and optionally key.
        
        Args:
            path: Path to audio file
            detect_key: Whether to detect musical key
            min_bpm: Minimum BPM to consider
            max_bpm: Maximum BPM to consider
            start_bpm: Starting BPM for detection
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing analysis results
        """
        # Load audio
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        if progress_callback:
            progress_callback(25)
        
        # BPM detection
        bpm, bpm_conf, top_bpms, top_hits = self.bpm_detector.detect(
            y, sr, min_bpm, max_bpm, start_bpm
        )
        if progress_callback:
            progress_callback(25)
        
        results = {
            'filename': path,
            'bpm': bpm,
            'bpm_confidence': bpm_conf,
            'bpm_candidates': list(zip(top_bpms, top_hits))
        }
        
        # Key detection
        if detect_key:
            key, key_conf = self.key_detector.detect(y, sr)
            results['key'] = key
            results['key_confidence'] = key_conf
            if progress_callback:
                progress_callback(25)
        
        if progress_callback:
            progress_callback(25)
        
        return results