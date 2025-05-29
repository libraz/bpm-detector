"""Melody and harmony analysis module."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any, Optional
from scipy.signal import find_peaks
from scipy.stats import entropy


class MelodyHarmonyAnalyzer:
    """Analyzes melodic and harmonic content."""
    
    # Note names for pitch analysis
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Interval names and their semitone distances
    INTERVALS = {
        0: 'unison',
        1: 'minor_second',
        2: 'major_second', 
        3: 'minor_third',
        4: 'major_third',
        5: 'perfect_fourth',
        6: 'tritone',
        7: 'perfect_fifth',
        8: 'minor_sixth',
        9: 'major_sixth',
        10: 'minor_seventh',
        11: 'major_seventh'
    }
    
    # Consonance ratings for intervals (0-1, higher = more consonant)
    CONSONANCE_RATINGS = {
        0: 1.0,    # unison
        1: 0.1,    # minor second
        2: 0.3,    # major second
        3: 0.6,    # minor third
        4: 0.8,    # major third
        5: 0.9,    # perfect fourth
        6: 0.2,    # tritone
        7: 1.0,    # perfect fifth
        8: 0.7,    # minor sixth
        9: 0.8,    # major sixth
        10: 0.4,   # minor seventh
        11: 0.5    # major seventh
    }
    
    def __init__(self, hop_length: int = 512, fmin: float = 80.0, fmax: float = 2000.0):
        """Initialize melody harmony analyzer.
        
        Args:
            hop_length: Hop length for analysis
            fmin: Minimum frequency for pitch tracking
            fmax: Maximum frequency for pitch tracking
        """
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        
    def extract_melody(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract melody line from audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing melody information
        """
        # Separate harmonic component for better pitch tracking
        harmonic, _ = librosa.effects.hpss(y)
        
        # Extract fundamental frequency using PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            harmonic,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=self.hop_length
        )
        
        # Convert to MIDI notes
        midi_notes = librosa.hz_to_midi(f0)
        
        # Clean up melody (remove unvoiced segments)
        clean_melody = midi_notes.copy()
        clean_melody[~voiced_flag] = np.nan
        
        # Calculate time axis
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=self.hop_length
        )
        
        return {
            'f0': f0,
            'midi_notes': midi_notes,
            'clean_melody': clean_melody,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'times': times
        }
    
    def analyze_melodic_range(self, melody: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze melodic range characteristics.
        
        Args:
            melody: Melody information from extract_melody
            
        Returns:
            Dictionary of range characteristics
        """
        clean_melody = melody['clean_melody']
        valid_notes = clean_melody[~np.isnan(clean_melody)]
        
        if len(valid_notes) == 0:
            return {
                'range_semitones': 0.0,
                'range_octaves': 0.0,
                'lowest_note': 0.0,
                'highest_note': 0.0,
                'mean_pitch': 0.0,
                'pitch_std': 0.0
            }
        
        lowest = np.min(valid_notes)
        highest = np.max(valid_notes)
        range_semitones = highest - lowest
        range_octaves = range_semitones / 12.0
        
        return {
            'range_semitones': float(range_semitones),
            'range_octaves': float(range_octaves),
            'lowest_note': float(lowest),
            'highest_note': float(highest),
            'mean_pitch': float(np.mean(valid_notes)),
            'pitch_std': float(np.std(valid_notes))
        }
    
    def analyze_melodic_direction(self, melody: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze melodic direction and contour.
        
        Args:
            melody: Melody information from extract_melody
            
        Returns:
            Dictionary of direction characteristics
        """
        clean_melody = melody['clean_melody']
        valid_indices = ~np.isnan(clean_melody)
        
        if np.sum(valid_indices) < 2:
            return {
                'direction_tendency': 'static',
                'ascending_ratio': 0.0,
                'descending_ratio': 0.0,
                'static_ratio': 1.0,
                'contour_complexity': 0.0,
                'average_step_size': 0.0
            }
        
        # Calculate pitch differences
        valid_melody = clean_melody[valid_indices]
        pitch_diffs = np.diff(valid_melody)
        
        if len(pitch_diffs) == 0:
            return {
                'direction_tendency': 'static',
                'ascending_ratio': 0.0,
                'descending_ratio': 0.0,
                'static_ratio': 1.0,
                'contour_complexity': 0.0,
                'average_step_size': 0.0
            }
        
        # Count direction changes
        ascending = np.sum(pitch_diffs > 0.5)  # Threshold to avoid noise
        descending = np.sum(pitch_diffs < -0.5)
        static = len(pitch_diffs) - ascending - descending
        
        total = len(pitch_diffs)
        ascending_ratio = ascending / total
        descending_ratio = descending / total
        static_ratio = static / total
        
        # Determine overall tendency
        if ascending_ratio > descending_ratio + 0.1:
            direction_tendency = 'ascending'
        elif descending_ratio > ascending_ratio + 0.1:
            direction_tendency = 'descending'
        else:
            direction_tendency = 'balanced'
        
        # Calculate contour complexity (number of direction changes)
        direction_changes = 0
        for i in range(1, len(pitch_diffs)):
            if (pitch_diffs[i-1] > 0) != (pitch_diffs[i] > 0):
                direction_changes += 1
        
        contour_complexity = direction_changes / len(pitch_diffs) if len(pitch_diffs) > 0 else 0
        
        # Average step size
        average_step_size = np.mean(np.abs(pitch_diffs))
        
        return {
            'direction_tendency': direction_tendency,
            'ascending_ratio': float(ascending_ratio),
            'descending_ratio': float(descending_ratio),
            'static_ratio': float(static_ratio),
            'contour_complexity': float(contour_complexity),
            'average_step_size': float(average_step_size)
        }
    
    def analyze_interval_distribution(self, melody: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze distribution of melodic intervals.
        
        Args:
            melody: Melody information from extract_melody
            
        Returns:
            Dictionary of interval distributions
        """
        clean_melody = melody['clean_melody']
        valid_indices = ~np.isnan(clean_melody)
        
        if np.sum(valid_indices) < 2:
            return {interval: 0.0 for interval in self.INTERVALS.values()}
        
        valid_melody = clean_melody[valid_indices]
        pitch_diffs = np.diff(valid_melody)
        
        # Convert to semitone intervals (absolute values)
        intervals = np.abs(pitch_diffs)
        
        # Count intervals
        interval_counts = {interval: 0 for interval in self.INTERVALS.values()}
        
        for interval_size in intervals:
            # Round to nearest semitone
            semitone = int(round(interval_size)) % 12
            interval_name = self.INTERVALS.get(semitone, 'other')
            if interval_name in interval_counts:
                interval_counts[interval_name] += 1
        
        # Convert to ratios
        total_intervals = len(intervals)
        if total_intervals > 0:
            interval_distribution = {
                interval: count / total_intervals 
                for interval, count in interval_counts.items()
            }
        else:
            interval_distribution = {interval: 0.0 for interval in self.INTERVALS.values()}
        
        return interval_distribution
    
    def analyze_harmony_complexity(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic complexity.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of harmony complexity measures
        """
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        if chroma.size == 0:
            return {
                'harmonic_entropy': 0.0,
                'harmonic_variance': 0.0,
                'harmonic_complexity': 0.0,
                'spectral_complexity': 0.0
            }
        
        # Calculate harmonic entropy
        chroma_mean = np.mean(chroma, axis=1)
        chroma_normalized = chroma_mean / (np.sum(chroma_mean) + 1e-8)
        harmonic_entropy = entropy(chroma_normalized + 1e-8)
        
        # Normalize entropy (max entropy for 12 bins is log(12))
        max_entropy = np.log(12)
        normalized_entropy = harmonic_entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate harmonic variance
        harmonic_variance = np.var(chroma, axis=1).mean()
        
        # Calculate spectral complexity
        stft = librosa.stft(y, hop_length=self.hop_length)
        spectral_complexity = np.std(np.abs(stft)) / (np.mean(np.abs(stft)) + 1e-8)
        
        # Overall harmonic complexity
        harmonic_complexity = (normalized_entropy + min(1.0, harmonic_variance) + 
                             min(1.0, spectral_complexity)) / 3.0
        
        return {
            'harmonic_entropy': float(normalized_entropy),
            'harmonic_variance': float(harmonic_variance),
            'harmonic_complexity': float(harmonic_complexity),
            'spectral_complexity': float(min(1.0, spectral_complexity))
        }
    
    def analyze_consonance(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic consonance/dissonance.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of consonance measures
        """
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        if chroma.size == 0:
            return {
                'consonance_level': 0.0,
                'dissonance_level': 0.0,
                'harmonic_tension': 0.0
            }
        
        consonance_scores = []
        
        # Analyze each time frame
        for frame in range(chroma.shape[1]):
            frame_chroma = chroma[:, frame]
            
            # Find active notes (above threshold)
            threshold = np.max(frame_chroma) * 0.3
            active_notes = np.where(frame_chroma > threshold)[0]
            
            if len(active_notes) < 2:
                consonance_scores.append(1.0)  # Single note = consonant
                continue
            
            # Calculate consonance for all pairs of active notes
            frame_consonance = []
            
            for i in range(len(active_notes)):
                for j in range(i + 1, len(active_notes)):
                    interval = abs(active_notes[i] - active_notes[j]) % 12
                    consonance = self.CONSONANCE_RATINGS.get(interval, 0.5)
                    
                    # Weight by note strengths
                    weight = frame_chroma[active_notes[i]] * frame_chroma[active_notes[j]]
                    frame_consonance.append(consonance * weight)
            
            if frame_consonance:
                consonance_scores.append(np.mean(frame_consonance))
            else:
                consonance_scores.append(1.0)
        
        # Calculate overall measures
        consonance_level = np.mean(consonance_scores)
        dissonance_level = 1.0 - consonance_level
        
        # Calculate harmonic tension (variance in consonance)
        harmonic_tension = np.std(consonance_scores)
        
        return {
            'consonance_level': float(consonance_level),
            'dissonance_level': float(dissonance_level),
            'harmonic_tension': float(harmonic_tension)
        }
    
    def analyze_pitch_stability(self, melody: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze pitch stability and vibrato.
        
        Args:
            melody: Melody information from extract_melody
            
        Returns:
            Dictionary of pitch stability measures
        """
        f0 = melody['f0']
        voiced_flag = melody['voiced_flag']
        
        if not np.any(voiced_flag):
            return {
                'pitch_stability': 0.0,
                'vibrato_rate': 0.0,
                'vibrato_extent': 0.0,
                'pitch_drift': 0.0
            }
        
        # Extract voiced segments
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) < 10:  # Need sufficient data
            return {
                'pitch_stability': 0.0,
                'vibrato_rate': 0.0,
                'vibrato_extent': 0.0,
                'pitch_drift': 0.0
            }
        
        # Calculate pitch stability (inverse of coefficient of variation)
        pitch_cv = np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-8)
        pitch_stability = 1.0 / (1.0 + pitch_cv)
        
        # Detect vibrato
        vibrato_rate, vibrato_extent = self._detect_vibrato(voiced_f0)
        
        # Calculate pitch drift (long-term trend)
        pitch_drift = self._calculate_pitch_drift(voiced_f0)
        
        return {
            'pitch_stability': float(pitch_stability),
            'vibrato_rate': float(vibrato_rate),
            'vibrato_extent': float(vibrato_extent),
            'pitch_drift': float(pitch_drift)
        }
    
    def _detect_vibrato(self, f0: np.ndarray) -> Tuple[float, float]:
        """Detect vibrato characteristics.
        
        Args:
            f0: Fundamental frequency array
            
        Returns:
            (vibrato_rate, vibrato_extent)
        """
        if len(f0) < 20:
            return 0.0, 0.0
        
        # Detrend the pitch
        detrended = f0 - np.mean(f0)
        
        # Calculate autocorrelation to find periodic patterns
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (indicating periodicity)
        peaks, _ = find_peaks(autocorr, height=np.max(autocorr) * 0.3, distance=5)
        
        if len(peaks) == 0:
            return 0.0, 0.0
        
        # Estimate vibrato rate (first significant peak)
        vibrato_period = peaks[0] if len(peaks) > 0 else 0
        
        # Convert to Hz (assuming hop_length corresponds to time)
        # This is a rough approximation
        vibrato_rate = 1.0 / (vibrato_period + 1e-8) if vibrato_period > 0 else 0.0
        vibrato_rate = min(10.0, vibrato_rate)  # Cap at reasonable value
        
        # Calculate vibrato extent (amplitude of oscillation)
        vibrato_extent = np.std(detrended) / (np.mean(f0) + 1e-8)
        
        return vibrato_rate, vibrato_extent
    
    def _calculate_pitch_drift(self, f0: np.ndarray) -> float:
        """Calculate long-term pitch drift.
        
        Args:
            f0: Fundamental frequency array
            
        Returns:
            Pitch drift measure
        """
        if len(f0) < 10:
            return 0.0
        
        # Fit linear trend
        x = np.arange(len(f0))
        coeffs = np.polyfit(x, f0, 1)
        slope = coeffs[0]
        
        # Normalize by mean frequency
        drift = abs(slope) / (np.mean(f0) + 1e-8)
        
        return min(1.0, drift * 100)  # Scale and cap
    
    def analyze_harmonic_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic rhythm (rate of harmonic change).
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of harmonic rhythm measures
        """
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        if chroma.shape[1] < 2:
            return {
                'harmonic_change_rate': 0.0,
                'harmonic_stability': 0.0,
                'harmonic_rhythm_regularity': 0.0
            }
        
        # Calculate frame-to-frame harmonic changes
        chroma_diffs = np.diff(chroma, axis=1)
        change_magnitudes = np.sqrt(np.sum(chroma_diffs**2, axis=0))
        
        # Calculate harmonic change rate
        total_time = chroma.shape[1] * self.hop_length / sr
        harmonic_change_rate = np.sum(change_magnitudes > np.std(change_magnitudes)) / total_time
        
        # Calculate harmonic stability (inverse of change variance)
        harmonic_stability = 1.0 / (1.0 + np.var(change_magnitudes))
        
        # Calculate rhythm regularity
        if len(change_magnitudes) > 4:
            # Find peaks in change magnitudes (harmonic changes)
            peaks, _ = find_peaks(change_magnitudes, height=np.mean(change_magnitudes))
            
            if len(peaks) > 2:
                peak_intervals = np.diff(peaks)
                rhythm_regularity = 1.0 / (1.0 + np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8))
            else:
                rhythm_regularity = 0.0
        else:
            rhythm_regularity = 0.0
        
        return {
            'harmonic_change_rate': float(harmonic_change_rate),
            'harmonic_stability': float(harmonic_stability),
            'harmonic_rhythm_regularity': float(rhythm_regularity)
        }
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete melody and harmony analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Complete melody and harmony analysis results
        """
        # Extract melody
        melody = self.extract_melody(y, sr)
        
        # Analyze melodic characteristics
        melodic_range = self.analyze_melodic_range(melody)
        melodic_direction = self.analyze_melodic_direction(melody)
        interval_distribution = self.analyze_interval_distribution(melody)
        pitch_stability = self.analyze_pitch_stability(melody)
        
        # Analyze harmonic characteristics
        harmony_complexity = self.analyze_harmony_complexity(y, sr)
        consonance = self.analyze_consonance(y, sr)
        harmonic_rhythm = self.analyze_harmonic_rhythm(y, sr)
        
        return {
            'melodic_range': melodic_range,
            'melodic_direction': melodic_direction,
            'interval_distribution': interval_distribution,
            'pitch_stability': pitch_stability,
            'harmony_complexity': harmony_complexity,
            'consonance': consonance,
            'harmonic_rhythm': harmonic_rhythm,
            'melody_present': bool(np.any(melody['voiced_flag'])),
            'melody_coverage': float(np.mean(melody['voiced_flag'])) if len(melody['voiced_flag']) > 0 else 0.0
        }