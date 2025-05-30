"""Rhythm and time signature analysis module."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
from scipy.signal import find_peaks, correlate
from scipy.stats import entropy


class RhythmAnalyzer:
    """Analyzes rhythm patterns and time signatures."""
    
    # Common time signatures and their beat patterns
    TIME_SIGNATURES = {
        '4/4': {'beats_per_measure': 4, 'beat_unit': 4, 'strong_beats': [0]},
        '3/4': {'beats_per_measure': 3, 'beat_unit': 4, 'strong_beats': [0]},
        '2/4': {'beats_per_measure': 2, 'beat_unit': 4, 'strong_beats': [0]},
        '6/8': {'beats_per_measure': 6, 'beat_unit': 8, 'strong_beats': [0, 3]},
        '9/8': {'beats_per_measure': 9, 'beat_unit': 8, 'strong_beats': [0, 3, 6]},
        '12/8': {'beats_per_measure': 12, 'beat_unit': 8, 'strong_beats': [0, 3, 6, 9]},
        '5/4': {'beats_per_measure': 5, 'beat_unit': 4, 'strong_beats': [0]},
        '7/8': {'beats_per_measure': 7, 'beat_unit': 8, 'strong_beats': [0, 3]}
    }
    
    def __init__(self, hop_length: int = 512):
        """Initialize rhythm analyzer.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length
        
    def extract_onset_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract onset-related features.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of onset features
        """
        features = {}
        
        # Onset strength
        features['onset_strength'] = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Onset times
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length, units='time'
        )
        features['onset_times'] = onsets
        
        # Onset frames
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length, units='frames'
        )
        features['onset_frames'] = onset_frames
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        features['tempo'] = tempo
        features['beat_frames'] = beats
        features['beat_times'] = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        return features
    
    def detect_time_signature(self, onset_features: Dict[str, np.ndarray],
                             sr: int) -> Tuple[str, float]:
        """Detect time signature from onset features.
        
        Args:
            onset_features: Dictionary of onset features
            sr: Sample rate
            
        Returns:
            (time_signature, confidence)
        """
        beat_times = onset_features.get('beat_times', np.array([]))
        
        if len(beat_times) < 8:  # Need sufficient beats
            return '4/4', 0.5
        
        # Calculate inter-beat intervals
        beat_intervals = np.diff(beat_times)
        
        if len(beat_intervals) == 0:
            return '4/4', 0.5
        
        # Analyze beat patterns for different time signatures
        # Start with strong bias toward 4/4
        best_signature = '4/4'
        best_score = 0.6  # Give 4/4 a head start
        
        # Score 4/4 first with bias
        four_four_score = self._score_time_signature(beat_intervals, self.TIME_SIGNATURES['4/4'], onset_features)
        four_four_score += 0.3  # Add bias for 4/4
        
        if four_four_score > best_score:
            best_score = four_four_score
            best_signature = '4/4'
        
        # Only check other signatures if they significantly outperform 4/4
        for signature, info in self.TIME_SIGNATURES.items():
            if signature == '4/4':
                continue  # Already checked
                
            score = self._score_time_signature(beat_intervals, info, onset_features)
            
            # Require significantly higher score to override 4/4
            threshold = 0.8 if signature in ['3/4', '2/4'] else 0.9
            
            if score > threshold and score > best_score:
                best_score = score
                best_signature = signature
        
        # Cap confidence and ensure 4/4 for uncertain cases
        if best_score < 0.7 and best_signature != '4/4':
            return '4/4', 0.6
        
        return best_signature, min(1.0, best_score)
    
    def _score_time_signature(self, beat_intervals: np.ndarray, 
                             signature_info: Dict[str, Any],
                             onset_features: Dict[str, np.ndarray]) -> float:
        """Score how well beat intervals fit a time signature.
        
        Args:
            beat_intervals: Inter-beat intervals
            signature_info: Time signature information
            onset_features: Onset features
            
        Returns:
            Fit score (0-1)
        """
        beats_per_measure = signature_info['beats_per_measure']
        
        if len(beat_intervals) < beats_per_measure * 2:
            return 0.0
        
        # Group beats into measures
        measure_length = beats_per_measure
        n_complete_measures = len(beat_intervals) // measure_length
        
        if n_complete_measures < 2:
            return 0.0
        
        # Calculate consistency of measure patterns
        measure_patterns = []
        
        for i in range(n_complete_measures):
            start_idx = i * measure_length
            end_idx = start_idx + measure_length
            measure_pattern = beat_intervals[start_idx:end_idx]
            measure_patterns.append(measure_pattern)
        
        # Calculate pattern consistency
        if not measure_patterns:
            return 0.0
        
        # Compute variance across measures
        pattern_matrix = np.array(measure_patterns)
        pattern_variance = np.mean(np.var(pattern_matrix, axis=0))
        
        # Lower variance = higher consistency = better fit
        consistency_score = np.exp(-pattern_variance * 10)  # Exponential decay
        
        # Check for strong beat emphasis
        strong_beat_score = self._analyze_strong_beats(
            onset_features, signature_info, beat_intervals
        )
        
        # Combine scores
        total_score = (consistency_score + strong_beat_score) / 2.0
        
        return min(1.0, total_score)
    
    def _analyze_strong_beats(self, onset_features: Dict[str, np.ndarray],
                             signature_info: Dict[str, Any],
                             beat_intervals: np.ndarray) -> float:
        """Analyze emphasis on strong beats.
        
        Args:
            onset_features: Onset features
            signature_info: Time signature information
            beat_intervals: Inter-beat intervals
            
        Returns:
            Strong beat emphasis score (0-1)
        """
        onset_strength = onset_features.get('onset_strength', np.array([]))
        beat_frames = onset_features.get('beat_frames', np.array([]))
        
        if len(onset_strength) == 0 or len(beat_frames) == 0:
            return 0.0
        
        beats_per_measure = signature_info['beats_per_measure']
        strong_beats = signature_info['strong_beats']
        
        # Group beats into measures
        n_complete_measures = len(beat_frames) // beats_per_measure
        
        if n_complete_measures < 2:
            return 0.0
        
        strong_beat_strengths = []
        weak_beat_strengths = []
        
        for measure in range(n_complete_measures):
            measure_start = measure * beats_per_measure
            
            for beat_in_measure in range(beats_per_measure):
                beat_idx = measure_start + beat_in_measure
                
                if beat_idx >= len(beat_frames):
                    break
                
                frame_idx = beat_frames[beat_idx]
                
                if frame_idx < len(onset_strength):
                    strength = onset_strength[frame_idx]
                    
                    if beat_in_measure in strong_beats:
                        strong_beat_strengths.append(strength)
                    else:
                        weak_beat_strengths.append(strength)
        
        if not strong_beat_strengths or not weak_beat_strengths:
            return 0.0
        
        # Calculate ratio of strong to weak beat emphasis
        strong_avg = np.mean(strong_beat_strengths)
        weak_avg = np.mean(weak_beat_strengths)
        
        if weak_avg == 0:
            return 1.0 if strong_avg > 0 else 0.0
        
        emphasis_ratio = strong_avg / weak_avg
        
        # Convert to 0-1 score (ratio > 1 is good)
        score = min(1.0, (emphasis_ratio - 1.0) / 2.0 + 0.5)
        
        return max(0.0, score)
    
    def extract_rhythm_patterns(self, onset_features: Dict[str, np.ndarray],
                               time_signature: str) -> Dict[str, Any]:
        """Extract rhythm patterns from onset features.
        
        Args:
            onset_features: Dictionary of onset features
            time_signature: Detected time signature
            
        Returns:
            Dictionary of rhythm pattern features
        """
        onset_times = onset_features.get('onset_times', np.array([]))
        beat_times = onset_features.get('beat_times', np.array([]))
        
        if len(onset_times) == 0 or len(beat_times) == 0:
            return {
                'rhythmic_complexity': 0.0,
                'syncopation_level': 0.0,
                'pattern_regularity': 0.0,
                'subdivision_density': 0.0
            }
        
        # Calculate rhythmic complexity
        complexity = self._calculate_rhythmic_complexity(onset_times, beat_times)
        
        # Calculate syncopation level
        syncopation = self._calculate_syncopation(onset_times, beat_times, time_signature)
        
        # Calculate pattern regularity
        regularity = self._calculate_pattern_regularity(onset_times, beat_times)
        
        # Calculate subdivision density
        subdivision_density = self._calculate_subdivision_density(onset_times, beat_times)
        
        return {
            'rhythmic_complexity': complexity,
            'syncopation_level': syncopation,
            'pattern_regularity': regularity,
            'subdivision_density': subdivision_density
        }
    
    def _calculate_rhythmic_complexity(self, onset_times: np.ndarray, 
                                     beat_times: np.ndarray) -> float:
        """Calculate rhythmic complexity score.
        
        Args:
            onset_times: Onset times
            beat_times: Beat times
            
        Returns:
            Complexity score (0-1)
        """
        if len(onset_times) == 0 or len(beat_times) == 0:
            return 0.0
        
        # Calculate onset density (onsets per beat)
        if len(beat_times) < 2:
            return 0.0
        
        total_duration = beat_times[-1] - beat_times[0]
        onset_density = len(onset_times) / total_duration if total_duration > 0 else 0
        
        # Calculate rhythmic entropy
        beat_intervals = np.diff(beat_times)
        
        if len(beat_intervals) == 0:
            return 0.0
        
        # Quantize onset times to beat grid
        quantized_onsets = []
        
        for onset in onset_times:
            # Find closest beat
            beat_distances = np.abs(beat_times - onset)
            closest_beat_idx = np.argmin(beat_distances)
            
            # Calculate position within beat (0-1)
            if closest_beat_idx < len(beat_times) - 1:
                beat_start = beat_times[closest_beat_idx]
                beat_end = beat_times[closest_beat_idx + 1]
                position = (onset - beat_start) / (beat_end - beat_start)
            else:
                position = 0.0
            
            quantized_onsets.append(position)
        
        # Calculate entropy of onset positions
        if quantized_onsets:
            hist, _ = np.histogram(quantized_onsets, bins=8, range=(0, 1))
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            rhythmic_entropy = entropy(hist + 1e-8)  # Add small value to avoid log(0)
        else:
            rhythmic_entropy = 0.0
        
        # Normalize entropy (max entropy for 8 bins is log(8))
        max_entropy = np.log(8)
        normalized_entropy = rhythmic_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine density and entropy
        complexity = (min(1.0, onset_density / 10.0) + normalized_entropy) / 2.0
        
        return complexity
    
    def _calculate_syncopation(self, onset_times: np.ndarray, 
                              beat_times: np.ndarray, time_signature: str) -> float:
        """Calculate syncopation level.
        
        Args:
            onset_times: Onset times
            beat_times: Beat times
            time_signature: Time signature
            
        Returns:
            Syncopation level (0-1)
        """
        if len(onset_times) == 0 or len(beat_times) == 0:
            return 0.0
        
        signature_info = self.TIME_SIGNATURES.get(time_signature, self.TIME_SIGNATURES['4/4'])
        beats_per_measure = signature_info['beats_per_measure']
        strong_beats = signature_info['strong_beats']
        
        # Count onsets on strong vs weak beats
        strong_beat_onsets = 0
        weak_beat_onsets = 0
        off_beat_onsets = 0
        
        for onset in onset_times:
            # Find closest beat
            beat_distances = np.abs(beat_times - onset)
            closest_beat_idx = np.argmin(beat_distances)
            closest_distance = beat_distances[closest_beat_idx]
            
            # Determine if onset is on beat, off beat, or syncopated
            beat_tolerance = 0.1  # 100ms tolerance
            
            if closest_distance < beat_tolerance:
                # On beat - check if strong or weak
                beat_in_measure = closest_beat_idx % beats_per_measure
                
                if beat_in_measure in strong_beats:
                    strong_beat_onsets += 1
                else:
                    weak_beat_onsets += 1
            else:
                # Off beat (syncopated)
                off_beat_onsets += 1
        
        total_onsets = strong_beat_onsets + weak_beat_onsets + off_beat_onsets
        
        if total_onsets == 0:
            return 0.0
        
        # Calculate syncopation as ratio of off-beat and weak-beat onsets
        syncopated_onsets = off_beat_onsets + weak_beat_onsets
        syncopation_ratio = syncopated_onsets / total_onsets
        
        return min(1.0, syncopation_ratio)
    
    def _calculate_pattern_regularity(self, onset_times: np.ndarray, 
                                    beat_times: np.ndarray) -> float:
        """Calculate pattern regularity.
        
        Args:
            onset_times: Onset times
            beat_times: Beat times
            
        Returns:
            Regularity score (0-1)
        """
        if len(onset_times) < 4 or len(beat_times) < 4:
            return 0.0
        
        # Calculate inter-onset intervals
        onset_intervals = np.diff(onset_times)
        
        if len(onset_intervals) == 0:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        mean_interval = np.mean(onset_intervals)
        std_interval = np.std(onset_intervals)
        
        if mean_interval == 0:
            return 0.0
        
        cv = std_interval / mean_interval
        
        # Convert to regularity score (lower CV = higher regularity)
        regularity = np.exp(-cv * 2)  # Exponential decay
        
        return min(1.0, regularity)
    
    def _calculate_subdivision_density(self, onset_times: np.ndarray, 
                                     beat_times: np.ndarray) -> float:
        """Calculate subdivision density.
        
        Args:
            onset_times: Onset times
            beat_times: Beat times
            
        Returns:
            Subdivision density (0-1)
        """
        if len(onset_times) == 0 or len(beat_times) < 2:
            return 0.0
        
        # Calculate average beat interval
        beat_intervals = np.diff(beat_times)
        avg_beat_interval = np.mean(beat_intervals)
        
        if avg_beat_interval == 0:
            return 0.0
        
        # Count onsets per beat
        onsets_per_beat = len(onset_times) / len(beat_times)
        
        # Normalize to 0-1 (assuming max 8 subdivisions per beat)
        max_subdivisions = 8
        density = min(1.0, onsets_per_beat / max_subdivisions)
        
        return density
    
    def detect_groove_type(self, onset_features: Dict[str, np.ndarray],
                          rhythm_patterns: Dict[str, Any]) -> Tuple[str, float]:
        """Detect groove type (straight, swing, etc.).
        
        Args:
            onset_features: Onset features
            rhythm_patterns: Rhythm pattern analysis
            
        Returns:
            (groove_type, confidence)
        """
        beat_times = onset_features.get('beat_times', np.array([]))
        onset_times = onset_features.get('onset_times', np.array([]))
        
        if len(beat_times) < 4 or len(onset_times) < 8:
            return 'straight', 0.0
        
        # Calculate swing ratio
        swing_ratio = self._calculate_swing_ratio(onset_times, beat_times)
        
        # Determine groove type based on swing ratio and other features
        if swing_ratio > 0.6:
            return 'swing', min(1.0, (swing_ratio - 0.6) / 0.3)
        elif swing_ratio > 0.55:
            return 'shuffle', min(1.0, (swing_ratio - 0.55) / 0.1)
        else:
            return 'straight', min(1.0, (0.55 - swing_ratio) / 0.05)
    
    def _calculate_swing_ratio(self, onset_times: np.ndarray, 
                              beat_times: np.ndarray) -> float:
        """Calculate swing ratio.
        
        Args:
            onset_times: Onset times
            beat_times: Beat times
            
        Returns:
            Swing ratio (0.5 = straight, >0.5 = swing)
        """
        if len(onset_times) < 4 or len(beat_times) < 4:
            return 0.5
        
        # Find eighth note subdivisions
        eighth_note_onsets = []
        
        for i in range(len(beat_times) - 1):
            beat_start = beat_times[i]
            beat_end = beat_times[i + 1]
            beat_duration = beat_end - beat_start
            
            # Find onsets within this beat
            beat_onsets = onset_times[(onset_times >= beat_start) & (onset_times < beat_end)]
            
            for onset in beat_onsets:
                # Calculate position within beat (0-1)
                position = (onset - beat_start) / beat_duration
                eighth_note_onsets.append(position)
        
        if len(eighth_note_onsets) < 4:
            return 0.5
        
        # Look for patterns around 0.33 and 0.67 (swing eighths)
        # vs 0.25 and 0.75 (straight eighths)
        
        # Count onsets near swing positions vs straight positions
        swing_positions = [0.33, 0.67]
        straight_positions = [0.25, 0.5, 0.75]
        
        swing_count = 0
        straight_count = 0
        tolerance = 0.1
        
        for onset_pos in eighth_note_onsets:
            # Check swing positions
            for swing_pos in swing_positions:
                if abs(onset_pos - swing_pos) < tolerance:
                    swing_count += 1
                    break
            else:
                # Check straight positions
                for straight_pos in straight_positions:
                    if abs(onset_pos - straight_pos) < tolerance:
                        straight_count += 1
                        break
        
        total_count = swing_count + straight_count
        
        if total_count == 0:
            return 0.5
        
        # Calculate swing ratio
        swing_ratio = 0.5 + (swing_count - straight_count) / (total_count * 2)
        
        return max(0.0, min(1.0, swing_ratio))
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete rhythm analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Complete rhythm analysis results
        """
        # Extract onset features
        onset_features = self.extract_onset_features(y, sr)
        
        # Detect time signature
        time_signature, ts_confidence = self.detect_time_signature(onset_features, sr)
        
        # Extract rhythm patterns
        rhythm_patterns = self.extract_rhythm_patterns(onset_features, time_signature)
        
        # Detect groove type
        groove_type, groove_confidence = self.detect_groove_type(onset_features, rhythm_patterns)
        
        # Calculate swing ratio
        swing_ratio = self._calculate_swing_ratio(
            onset_features.get('onset_times', np.array([])),
            onset_features.get('beat_times', np.array([]))
        )
        
        # Detect polyrhythm (simplified)
        polyrhythm_detected = self._detect_polyrhythm(onset_features)
        
        return {
            'time_signature': time_signature,
            'time_signature_confidence': ts_confidence,
            'groove_type': groove_type,
            'groove_confidence': groove_confidence,
            'rhythmic_complexity': rhythm_patterns['rhythmic_complexity'],
            'syncopation_level': rhythm_patterns['syncopation_level'],
            'pattern_regularity': rhythm_patterns['pattern_regularity'],
            'subdivision_density': rhythm_patterns['subdivision_density'],
            'swing_ratio': swing_ratio,
            'polyrhythm_detected': polyrhythm_detected,
            'onset_count': len(onset_features.get('onset_times', [])),
            'beat_count': len(onset_features.get('beat_times', []))
        }
    
    def _detect_polyrhythm(self, onset_features: Dict[str, np.ndarray]) -> bool:
        """Simple polyrhythm detection.
        
        Args:
            onset_features: Onset features
            
        Returns:
            True if polyrhythm is detected
        """
        onset_strength = onset_features.get('onset_strength', np.array([]))
        
        if len(onset_strength) < 100:  # Need sufficient data
            return False
        
        # Look for multiple periodic patterns in onset strength
        # This is a simplified approach
        
        # Calculate autocorrelation
        autocorr = correlate(onset_strength, onset_strength, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr, height=np.max(autocorr) * 0.3, distance=10)
        
        # If multiple strong peaks at different periods, might indicate polyrhythm
        if len(peaks) > 3:
            # Check if peaks are at non-harmonic intervals
            peak_ratios = []
            for i in range(len(peaks) - 1):
                ratio = peaks[i+1] / peaks[i]
                peak_ratios.append(ratio)
            
            # Look for non-integer ratios (indicating polyrhythm)
            non_integer_ratios = [r for r in peak_ratios if abs(r - round(r)) > 0.1]
            
            return len(non_integer_ratios) > 1
        
        return False