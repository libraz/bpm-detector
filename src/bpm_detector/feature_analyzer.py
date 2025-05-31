"""Audio feature analysis module for section classification."""

from typing import Any, Dict, List

import librosa
import numpy as np


class FeatureAnalyzer:
    """Analyzes audio features for section classification."""

    def __init__(self, hop_length: int = 512):
        """Initialize feature analyzer.

        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length

    def analyze_segment_characteristics_enhanced(
        self, segment: np.ndarray, sr: int, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """Enhanced segment analysis with additional features.

        Args:
            segment: Audio segment
            sr: Sample rate
            start_time: Start time of segment
            end_time: End time of segment

        Returns:
            Dictionary of enhanced characteristics
        """
        if len(segment) == 0:
            return self._get_default_characteristics()

        # Basic characteristics
        basic_chars = self.analyze_segment_characteristics(segment, sr)

        # Enhanced features
        enhanced_chars = basic_chars.copy()

        # Rhythm density
        enhanced_chars['rhythm_density'] = self._calculate_rhythm_density(segment, sr)

        # Melody jump rate
        enhanced_chars['melody_jump_rate'] = self._calculate_melody_jump_rate(
            segment, sr
        )

        # Vocal presence
        enhanced_chars['vocal_presence'] = self._detect_vocal_presence(segment, sr)

        # Spoken word detection
        enhanced_chars['is_spoken'] = self._detect_spoken_word(
            segment, sr, enhanced_chars['energy'], enhanced_chars['spectral_complexity']
        )

        # Voiced ratio
        enhanced_chars['voiced_ratio'] = self._calculate_voiced_ratio(enhanced_chars)

        # First peak time
        enhanced_chars['first_peak_time'] = self._find_first_peak_time(
            enhanced_chars, start_time
        )

        # Time information
        enhanced_chars['start_time'] = start_time
        enhanced_chars['end_time'] = end_time
        enhanced_chars['duration'] = end_time - start_time

        return enhanced_chars

    def analyze_segment_characteristics(
        self, segment: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Analyze basic characteristics of an audio segment.

        Args:
            segment: Audio segment
            sr: Sample rate

        Returns:
            Dictionary of characteristics
        """
        if len(segment) == 0:
            return self._get_default_characteristics()

        try:
            # Basic energy
            energy = np.mean(segment**2)

            # Spectral features
            stft = librosa.stft(segment, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Spectral centroid (brightness)
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(S=magnitude, sr=sr)
            )

            # Spectral rolloff
            spectral_rolloff = np.mean(
                librosa.feature.spectral_rolloff(S=magnitude, sr=sr)
            )

            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment))

            # Spectral complexity (spectral spread)
            spectral_bandwidth = np.mean(
                librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)
            )
            spectral_complexity = spectral_bandwidth / (spectral_centroid + 1e-8)

            # Harmonic content
            harmonic, percussive = librosa.effects.hpss(segment)
            harmonic_ratio = np.mean(harmonic**2) / (energy + 1e-8)

            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(
                y=segment, sr=sr, hop_length=self.hop_length
            )
            rhythmic_density = (
                len(beats) / (len(segment) / sr) if len(segment) > 0 else 0
            )

            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=segment, sr=sr, n_mfcc=13, hop_length=self.hop_length
            )
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)

            return {
                'energy': float(energy),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zcr),
                'spectral_complexity': float(spectral_complexity),
                'harmonic_content': float(harmonic_ratio),
                'rhythmic_density': float(rhythmic_density),
                'tempo': (
                    float(tempo)
                    if np.isscalar(tempo)
                    else float(tempo[0]) if len(tempo) > 0 else 120.0
                ),
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_var': mfcc_var.tolist(),
                'segment_length': len(segment),
            }

        except Exception as e:
            print(f"Warning: Error in segment analysis: {e}")
            return self._get_default_characteristics()

    def _calculate_rhythm_density(self, segment: np.ndarray, sr: int) -> float:
        """Calculate rhythm density of a segment.

        Args:
            segment: Audio segment
            sr: Sample rate

        Returns:
            Rhythm density value
        """
        if len(segment) == 0:
            return 0.0

        try:
            # Use onset detection for rhythm density
            onset_frames = librosa.onset.onset_detect(
                y=segment, sr=sr, hop_length=self.hop_length, units='frames'
            )

            # Calculate density as onsets per second
            duration = len(segment) / sr
            density = len(onset_frames) / duration if duration > 0 else 0.0

            # Normalize to 0-1 range (typical range is 0-10 onsets/sec)
            return min(1.0, density / 10.0)

        except Exception:
            return 0.0

    def _calculate_melody_jump_rate(self, segment: np.ndarray, sr: int) -> float:
        """Calculate melody jump rate (percentage of intervals > 2 semitones).

        Args:
            segment: Audio segment
            sr: Sample rate

        Returns:
            Melody jump rate (0-1)
        """
        if len(segment) == 0:
            return 0.0

        try:
            # Extract fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )

            # Filter out unvoiced frames
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) < 2:
                return 0.0

            # Convert to semitones
            semitones = 12 * np.log2(voiced_f0 / 440.0)  # A4 = 440Hz as reference

            # Calculate intervals
            intervals = np.abs(np.diff(semitones))

            # Count jumps > 2 semitones
            large_jumps = np.sum(intervals > 2.0)
            total_intervals = len(intervals)

            return float(large_jumps / total_intervals) if total_intervals > 0 else 0.0

        except Exception:
            return 0.0

    def _detect_spoken_word(
        self, segment: np.ndarray, sr: int, energy: float, complexity: float
    ) -> bool:
        """Detect if segment contains spoken word.

        Args:
            segment: Audio segment
            sr: Sample rate
            energy: Energy level
            complexity: Spectral complexity

        Returns:
            True if spoken word is detected
        """
        if len(segment) == 0:
            return False

        try:
            # Spoken word characteristics:
            # 1. Lower energy than singing
            # 2. More irregular rhythm
            # 3. Different spectral characteristics

            # Energy threshold (spoken word typically has lower energy)
            energy_threshold = 0.01  # Adjust based on normalization

            # Spectral characteristics
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=segment, sr=sr)
            )
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(segment))

            # Spoken word typically has:
            # - Moderate spectral centroid (1000-3000 Hz)
            # - Higher zero crossing rate than singing
            # - Lower overall energy

            is_spoken = (
                energy < energy_threshold * 2  # Lower energy
                and 1000 < spectral_centroid < 3000  # Speech frequency range
                and zero_crossing_rate > 0.1  # Higher ZCR
                and complexity < 0.5  # Lower spectral complexity
            )

            return is_spoken

        except Exception:
            return False

    def _detect_vocal_presence(self, segment: np.ndarray, sr: int) -> bool:
        """Detect vocal presence in segment.

        Args:
            segment: Audio segment
            sr: Sample rate

        Returns:
            True if vocals are detected
        """
        if len(segment) == 0:
            return False

        try:
            # Use harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(segment)

            # Calculate harmonic ratio
            harmonic_energy = np.mean(harmonic**2)
            total_energy = np.mean(segment**2)
            harmonic_ratio = harmonic_energy / (total_energy + 1e-8)

            # Spectral centroid (vocals typically in mid-frequency range)
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=segment, sr=sr)
            )

            # Vocal presence indicators
            has_vocals = (
                harmonic_ratio > 0.3  # Sufficient harmonic content
                and 1000 < spectral_centroid < 4000  # Vocal frequency range
            )

            return bool(has_vocals)

        except Exception:
            return False

    def _calculate_voiced_ratio(self, characteristics: Dict[str, Any]) -> float:
        """Calculate voiced ratio from characteristics.

        Args:
            characteristics: Audio characteristics

        Returns:
            Voiced ratio (0-1)
        """
        # Use harmonic content and spectral characteristics as proxy
        harmonic_content = characteristics.get('harmonic_content', 0.0)
        spectral_centroid = characteristics.get('spectral_centroid', 0.0)
        zcr = characteristics.get('zero_crossing_rate', 0.0)

        # Voiced sounds typically have:
        # - Higher harmonic content
        # - Moderate spectral centroid
        # - Lower zero crossing rate

        voiced_score = 0.0

        # Harmonic content contribution
        if harmonic_content > 0.3:
            voiced_score += 0.4
        elif harmonic_content > 0.1:
            voiced_score += 0.2

        # Spectral centroid contribution (vocal range)
        if 1000 <= spectral_centroid <= 3000:
            voiced_score += 0.3
        elif 500 <= spectral_centroid <= 5000:
            voiced_score += 0.1

        # Zero crossing rate contribution (lower for voiced)
        if zcr < 0.1:
            voiced_score += 0.3
        elif zcr < 0.2:
            voiced_score += 0.1

        return min(1.0, voiced_score)

    def _find_first_peak_time(
        self, characteristics: Dict[str, Any], start_time: float
    ) -> float:
        """Find first peak time in segment.

        Args:
            characteristics: Audio characteristics
            start_time: Start time of segment

        Returns:
            Time of first peak
        """
        # Simple heuristic: assume peak is at 1/3 of the segment for high energy sections
        energy = characteristics.get('energy', 0.0)
        duration = characteristics.get('duration', 0.0)

        if energy > 0.5 and duration > 0:
            # High energy sections typically have early peaks
            return start_time + duration * 0.3
        else:
            # Low energy sections have peaks later
            return start_time + duration * 0.5

    def _get_default_characteristics(self) -> Dict[str, Any]:
        """Get default characteristics for empty segments.

        Returns:
            Default characteristics dictionary
        """
        return {
            'energy': 0.0,
            'spectral_centroid': 1000.0,
            'spectral_rolloff': 2000.0,
            'zero_crossing_rate': 0.1,
            'spectral_complexity': 0.5,
            'harmonic_content': 0.0,
            'rhythmic_density': 0.0,
            'tempo': 120.0,
            'mfcc_mean': [0.0] * 13,
            'mfcc_var': [0.0] * 13,
            'segment_length': 0,
            'rhythm_density': 0.0,
            'melody_jump_rate': 0.0,
            'vocal_presence': False,
            'is_spoken': False,
            'voiced_ratio': 0.0,
            'first_peak_time': 0.0,
            'start_time': 0.0,
            'end_time': 0.0,
            'duration': 0.0,
        }
