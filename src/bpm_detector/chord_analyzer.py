"""Chord progression analysis module."""

from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from .music_theory import NOTE_NAMES


class ChordProgressionAnalyzer:
    """Analyzes chord progressions and harmonic features."""

    # Chord templates including triads, 7th and sus4 chords for all roots
    CHORD_TEMPLATES: Dict[str, List[int]] = {}

    for i, note in enumerate(NOTE_NAMES):
        # Major triad
        vec = [0] * 12
        vec[i] = 1
        vec[(i + 4) % 12] = 1
        vec[(i + 7) % 12] = 1
        CHORD_TEMPLATES[note] = vec

        # Minor triad
        vec = [0] * 12
        vec[i] = 1
        vec[(i + 3) % 12] = 1
        vec[(i + 7) % 12] = 1
        CHORD_TEMPLATES[f"{note}m"] = vec

        # Dominant 7th
        vec = [0] * 12
        vec[i] = 1
        vec[(i + 4) % 12] = 1
        vec[(i + 7) % 12] = 1
        vec[(i + 10) % 12] = 1
        CHORD_TEMPLATES[f"{note}7"] = vec

        # Major 7th (simplified as triad for better detection)
        vec = [0] * 12
        vec[i] = 1
        vec[(i + 4) % 12] = 1
        vec[(i + 7) % 12] = 1
        CHORD_TEMPLATES[f"{note}maj7_tri"] = vec

        # Sus4
        vec = [0] * 12
        vec[i] = 1
        vec[(i + 5) % 12] = 1
        vec[(i + 7) % 12] = 1
        CHORD_TEMPLATES[f"{note}sus4"] = vec

    # Roman numeral mapping for functional analysis
    FUNCTIONAL_MAPPING = {
        'major': {
            0: 'I',
            1: 'bII',
            2: 'II',
            3: 'bIII',
            4: 'III',
            5: 'IV',
            6: 'bV',
            7: 'V',
            8: 'bVI',
            9: 'VI',
            10: 'bVII',
            11: 'VII',
        },
        'minor': {
            0: 'i',
            1: 'bII',
            2: 'II',
            3: 'bIII',
            4: 'III',
            5: 'iv',
            6: 'bV',
            7: 'V',
            8: 'bVI',
            9: 'VI',
            10: 'bVII',
            11: 'VII',
        },
    }

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize chord analyzer.

        Args:
            hop_length: Hop length for analysis
            n_fft: FFT size for spectral analysis
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_chroma_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract high-resolution chroma features with noise reduction.

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Chroma features matrix (12 x time_frames)
        """
        # Apply high-pass filter to remove low-frequency noise
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        high_cutoff = 80.0  # Remove frequencies below 80Hz
        high_normal = high_cutoff / nyquist
        b, a = butter(4, high_normal, btype='high', analog=False)
        y_filtered = filtfilt(b, a, y)

        # Apply harmonic-percussive separation for cleaner harmonic content
        y_harmonic, _ = librosa.effects.hpss(y_filtered, margin=3.0)

        # Use STFT-based chroma with larger FFT for tension chord detection
        stft = librosa.stft(y_harmonic, hop_length=self.hop_length, n_fft=self.n_fft)
        chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr, hop_length=self.hop_length, n_chroma=12, norm=2)

        # Apply 2-second moving window average for stability
        window_frames = int(2.0 * sr / self.hop_length)  # 2 seconds
        if window_frames > 1:
            chroma = self._apply_moving_average(chroma, window_frames)

        return chroma

    def _apply_moving_average(self, chroma: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average to chroma features for stability.

        Args:
            chroma: Input chroma matrix
            window_size: Size of moving window in frames

        Returns:
            Smoothed chroma matrix
        """
        smoothed = np.zeros_like(chroma)
        half_window = window_size // 2

        for i in range(chroma.shape[1]):
            start = max(0, i - half_window)
            end = min(chroma.shape[1], i + half_window + 1)
            smoothed[:, i] = np.mean(chroma[:, start:end], axis=1)

        return np.asarray(smoothed)

    def detect_chords(self, chroma: np.ndarray, bpm: float = 130.0) -> List[Tuple[str, float, int, int]]:
        """Detect chords from chroma features with dynamic window sizing.

        Args:
            chroma: Chroma features matrix
            bpm: BPM for dynamic window calculation

        Returns:
            List of (chord_name, confidence, start_frame, end_frame)
        """
        chords = []
        n_frames = chroma.shape[1]

        # ------------------------------------------------------------------
        # For Verse/Chorus that repeat every 4 bars (≈ 7.3s),
        # window size = 2 bars (1/2) is appropriate for chord detection
        # ------------------------------------------------------------------
        bars_per_window = 2
        window_duration = (bars_per_window * 4 * 60.0) / bpm  # 2 bars in seconds
        window_duration = max(1.0, window_duration)  # At least 1 second
        window_size = max(1, int(22050 * window_duration / self.hop_length))
        step_size = max(1, window_size // 4)  # Overlap windows for better detection

        detected_chords = []

        for i in range(0, n_frames - window_size + 1, step_size):
            end_frame = min(i + window_size, n_frames)

            # Average chroma over the window
            window_chroma = np.mean(chroma[:, i:end_frame], axis=1)

            # Find best matching chord
            best_chord, confidence = self._match_chord_template(window_chroma)

            # Adaptive confidence threshold based on signal strength
            signal_strength: float = np.max(window_chroma)
            adaptive_threshold = max(0.3, 0.55 - (1.0 - signal_strength) * 0.2)  # Lowered threshold

            if confidence > adaptive_threshold:
                detected_chords.append((best_chord, confidence, i, end_frame))

        # Very relaxed merging to preserve chord diversity
        chords = self._merge_consecutive_chords(detected_chords, 0.50)  # Very relaxed threshold

        return chords

    def _match_chord_template(self, chroma_frame: np.ndarray) -> Tuple[str, float]:
        """Match chroma frame to chord templates with improved triad/tetrad separation.

        Args:
            chroma_frame: Single chroma vector

        Returns:
            (best_chord_name, confidence)
        """
        # Enhanced chord detection with triad/tetrad prioritization
        best_chord = 'N'  # No chord
        best_score = 0.0

        # Find the top 4 strongest notes for analysis
        top_4_indices = np.argsort(chroma_frame)[-4:]
        top_4_strengths = chroma_frame[top_4_indices]

        # Only proceed if we have significant energy in top notes
        if np.max(top_4_strengths) < 0.1:
            return 'N', 0.0

        # Separate triads and tetrads for better detection
        triad_candidates = []
        tetrad_candidates = []

        # Template matching with triad/tetrad separation
        for chord_name, template in self.CHORD_TEMPLATES.items():
            template = np.array(template, dtype=np.float32)

            # Improved correlation calculation (cosine similarity)
            chroma_norm = np.linalg.norm(chroma_frame)
            template_norm = np.linalg.norm(template)

            if chroma_norm > 1e-8 and template_norm > 1e-8:
                correlation = np.dot(chroma_frame, template) / (chroma_norm * template_norm)
            else:
                correlation = 0.0

            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0

            # Categorize by chord type
            if '7' in chord_name or 'sus' in chord_name:
                tetrad_candidates.append((chord_name, correlation))
            else:
                triad_candidates.append((chord_name, correlation))

        # Prioritize triads unless tetrad has significantly higher confidence
        best_triad = max(triad_candidates, key=lambda x: x[1]) if triad_candidates else ('N', 0.0)
        best_tetrad = max(tetrad_candidates, key=lambda x: x[1]) if tetrad_candidates else ('N', 0.0)

        # Prefer triads over 7th chords unless 7th chord is significantly better
        # Use tetrad only if it's significantly better than triad (threshold: 0.15)
        if best_tetrad[1] > best_triad[1] + 0.15:  # Higher threshold to prefer triads
            best_chord, best_score = best_tetrad
        else:
            best_chord, best_score = best_triad

        # Also try cluster-based detection for comparison
        cluster_chord = self._identify_chord_from_cluster(top_4_indices[:3], top_4_strengths[:3])
        if cluster_chord is not None:
            cluster_score = np.mean(top_4_strengths[:3])
            # Only use cluster result if it's competitive
            if cluster_score > best_score * 0.8:
                best_score = float(cluster_score)
                best_chord = cluster_chord

        return best_chord, max(0.0, best_score)

    def _identify_chord_from_cluster(self, note_indices: np.ndarray, strengths: np.ndarray) -> Optional[str]:
        """Identify chord from top 3 notes using music theory.

        Args:
            note_indices: Indices of top 3 notes
            strengths: Strengths of top 3 notes

        Returns:
            Chord name or None
        """
        if len(note_indices) < 3:
            return None

        # Sort by strength (strongest first)
        sorted_indices = note_indices[np.argsort(strengths)[::-1]]

        # Try different root assumptions
        for root_idx in sorted_indices:
            # Check for major triad (root, major third, fifth)
            major_third = (root_idx + 4) % 12
            fifth = (root_idx + 7) % 12

            if major_third in note_indices and fifth in note_indices:
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                return str(note_names[root_idx])

            # Check for minor triad (root, minor third, fifth)
            minor_third = (root_idx + 3) % 12

            if minor_third in note_indices and fifth in note_indices:
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                return str(note_names[root_idx]) + 'm'

        return None

    def _merge_consecutive_chords(
        self, chords: List[Tuple[str, float, int, int]], merge_threshold: float = 0.6
    ) -> List[Tuple[str, float, int, int]]:
        """Merge consecutive identical chords with configurable threshold.

        Args:
            chords: List of detected chords
            merge_threshold: Confidence threshold for merging (higher = less merging)

        Returns:
            Merged chord list
        """
        if not chords:
            return []

        merged = []
        current_chord = chords[0]

        for i in range(1, len(chords)):
            next_chord = chords[i]

            # More restrictive merging: only merge if confidence is high enough
            should_merge = (
                current_chord[0] == next_chord[0]
                and next_chord[2] <= current_chord[3] + self.hop_length
                and min(current_chord[1], next_chord[1]) > merge_threshold
            )

            if should_merge:
                # Extend current chord
                current_chord = (
                    current_chord[0],
                    max(current_chord[1], next_chord[1]),  # Take higher confidence
                    current_chord[2],
                    max(current_chord[3], next_chord[3]),  # Extend end frame
                )
            else:
                # Different chord or low confidence, add current and start new
                merged.append(current_chord)
                current_chord = next_chord

        # Add the last chord
        merged.append(current_chord)

        return merged

    def analyze_progression(self, chords: List[Tuple[str, float, int, int]]) -> Dict[str, Any]:
        """Analyze chord progression patterns.

        Args:
            chords: List of detected chords

        Returns:
            Dictionary containing progression analysis
        """
        if not chords:
            return {
                'main_progression': [],
                'progression_pattern': '',
                'harmonic_rhythm': 0.0,
                'chord_complexity': 0.0,
                'unique_chords': 0,
                'chord_changes': 0,
            }

        # Extract chord names
        chord_names = [chord[0] for chord in chords if chord[0] != 'N']

        if not chord_names:
            return {
                'main_progression': [],
                'progression_pattern': '',
                'harmonic_rhythm': 0.0,
                'chord_complexity': 0.0,
                'unique_chords': 0,
                'chord_changes': 0,
            }

        # ------------------------------------------------------------------
        # For 4-bar progression fixed songs, prioritize 4-degree progression patterns
        # ------------------------------------------------------------------
        main_progression = self._find_main_progression(chord_names, pattern_length=4)

        # Calculate harmonic rhythm (chord changes per second)
        total_duration = sum(chord[3] - chord[2] for chord in chords)
        harmonic_rhythm = (
            len([c for c in chords if c[0] != 'N']) / (total_duration * self.hop_length / 22050)
            if total_duration > 0
            else 0
        )

        # Calculate chord complexity
        unique_chords = len(set(chord_names))
        chord_complexity = min(1.0, unique_chords / 12.0)  # Normalize to 0-1

        # Count chord changes
        chord_changes = len([i for i in range(1, len(chord_names)) if chord_names[i] != chord_names[i - 1]])

        return {
            'main_progression': main_progression,
            'progression_pattern': ' - '.join(main_progression),
            'harmonic_rhythm': harmonic_rhythm,
            'chord_complexity': chord_complexity,
            'unique_chords': unique_chords,
            'chord_changes': chord_changes,
        }

    def _find_main_progression(self, chord_names: List[str], pattern_length: int = 4) -> List[str]:
        """Find the most common chord progression pattern with improved diversity.

        Args:
            chord_names: List of chord names
            pattern_length: Length of pattern to search for

        Returns:
            Most common progression pattern
        """
        if len(chord_names) < pattern_length:
            return chord_names[:4] if len(chord_names) >= 4 else chord_names

        # Count all possible patterns of given length
        pattern_counts: Dict[Tuple[str, ...], int] = {}

        for i in range(len(chord_names) - pattern_length + 1):
            pattern = tuple(chord_names[i : i + pattern_length])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        if pattern_counts:
            # Strongly prioritize diverse patterns over repetitive ones
            diverse_patterns = {}
            for pattern, count in pattern_counts.items():
                unique_chords = len(set(pattern))
                # Moderate preference for diverse patterns
                if unique_chords >= 3:
                    diversity_bonus = unique_chords * 2  # Moderate diversity bonus
                    diverse_patterns[pattern] = count + diversity_bonus
                elif unique_chords == 2:
                    diversity_bonus = unique_chords * 2  # Equal bonus
                    diverse_patterns[pattern] = count + diversity_bonus
                elif unique_chords == 1:
                    # Heavily penalize single-chord patterns
                    diverse_patterns[pattern] = int(count * 0.01)

            # Use diverse patterns if available, otherwise fall back to original
            if diverse_patterns:
                most_common = max(diverse_patterns.items(), key=lambda x: x[1])
                return list(most_common[0])
            else:
                most_common = max(pattern_counts.items(), key=lambda x: x[1])
                return list(most_common[0])

        return chord_names[:pattern_length]

    def functional_analysis(self, chords: List[str], key: str) -> List[str]:
        """Perform functional harmonic analysis.

        Args:
            chords: List of chord names
            key: Key of the song (e.g., 'C Major', 'A Minor')

        Returns:
            List of roman numeral analysis
        """
        if not chords or not key:
            return []

        # Parse key
        key_parts = key.split()
        if len(key_parts) != 2:
            return []

        root_note = key_parts[0]
        mode = key_parts[1].lower()

        # Convert root note to semitone offset
        note_to_semitone = {
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
            'A': 9,
            'A#': 10,
            'B': 11,
        }

        if root_note not in note_to_semitone:
            return []

        key_root = note_to_semitone[root_note]

        # Get appropriate mapping
        if mode == 'major':
            mapping = self.FUNCTIONAL_MAPPING['major']
        elif mode == 'minor':
            mapping = self.FUNCTIONAL_MAPPING['minor']
        else:
            return []

        # Analyze each chord
        roman_numerals = []

        for chord in chords:
            if chord == 'N':
                roman_numerals.append('N')
                continue

            # Extract root note from chord
            if len(chord) > 1 and chord[1] == '#':
                chord_root = chord[:2]
            else:
                chord_root = chord[0]

            if chord_root in note_to_semitone:
                # Calculate interval from key root
                chord_semitone = note_to_semitone[chord_root]
                interval = (chord_semitone - key_root) % 12

                # Get roman numeral
                roman = mapping.get(interval, '?')

                # Add minor indication if chord is minor
                if 'm' in chord and mode == 'major':
                    roman = roman.lower()
                elif 'm' not in chord and mode == 'minor' and roman != '?':
                    roman = roman.upper()

                roman_numerals.append(roman)
            else:
                roman_numerals.append('?')

        return roman_numerals

    def detect_modulations(
        self, chords: List[Tuple[str, float, int, int]], original_key: str, sr: int
    ) -> List[Dict[str, Any]]:
        """Detect key modulations in the progression.

        Args:
            chords: List of detected chords with timing
            original_key: Original key of the song
            sr: Sample rate

        Returns:
            List of detected modulations
        """
        modulations: List[Dict[str, Any]] = []

        if len(chords) < 8:  # Need sufficient chords to detect modulation
            return modulations

        # Analyze chord progressions in windows
        window_size = 8  # Analyze 8 chords at a time

        for i in range(0, len(chords) - window_size + 1, window_size // 2):
            window_chords = chords[i : i + window_size]
            chord_names = [c[0] for c in window_chords if c[0] != 'N']

            if len(chord_names) < 4:
                continue

            # Try to detect key of this window
            detected_key = self._detect_local_key(chord_names)

            if detected_key and detected_key != original_key:
                # Calculate time of modulation
                time_seconds = window_chords[0][2] * self.hop_length / sr

                modulations.append(
                    {
                        'time': time_seconds,
                        'from_key': original_key,
                        'to_key': detected_key,
                        'confidence': 0.7,  # Placeholder confidence
                    }
                )

                original_key = detected_key  # Update for next detection

        return modulations

    def _detect_local_key(self, chord_names: List[str]) -> Optional[str]:
        """Detect the key of a local chord progression.

        Args:
            chord_names: List of chord names

        Returns:
            Detected key or None
        """
        # Simple key detection based on chord frequency
        # This is a simplified approach - could be improved with more sophisticated algorithms

        major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_key: Optional[str] = None
        best_score: float = 0.0

        # Test each possible key
        for key_root in major_keys:
            # Test major key
            score = self._score_key_fit(chord_names, key_root, 'major')
            if score > best_score:
                best_score = float(score)
                best_key = f"{key_root} Major"

            # Test minor key
            score = self._score_key_fit(chord_names, key_root, 'minor')
            if score > best_score:
                best_score = float(score)
                best_key = f"{key_root} Minor"

        return best_key if best_score > 0.5 else None

    def _score_key_fit(self, chord_names: List[str], key_root: str, mode: str) -> float:
        """Score how well chords fit a given key.

        Args:
            chord_names: List of chord names
            key_root: Root note of the key
            mode: 'major' or 'minor'

        Returns:
            Fit score (0-1)
        """
        if not chord_names:
            return 0.0

        # Define scale degrees for major and minor keys
        if mode == 'major':
            scale_chords = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
        else:
            scale_chords = ['i', 'ii°', 'III', 'iv', 'V', 'VI', 'VII']

        # Convert chords to roman numerals for this key
        roman_numerals = self.functional_analysis(chord_names, f"{key_root} {mode.title()}")

        # Count how many chords fit the key
        fitting_chords = 0
        for roman in roman_numerals:
            if roman in scale_chords or roman.lower() in [c.lower() for c in scale_chords]:
                fitting_chords += 1

        return fitting_chords / len(chord_names) if chord_names else 0.0

    def analyze(self, y: np.ndarray, sr: int, key: Optional[str] = None, bpm: float = 130.0) -> Dict[str, Any]:
        """Perform complete chord progression analysis.

        Args:
            y: Audio signal
            sr: Sample rate
            key: Optional key information
            bpm: BPM for dynamic window calculation

        Returns:
            Complete chord analysis results
        """
        # Extract chroma features
        chroma = self.extract_chroma_features(y, sr)

        # Detect chords with dynamic window sizing
        chords = self.detect_chords(chroma, bpm)

        # Analyze progression
        progression_analysis = self.analyze_progression(chords)

        # Functional analysis if key is provided
        functional_analysis = []
        if key and chords:
            chord_names = [c[0] for c in chords if c[0] != 'N']
            functional_analysis = self.functional_analysis(chord_names, key)

        # Detect modulations
        modulations = []
        if key and len(chords) > 8:
            modulations = self.detect_modulations(chords, key, sr)

        # Calculate substitute chord ratio
        chord_names = [c[0] for c in chords if c[0] != 'N']
        substitute_ratio = self._calculate_substitute_ratio(chord_names, key) if key else 0.0

        return {
            'chords': chords,
            'main_progression': progression_analysis['main_progression'],
            'progression_pattern': progression_analysis['progression_pattern'],
            'harmonic_rhythm': progression_analysis['harmonic_rhythm'],
            'chord_complexity': progression_analysis['chord_complexity'],
            'unique_chords': progression_analysis['unique_chords'],
            'chord_changes': progression_analysis['chord_changes'],
            'functional_analysis': functional_analysis,
            'modulations': modulations,
            'substitute_chords_ratio': substitute_ratio,
        }

    def _calculate_substitute_ratio(self, chord_names: List[str], key: str) -> float:
        """Calculate the ratio of substitute/extended chords.

        Args:
            chord_names: List of chord names
            key: Key of the song

        Returns:
            Ratio of substitute chords (0-1)
        """
        if not chord_names or not key:
            return 0.0

        # Simple heuristic: count chords with extensions or alterations
        substitute_count = 0

        for chord in chord_names:
            # Look for chord extensions/alterations
            if any(ext in chord for ext in ['7', '9', '11', '13', 'sus', 'add', 'dim', 'aug']):
                substitute_count += 1

        return substitute_count / len(chord_names) if chord_names else 0.0
