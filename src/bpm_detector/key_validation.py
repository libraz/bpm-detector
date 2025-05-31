"""Key validation and specialized detection for key detection."""

import numpy as np
from typing import List, Tuple
from .music_theory import NOTE_NAMES
from .key_profiles import _Constants


class KeyValidator:
    """Validates and refines key detection results."""

    @staticmethod
    def validate_relative_keys(
        key_note: str,
        mode: str,
        confidence: float,
        chroma_mean: np.ndarray,
        correlations: List[float] = None,
        key_names: List[str] = None,
    ) -> Tuple[str, str, float]:
        """Validate and potentially correct key detection using relative major/minor analysis.

        Args:
            key_note: Detected key note
            mode: Detected mode
            confidence: Initial confidence
            chroma_mean: Average chroma vector
            correlations: All key correlations
            key_names: All key names

        Returns:
            (final_key, final_mode, final_confidence)
        """
        try:
            key_index = NOTE_NAMES.index(key_note)
        except ValueError:
            return key_note, mode, confidence

        # If correlations and key_names are not provided, just return original values
        if correlations is None or key_names is None:
            return key_note, mode, confidence

        # Find relative major/minor
        if mode == 'Major':
            # Relative minor is 3 semitones down (minor third)
            relative_minor_index = (key_index - 3) % 12
            relative_minor_name = NOTE_NAMES[relative_minor_index]
            relative_key_name = f"{relative_minor_name} Minor"
        else:  # Minor
            # Relative major is 3 semitones up (minor third)
            relative_major_index = (key_index + 3) % 12
            relative_major_name = NOTE_NAMES[relative_major_index]
            relative_key_name = f"{relative_major_name} Major"

        # Find correlation for relative key
        try:
            relative_key_idx = key_names.index(relative_key_name)
            relative_correlation = correlations[relative_key_idx]
            current_correlation = correlations[key_names.index(f"{key_note} {mode}")]
        except (ValueError, IndexError):
            return key_note, mode, confidence

        # If relative key has significantly higher correlation, consider switching
        correlation_diff = relative_correlation - current_correlation

        # Relaxed threshold for relative major/minor switching
        if (
            abs(correlation_diff) < _Constants.REL_SWITCH_THRESH
        ):  # Close correlations, need deeper analysis
            # Analyze chord progression tendencies
            major_tendency = KeyValidator._analyze_major_tendency(
                chroma_mean, key_index
            )
            minor_tendency = KeyValidator._analyze_minor_tendency(
                chroma_mean, key_index
            )

            if mode == 'Major' and minor_tendency > major_tendency + 0.2:
                # Switch to relative minor
                return relative_minor_name, 'Minor', confidence * 0.9
            elif mode == 'Minor' and major_tendency > minor_tendency + 0.2:
                # Switch to relative major
                return relative_major_name, 'Major', confidence * 1.1

        elif correlation_diff > _Constants.REL_SWITCH_THRESH:  # Relative key stronger
            if mode == 'Major':
                return relative_minor_name, 'Minor', confidence * 1.1
            else:
                return relative_major_name, 'Major', confidence * 1.1

        return key_note, mode, confidence

    @staticmethod
    def _analyze_major_tendency(chroma_mean: np.ndarray, key_index: int) -> float:
        """Analyze tendency towards major tonality."""
        # Major chord tones: I, III, V (root, major third, fifth)
        major_third = (key_index + 4) % 12
        fifth = (key_index + 7) % 12

        major_strength = (
            chroma_mean[key_index]
            + chroma_mean[major_third] * 1.2  # Major third is characteristic
            + chroma_mean[fifth]
        ) / 3.2

        return major_strength

    @staticmethod
    def _analyze_minor_tendency(chroma_mean: np.ndarray, key_index: int) -> float:
        """Analyze tendency towards minor tonality."""
        # Minor chord tones: i, ♭III, V (root, minor third, fifth)
        minor_third = (key_index + 3) % 12
        fifth = (key_index + 7) % 12

        minor_strength = (
            chroma_mean[key_index]
            + chroma_mean[minor_third] * 1.2  # Minor third is characteristic
            + chroma_mean[fifth]
        ) / 3.2

        return minor_strength


class JPOPKeyDetector:
    """Specialized detector for J-Pop keys."""

    @staticmethod
    def detect_jpop_keys(
        chroma_mean: np.ndarray,
        correlations: List[float],
        enable_jpop: bool = True,
        key_names: List[str] = None,
    ) -> Tuple[str, str, float]:
        """Special detection for common J-Pop keys like G# minor."""

        # Common J-Pop keys to check specifically
        jpop_keys = [
            ('G#', 'Minor'),  # Very common in J-Pop
            ('D#', 'Minor'),  # Also very common
            ('F#', 'Minor'),  # Common
            ('C#', 'Minor'),  # Common
            ('B', 'Major'),  # Relative major of G# minor
            ('F#', 'Major'),  # Relative major of D# minor
        ]

        best_key = 'None'
        best_mode = 'Unknown'
        best_strength = 0.0

        for key_note, mode in jpop_keys:
            try:
                key_index = NOTE_NAMES.index(key_note)

                # Calculate specific strength for this key
                if mode == 'Minor':
                    # Check for characteristic minor chord patterns
                    tonic = chroma_mean[key_index]
                    minor_third = chroma_mean[(key_index + 3) % 12]
                    fifth = chroma_mean[(key_index + 7) % 12]
                    # G# minor specific pattern: G#m - F# - E - C#m
                    if key_note == 'G#':
                        f_sharp = chroma_mean[6]  # F#
                        e = chroma_mean[4]  # E
                        c_sharp = chroma_mean[1]  # C#

                        # Check for G#m - F# - E - C#m pattern
                        pattern_strength = (
                            tonic * 1.5 + f_sharp * 1.2 + e * 1.1 + c_sharp * 1.0
                        ) / 4.8

                        # Boost if this pattern is strong (lowered threshold)
                        if pattern_strength > _Constants.PATTERN_THRESH:
                            strength = pattern_strength * 1.5  # Increased boost
                        else:
                            strength = (tonic + minor_third * 1.2 + fifth) / 3.2

                        # Additional boost for G# minor specifically
                        strength *= 1.2
                    else:
                        # Standard minor key strength
                        strength = (tonic + minor_third * 1.2 + fifth) / 3.2

                else:  # Major
                    tonic = chroma_mean[key_index]
                    major_third = chroma_mean[(key_index + 4) % 12]
                    fifth = chroma_mean[(key_index + 7) % 12]
                    strength = (tonic + major_third * 1.1 + fifth) / 3.1

                # Check against existing correlation
                key_name = f"{key_note} {mode}"
                if key_names is not None and key_name in key_names:
                    correlation_idx = key_names.index(key_name)
                    correlation_strength = correlations[correlation_idx]

                    # Combine pattern strength with correlation
                    combined_strength = (strength + correlation_strength) / 2.0
                else:
                    combined_strength = strength

                if combined_strength > best_strength:
                    best_strength = combined_strength
                    best_key = key_note
                    best_mode = mode

            except (ValueError, IndexError):
                continue

        return best_key, best_mode, best_strength
