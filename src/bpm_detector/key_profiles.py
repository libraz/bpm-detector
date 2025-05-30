"""Key profiles and hint mapping for key detection."""

import numpy as np
from typing import Dict, Tuple, List
from .music_theory import NOTE_NAMES


class _Constants:
    """Key detection constants with clear documentation."""
    SEP_BOOST = 0.80              # Confidence boost factor (0-1) for separation between top candidates
    REL_SWITCH_THRESH = 0.10      # Correlation difference threshold (0-1) for relative key switching
    JPOP_CONF_THRESH = 0.25       # Minimum confidence (0-1) for J-Pop specific detection
    MIN_CONFIDENCE = 0.02         # Global minimum confidence (0-1) to avoid 'None' results (lowered for better detection)
    PATTERN_THRESH = 0.15         # Pattern strength threshold (0-1) for G# minor specific detection
    CHORD_WEIGHT_THRESH = 0.50    # Minimum confidence (0-1) for chord-driven key re-estimation
    
    # Profile enhancement constants
    TONIC_BOOST = 1.1             # Boost factor for tonic in minor profile
    MINOR_THIRD_BOOST = 1.3       # Boost factor for minor third
    FIFTH_BOOST = 1.2             # Boost factor for fifth
    MINOR_SEVENTH_BOOST = 1.2     # Boost factor for minor seventh


class KeyProfileBuilder:
    """Builds key profiles for key detection."""
    
    @staticmethod
    def build_profiles(
        tonic_boost: float = None,
        minor_third_boost: float = None,
        fifth_boost: float = None,
        minor_seventh_boost: float = None,
        profile_type: str = 'krumhansl'
    ) -> List[np.ndarray]:
        """Build enhanced Krumhansl-Schmuckler key profiles for modern pop music.
        
        Args:
            tonic_boost: Boost factor for tonic (default from constants)
            minor_third_boost: Boost factor for minor third (default from constants)
            fifth_boost: Boost factor for fifth (default from constants)
            minor_seventh_boost: Boost factor for minor seventh (default from constants)
            profile_type: Type of profile to build
        
        Returns:
            List of 24 key profiles (12 major + 12 minor)
        """
        # Use constants as defaults
        tonic_boost = tonic_boost or _Constants.TONIC_BOOST
        minor_third_boost = minor_third_boost or _Constants.MINOR_THIRD_BOOST
        fifth_boost = fifth_boost or _Constants.FIFTH_BOOST
        minor_seventh_boost = minor_seventh_boost or _Constants.MINOR_SEVENTH_BOOST
        
        # Enhanced Krumhansl-Schmuckler key profiles for modern pop music
        # Adjusted for better discrimination between keys
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Enhance tonic prominence for better key detection
        major_profile[0] *= 1.5  # Boost tonic
        major_profile[4] *= 1.3  # Boost major third
        major_profile[7] *= 1.2  # Boost perfect fifth
        
        minor_profile[0] *= 1.5  # Boost tonic
        minor_profile[3] *= 1.3  # Boost minor third
        minor_profile[7] *= 1.2  # Boost perfect fifth
        
        # Enhanced minor profile using parameterized boosts
        boost_factors = np.array([tonic_boost-1, 0, 0, minor_third_boost-1, 0, 0, 0, fifth_boost-1, 0, 0, minor_seventh_boost-1, 0])
        minor_profile = minor_profile * (1 + boost_factors)
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Create list of 24 profiles for test compatibility
        profiles = []
        for i in range(12):
            # Add major profile for each key (copy() でブーストにじみを防ぐ)
            profiles.append(np.roll(major_profile, i).copy())
            # Add minor profile for each key (copy() でブーストにじみを防ぐ)
            profiles.append(np.roll(minor_profile, i).copy())
        
        return profiles


class KeyHintMapper:
    """Handles external key hint mapping."""
    
    @staticmethod
    def build_hint_mapping() -> Dict[str, Tuple[str, str]]:
        """Build external hint mapping for normalized comparison.
        
        Returns:
            Dictionary mapping normalized hint strings to (key, mode) tuples
        """
        hint_map = {}
        
        # Also include enharmonic equivalents expected by tests
        all_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        
        for note in all_keys:
            note_lower = note.lower()
            # Major keys
            hint_map[note] = (note, "Major")
            hint_map[note_lower] = (note, "Major")
            hint_map[f"{note_lower}major"] = (note, "Major")
            hint_map[f"{note_lower}maj"] = (note, "Major")
            
            # Minor keys
            hint_map[f"{note_lower}minor"] = (note, "Minor")
            hint_map[f"{note_lower}min"] = (note, "Minor")
            hint_map[f"{note_lower}m"] = (note, "Minor")
            hint_map[f"{note_lower}-minor"] = (note, "Minor")
            hint_map[f"{note_lower}-min"] = (note, "Minor")
            hint_map[f"{note_lower}-m"] = (note, "Minor")
        
        # Add standard NOTE_NAMES as well
        for note in NOTE_NAMES:
            if note not in all_keys:
                note_lower = note.lower()
                # Major keys
                hint_map[note] = (note, "Major")
                hint_map[note_lower] = (note, "Major")
                hint_map[f"{note_lower}major"] = (note, "Major")
                hint_map[f"{note_lower}maj"] = (note, "Major")
                
                # Minor keys
                hint_map[f"{note_lower}minor"] = (note, "Minor")
                hint_map[f"{note_lower}min"] = (note, "Minor")
                hint_map[f"{note_lower}m"] = (note, "Minor")
                hint_map[f"{note_lower}-minor"] = (note, "Minor")
                hint_map[f"{note_lower}-min"] = (note, "Minor")
                hint_map[f"{note_lower}-m"] = (note, "Minor")
        
        return hint_map
    
    @staticmethod
    def apply_external_key_hint(
        hint: str, 
        current_key: str, 
        current_mode: str,
        current_confidence: float,
        hint_map: Dict[str, Tuple[str, str]]
    ) -> Tuple[str, str, float]:
        """Apply external key hint with normalized comparison.
        
        Args:
            hint: External key hint string
            current_key: Currently detected key
            current_mode: Currently detected mode
            current_confidence: Current confidence level
            hint_map: Hint mapping dictionary
            
        Returns:
            (final_key, final_mode, final_confidence)
        """
        # Normalize: case, whitespace, and symbols
        norm_hint = hint.lower().replace(" ", "").replace("♭", "b").replace("♯", "#")
        
        # Check against predefined hint mapping
        if norm_hint in hint_map:
            key, mode = hint_map[norm_hint]
            return key, mode, max(current_confidence, 0.8)
        
        # General pattern analysis for other keys
        for note in NOTE_NAMES:
            note_lower = note.lower()
            major_patterns = {f"{note_lower}major", f"{note_lower}maj"}
            minor_patterns = {f"{note_lower}minor", f"{note_lower}min"}
            
            # Handle ambiguous 'm' suffix carefully
            if norm_hint == f"{note_lower}m":
                # Default to minor for single 'm' suffix unless it's a known major pattern
                if note_lower not in {"c", "d#", "g#"}:  # Avoid conflicts with predefined patterns
                    return note, "Minor", max(current_confidence, 0.7)
            
            if norm_hint in major_patterns:
                return note, "Major", max(current_confidence, 0.7)
            elif norm_hint in minor_patterns:
                return note, "Minor", max(current_confidence, 0.7)
        
        # If no match found, maintain current values
        return current_key, current_mode, current_confidence