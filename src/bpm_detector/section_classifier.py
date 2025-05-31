"""Section classification module for musical structure analysis."""

from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from .context_analyzer import ContextAnalyzer
from .feature_analyzer import FeatureAnalyzer


class SectionClassifier:
    """Classifies musical sections based on audio characteristics."""

    def __init__(self, hop_length: int = 512):
        """Initialize section classifier.

        Args:
            hop_length: Hop length for STFT analysis
        """
        self.hop_length = hop_length
        self.feature_analyzer = FeatureAnalyzer(hop_length)
        self.context_analyzer = ContextAnalyzer()

    def classify_sections(
        self,
        y: np.ndarray,
        sr: int,
        boundaries: List[int],
        similarity_matrix: np.ndarray = None,
    ) -> List[Dict[str, Any]]:
        """Classify sections based on audio characteristics and boundaries.

        Args:
            y: Audio signal
            sr: Sample rate
            boundaries: List of section boundaries in frame indices
            similarity_matrix: Optional similarity matrix for repetition detection

        Returns:
            List of classified sections with characteristics
        """
        if len(boundaries) < 2:
            return []

        sections = []
        all_characteristics = []

        # First pass: Extract characteristics for all sections
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]

            # Convert frames to time
            start_time = start_frame * self.hop_length / sr
            end_time = end_frame * self.hop_length / sr

            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]

            # Analyze characteristics
            characteristics = (
                self.feature_analyzer.analyze_segment_characteristics_enhanced(
                    segment, sr, start_time, end_time
                )
            )

            all_characteristics.append(characteristics)

        # Extract energy values for relative analysis
        all_energies = [char['energy'] for char in all_characteristics]

        # Second pass: Classify with context
        for i, characteristics in enumerate(all_characteristics):
            # Get context
            previous_sections = sections.copy()  # All previous sections
            next_sections = all_characteristics[
                i + 1 : i + 4
            ]  # Next 3 sections for context

            # Enhanced classification with full context
            section_type = (
                self.context_analyzer.classify_section_type_with_enhanced_context(
                    characteristics, previous_sections, next_sections, all_energies, i
                )
            )

            # Create section
            section = {
                'start_time': characteristics['start_time'],
                'end_time': characteristics['end_time'],
                'type': section_type,
                'confidence': self._calculate_confidence(characteristics, section_type),
                'characteristics': characteristics,
            }

            sections.append(section)

        # Post-processing: Apply repetition detection if similarity matrix is provided
        if similarity_matrix is not None:
            repeated_indices = self.context_analyzer.detect_verse_repetition(
                similarity_matrix, sections
            )

            # Mark repeated sections
            for idx in repeated_indices:
                if idx < len(sections):
                    sections[idx]['is_repeated'] = True

        return sections

    def _classify_section_type_with_enhanced_context(
        self,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        next_sections: List[Dict[str, Any]],
        all_energies: List[float],
        current_index: int,
    ) -> str:
        """Enhanced section type classification with full context."""
        return self.context_analyzer.classify_section_type_with_enhanced_context(
            characteristics,
            previous_sections,
            next_sections,
            all_energies,
            current_index,
        )

    def _classify_with_relative_energy(
        self,
        characteristics: Dict[str, Any],
        all_energies: List[float],
        current_index: int,
    ) -> str:
        """Classify section type based on relative energy analysis."""
        return self.context_analyzer.classify_with_relative_energy(
            characteristics, all_energies, current_index
        )

    def _analyze_energy_trend(
        self, current_index: int, all_energies: List[float]
    ) -> str:
        """Analyze energy trend around current position."""
        return self.context_analyzer._analyze_energy_trend(current_index, all_energies)

    def _apply_rb_pairing_and_verse_recovery(
        self,
        base_type: str,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        next_sections: List[Dict[str, Any]],
    ) -> str:
        """Apply R-B pairing rules and verse recovery."""
        return self.context_analyzer.apply_rb_pairing_and_verse_recovery(
            base_type, characteristics, previous_sections, next_sections
        )

    def _apply_context_rules(
        self,
        base_type: str,
        previous_sections: List[Dict[str, Any]],
        characteristics: Dict[str, Any],
        section_index: int,
    ) -> str:
        """Apply context-based rules for section classification."""
        return self.context_analyzer.apply_context_rules(
            base_type, previous_sections, characteristics, section_index
        )

    def _is_energy_building_enhanced(
        self, prev_section: Dict[str, Any], current_characteristics: Dict[str, Any]
    ) -> bool:
        """Enhanced energy building detection."""
        return self.context_analyzer._is_energy_building_enhanced(
            prev_section, current_characteristics
        )

    def _resolve_consecutive_sections(
        self, section_type: str, characteristics: Dict[str, Any]
    ) -> str:
        """Resolve consecutive sections of the same type."""
        return self.context_analyzer._resolve_consecutive_sections(
            section_type, characteristics
        )

    def _classify_section_type_with_context(
        self,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        similarity_matrix: np.ndarray = None,
        section_index: int = 0,
    ) -> str:
        """Classify section type with context analysis."""
        return self.context_analyzer.classify_section_type_with_context(
            characteristics, previous_sections, similarity_matrix, section_index
        )

    def _calculate_rhythm_density(self, segment: np.ndarray, sr: int) -> float:
        """Calculate rhythm density of a segment."""
        return self.feature_analyzer._calculate_rhythm_density(segment, sr)

    def _calculate_melody_jump_rate(self, segment: np.ndarray, sr: int) -> float:
        """Calculate melody jump rate."""
        return self.feature_analyzer._calculate_melody_jump_rate(segment, sr)

    def _analyze_segment_characteristics_enhanced(
        self, segment: np.ndarray, sr: int, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """Enhanced segment analysis with additional features."""
        return self.feature_analyzer.analyze_segment_characteristics_enhanced(
            segment, sr, start_time, end_time
        )

    def _analyze_segment_characteristics(
        self, segment: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Analyze basic characteristics of an audio segment."""
        return self.feature_analyzer.analyze_segment_characteristics(segment, sr)

    def _is_energy_building(
        self, prev_section: Dict[str, Any], current_characteristics: Dict[str, Any]
    ) -> bool:
        """Detect if energy is building from previous section."""
        # Extract characteristics from prev_section if it has nested structure
        if 'characteristics' in prev_section:
            prev_chars = prev_section['characteristics']
        else:
            prev_chars = prev_section
        return self.context_analyzer._is_energy_building_enhanced(
            prev_chars, current_characteristics
        )

    def _detect_spoken_word(
        self, segment: np.ndarray, sr: int, energy: float, complexity: float
    ) -> bool:
        """Detect if segment contains spoken word."""
        return self.feature_analyzer._detect_spoken_word(
            segment, sr, energy, complexity
        )

    def _detect_vocal_presence(self, segment: np.ndarray, sr: int) -> bool:
        """Detect vocal presence in segment."""
        return self.feature_analyzer._detect_vocal_presence(segment, sr)

    def _classify_section_type(
        self,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]] = None,
        similarity_matrix: np.ndarray = None,
    ) -> str:
        """Basic section type classification."""
        if previous_sections is None:
            previous_sections = []

        return self.context_analyzer.classify_section_type_with_context(
            characteristics,
            previous_sections,
            similarity_matrix,
            len(previous_sections),
        )

    def _calculate_voiced_ratio(self, characteristics: Dict[str, Any]) -> float:
        """Calculate voiced ratio from characteristics."""
        return self.feature_analyzer._calculate_voiced_ratio(characteristics)

    def _find_first_peak_time(
        self, characteristics: Dict[str, Any], start_time: float
    ) -> float:
        """Find first peak time in segment."""
        return self.feature_analyzer._find_first_peak_time(characteristics, start_time)

    def _detect_verse_repetition(
        self,
        similarity_matrix: np.ndarray,
        sections: List[Dict[str, Any]],
        threshold: float = 0.8,
    ) -> List[int]:
        """Detect verse repetition patterns."""
        return self.context_analyzer.detect_verse_repetition(
            similarity_matrix, sections, threshold
        )

    def _calculate_confidence(
        self, characteristics: Dict[str, Any], section_type: str
    ) -> float:
        """Calculate confidence score for section classification.

        Args:
            characteristics: Section characteristics
            section_type: Classified section type

        Returns:
            Confidence score (0-1)
        """
        energy = characteristics.get('energy', 0.0)
        complexity = characteristics.get('spectral_complexity', 0.0)
        harmonic_content = characteristics.get('harmonic_content', 0.0)

        # Base confidence
        confidence = 0.5

        # Adjust based on section type and characteristics alignment
        if section_type == 'chorus':
            # Chorus should have high energy and complexity
            if energy > 0.6 and complexity > 0.5:
                confidence += 0.3
            elif energy > 0.4:
                confidence += 0.1

        elif section_type == 'verse':
            # Verse should have moderate energy
            if 0.2 <= energy <= 0.6:
                confidence += 0.2
            if complexity < 0.7:
                confidence += 0.1

        elif section_type == 'pre_chorus':
            # Pre-chorus should have building energy
            if 0.4 <= energy <= 0.7:
                confidence += 0.2
            if 0.4 <= complexity <= 0.8:
                confidence += 0.1

        elif section_type == 'bridge':
            # Bridge should have different characteristics
            if complexity > 0.6 and harmonic_content < 0.5:
                confidence += 0.3
            elif complexity > 0.5:
                confidence += 0.1

        elif section_type in ['intro', 'outro']:
            # Intro/outro should have lower energy
            if energy < 0.4:
                confidence += 0.3
            elif energy < 0.6:
                confidence += 0.1

        return min(1.0, confidence)
