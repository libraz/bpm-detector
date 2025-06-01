"""Context analysis module for section classification."""

from typing import Any, Dict, List, Optional

import numpy as np


class ContextAnalyzer:
    """Analyzes context and applies rules for section classification."""

    def __init__(self):
        """Initialize context analyzer."""

    def classify_with_relative_energy(
        self, characteristics: Dict[str, Any], all_energies: List[float], current_index: int
    ) -> str:
        """Classify section type based on relative energy analysis.

        Args:
            characteristics: Audio characteristics
            all_energies: List of all energy values
            current_index: Current section index

        Returns:
            Section type based on relative energy
        """
        if not all_energies or current_index >= len(all_energies):
            return 'verse'

        current_energy = all_energies[current_index]

        # Calculate energy percentiles
        energy_percentiles = np.percentile(all_energies, [25, 50, 75, 90])
        # Ensure we get an array and convert to list for indexing
        percentile_array = np.asarray(energy_percentiles)
        if percentile_array.ndim > 0 and len(percentile_array) >= 4:
            p50: float = float(percentile_array[1])
            p75: float = float(percentile_array[2])
            p90: float = float(percentile_array[3])
        else:
            # Fallback for scalar case
            p50 = p75 = p90 = float(percentile_array)

        # Analyze energy trend
        trend = self._analyze_energy_trend(current_index, all_energies)

        # Classification based on relative energy and trend
        if current_energy >= p90:
            # Very high energy - likely chorus
            return 'chorus'
        elif current_energy >= p75:
            # High energy
            if trend == 'building':
                return 'pre_chorus'
            else:
                return 'chorus'
        elif current_energy >= p50:
            # Medium energy
            if trend == 'building':
                return 'pre_chorus'
            elif trend == 'declining':
                return 'bridge'
            else:
                return 'verse'
        else:
            # Low energy
            if current_index == 0:
                return 'intro'
            elif current_index == len(all_energies) - 1:
                return 'outro'
            else:
                return 'verse'

    def _analyze_energy_trend(self, current_index: int, all_energies: List[float]) -> str:
        """Analyze energy trend around current position.

        Args:
            current_index: Current section index
            all_energies: List of all energy values

        Returns:
            Energy trend: 'building', 'declining', or 'stable'
        """
        if len(all_energies) < 3 or current_index == 0:
            return 'stable'

        # Look at previous and next sections
        prev_energy = all_energies[current_index - 1] if current_index > 0 else all_energies[current_index]
        current_energy = all_energies[current_index]
        next_energy = all_energies[current_index + 1] if current_index < len(all_energies) - 1 else current_energy

        # Calculate trends
        prev_trend = current_energy - prev_energy
        next_trend = next_energy - current_energy

        # Determine overall trend
        if prev_trend > 0.1 and next_trend > 0.1:
            return 'building'
        elif prev_trend < -0.1 and next_trend < -0.1:
            return 'declining'
        elif prev_trend > 0.1:
            return 'building'
        elif next_trend < -0.1:
            return 'declining'
        else:
            return 'stable'

    def apply_rb_pairing_and_verse_recovery(
        self,
        base_type: str,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        next_sections: List[Dict[str, Any]],
    ) -> str:
        """Apply R-B pairing rules and verse recovery.

        Args:
            base_type: Base classification
            characteristics: Current section characteristics
            previous_sections: Previous sections for context
            next_sections: Next sections for context

        Returns:
            Refined section type
        """
        # R-B pairing: pre_chorus should be followed by chorus
        if base_type == 'pre_chorus':
            # Check if followed by high-energy section
            if next_sections and len(next_sections) > 0:
                next_section = next_sections[0]
                next_energy = next_section.get('energy', 0.0)
                current_energy = characteristics.get('energy', 0.0)

                # If next section has significantly higher energy, keep as pre_chorus
                if next_energy > current_energy * 1.3:
                    return 'pre_chorus'
                else:
                    # No clear chorus following, might be verse
                    return 'verse'

        # Verse recovery: isolated high-energy sections might be verses
        elif base_type == 'chorus':
            current_energy = characteristics.get('energy', 0.0)

            # Check context
            prev_energy = 0.0
            next_energy = 0.0

            if previous_sections:
                prev_energy = previous_sections[-1].get('energy', 0.0)
            if next_sections:
                next_energy = next_sections[0].get('energy', 0.0)

            # If isolated high energy (not part of a high-energy sequence)
            if current_energy > 0.6 and prev_energy < current_energy * 0.7 and next_energy < current_energy * 0.7:
                # Might be a verse with high energy
                complexity = characteristics.get('spectral_complexity', 0.0)
                if complexity < 0.6:  # Lower complexity suggests verse
                    return 'verse'

        return base_type

    def apply_context_rules(
        self,
        base_type: str,
        previous_sections: List[Dict[str, Any]],
        characteristics: Dict[str, Any],
        section_index: int,
    ) -> str:
        """Apply context-based rules for section classification.

        Args:
            base_type: Base section type
            previous_sections: List of previous sections
            characteristics: Current section characteristics
            section_index: Index of current section

        Returns:
            Refined section type
        """
        # Rule 1: First section is likely intro
        if section_index == 0:
            energy = characteristics.get('energy', 0.0)
            if energy < 0.4:  # Low energy start
                return 'intro'

        # Rule 2: Energy building detection
        if len(previous_sections) > 0:
            prev_section = previous_sections[-1]
            if self._is_energy_building_enhanced(prev_section, characteristics):
                if base_type in ['verse', 'bridge']:
                    return 'pre_chorus'

        # Rule 3: Consecutive section resolution
        if len(previous_sections) > 0:
            prev_type = previous_sections[-1].get('type', '')
            resolved_type = self._resolve_consecutive_sections(base_type, characteristics)

            # Avoid consecutive pre_chorus
            if prev_type == 'pre_chorus' and resolved_type == 'pre_chorus':
                # Check if this should be chorus instead
                energy = characteristics.get('energy', 0.0)
                if energy > 0.6:
                    return 'chorus'
                else:
                    return 'verse'

        # Rule 4: Bridge detection (requires specific characteristics)
        if base_type == 'bridge':
            complexity = characteristics.get('spectral_complexity', 0.0)
            harmonic_content = characteristics.get('harmonic_content', 0.0)

            # Bridge should have higher complexity and different harmonic content
            if complexity < 0.6 or harmonic_content < 0.3:
                return 'verse'  # Downgrade to verse

        return base_type

    def _is_energy_building_enhanced(
        self, prev_section: Dict[str, Any], current_characteristics: Dict[str, Any]
    ) -> bool:
        """Enhanced energy building detection.

        Args:
            prev_section: Previous section
            current_characteristics: Current section characteristics

        Returns:
            True if energy is building
        """
        prev_energy = prev_section.get('energy', 0.0)
        current_energy = current_characteristics.get('energy', 0.0)

        # Basic energy increase (more lenient threshold)
        energy_increase = current_energy > prev_energy * 1.1

        # Additional factors
        prev_complexity = prev_section.get('spectral_complexity', 0.0)
        current_complexity = current_characteristics.get('spectral_complexity', 0.0)
        complexity_increase = current_complexity > prev_complexity * 1.05

        # Building if energy increases significantly OR both energy and complexity increase
        significant_energy_increase = current_energy > prev_energy * 1.5
        result: bool = significant_energy_increase or (energy_increase and complexity_increase)
        return result

    def _resolve_consecutive_sections(self, section_type: str, characteristics: Dict[str, Any]) -> str:
        """Resolve consecutive sections of the same type.

        Args:
            section_type: Current section type
            characteristics: Section characteristics

        Returns:
            Resolved section type
        """
        # For consecutive verses, check if one should be pre_chorus
        if section_type == 'verse':
            energy = characteristics.get('energy', 0.0)
            complexity = characteristics.get('spectral_complexity', 0.0)

            # Higher energy and complexity might indicate pre_chorus
            if energy > 0.5 and complexity > 0.6:
                return 'pre_chorus'

        # For consecutive chorus, check if one should be bridge
        elif section_type == 'chorus':
            harmonic_content = characteristics.get('harmonic_content', 0.0)
            complexity = characteristics.get('spectral_complexity', 0.0)

            # Different harmonic content might indicate bridge
            if harmonic_content < 0.4 and complexity > 0.7:
                return 'bridge'

        return section_type

    def detect_verse_repetition(
        self, similarity_matrix: np.ndarray, sections: List[Dict[str, Any]], threshold: float = 0.8
    ) -> List[int]:
        """Detect verse repetition patterns.

        Args:
            similarity_matrix: Similarity matrix between sections
            sections: List of sections
            threshold: Similarity threshold for repetition

        Returns:
            List of indices of repeated verses
        """
        repeated_indices: List[int] = []

        if similarity_matrix.shape[0] != len(sections):
            return repeated_indices

        # Find pairs of similar sections
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                if similarity_matrix[i, j] > threshold:
                    # Check if both are classified as verse or similar
                    type_i = sections[i].get('type', '')
                    type_j = sections[j].get('type', '')

                    if type_i in ['verse', 'pre_chorus'] and type_j in ['verse', 'pre_chorus']:
                        # Mark as repeated verses
                        if i not in repeated_indices:
                            repeated_indices.append(i)
                        if j not in repeated_indices:
                            repeated_indices.append(j)

        return sorted(repeated_indices)

    def classify_section_type_with_enhanced_context(
        self,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        next_sections: List[Dict[str, Any]],
        all_energies: List[float],
        current_index: int,
    ) -> str:
        """Enhanced section type classification with full context.

        Args:
            characteristics: Current section characteristics
            previous_sections: Previous sections for context
            next_sections: Next sections for context
            all_energies: All energy values for relative analysis
            current_index: Current section index

        Returns:
            Classified section type
        """
        # Start with relative energy classification
        base_type = self.classify_with_relative_energy(characteristics, all_energies, current_index)

        # Apply R-B pairing and verse recovery
        refined_type = self.apply_rb_pairing_and_verse_recovery(
            base_type, characteristics, previous_sections, next_sections
        )

        # Apply context rules
        final_type = self.apply_context_rules(refined_type, previous_sections, characteristics, current_index)

        return final_type

    def classify_section_type_with_context(
        self,
        characteristics: Dict[str, Any],
        previous_sections: List[Dict[str, Any]],
        similarity_matrix: Optional[np.ndarray] = None,
        section_index: int = 0,
    ) -> str:
        """Classify section type with context analysis.

        Args:
            characteristics: Section characteristics
            previous_sections: Previous sections for context
            similarity_matrix: Similarity matrix (optional)
            section_index: Current section index

        Returns:
            Section type
        """
        # Basic classification based on characteristics
        energy = characteristics.get('energy', 0.0)
        complexity = characteristics.get('spectral_complexity', 0.0)
        harmonic_content = characteristics.get('harmonic_content', 0.0)

        # Initial classification
        if section_index == 0:
            # First section
            if energy < 0.3:
                base_type = 'intro'
            elif energy > 0.7:
                base_type = 'chorus'
            else:
                base_type = 'verse'
        else:
            # Subsequent sections
            if energy > 0.7 and complexity > 0.6:
                base_type = 'chorus'
            elif energy > 0.5 and complexity > 0.5:
                if len(previous_sections) > 0 and previous_sections[-1].get('type') == 'verse':
                    base_type = 'pre_chorus'
                else:
                    base_type = 'verse'
            elif complexity > 0.8 and harmonic_content < 0.4:
                base_type = 'bridge'
            elif energy < 0.2:
                base_type = 'outro'
            else:
                base_type = 'verse'

        # Apply context rules
        final_type = self.apply_context_rules(base_type, previous_sections, characteristics, section_index)

        return final_type
