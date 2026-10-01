"""Regression tests for context-aware section classification."""

import unittest

from src.bpm_detector.context_analyzer import ContextAnalyzer


class TestContextAnalyzer(unittest.TestCase):
    """Test context features in the same nested shape as classifier output."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_nested_previous_characteristics_follow_energy_trend(self):
        """Nested section characteristics drive rising, stable, and falling context."""
        cases = (
            (
                'descending',
                {'energy': 0.8, 'spectral_complexity': 0.7},
                {'energy': 0.4, 'spectral_complexity': 0.5},
                False,
                'verse',
            ),
            (
                'stable',
                {'energy': 0.5, 'spectral_complexity': 0.5},
                {'energy': 0.52, 'spectral_complexity': 0.51},
                False,
                'verse',
            ),
            (
                'rising',
                {'energy': 0.4, 'spectral_complexity': 0.4},
                {'energy': 0.8, 'spectral_complexity': 0.7},
                True,
                'pre_chorus',
            ),
        )

        for name, previous_characteristics, current_characteristics, expected_building, expected_type in cases:
            with self.subTest(trend=name):
                previous_sections = [{'type': 'verse', 'characteristics': previous_characteristics}]

                self.assertEqual(
                    self.analyzer._is_energy_building_enhanced(previous_sections[-1], current_characteristics),
                    expected_building,
                )
                self.assertEqual(
                    self.analyzer.apply_context_rules(
                        'verse', previous_sections, current_characteristics, section_index=1
                    ),
                    expected_type,
                )

    def test_flat_previous_characteristics_remain_supported(self):
        """Legacy flat section dictionaries continue to participate in context rules."""
        previous_sections = [{'type': 'verse', 'energy': 0.4, 'spectral_complexity': 0.4}]
        current_characteristics = {'energy': 0.8, 'spectral_complexity': 0.7}

        self.assertTrue(self.analyzer._is_energy_building_enhanced(previous_sections[0], current_characteristics))
        self.assertEqual(
            self.analyzer.apply_context_rules('verse', previous_sections, current_characteristics, section_index=1),
            'pre_chorus',
        )

    def test_nested_next_characteristics_are_used_for_rb_pairing(self):
        """R-B pairing reads the nested energy emitted by SectionClassifier."""
        current_characteristics = {'energy': 0.5, 'spectral_complexity': 0.5}
        next_sections = [{'characteristics': {'energy': 0.8, 'spectral_complexity': 0.7}}]

        self.assertEqual(
            self.analyzer.apply_rb_pairing_and_verse_recovery('pre_chorus', current_characteristics, [], next_sections),
            'pre_chorus',
        )


if __name__ == '__main__':
    unittest.main()
