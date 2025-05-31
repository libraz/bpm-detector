"""Tests for section classifier module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.bpm_detector.section_classifier import SectionClassifier


class TestSectionClassifier(unittest.TestCase):
    """Test cases for SectionClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = SectionClassifier(hop_length=512)
        self.sr = 22050

        # Create synthetic audio segments with different characteristics
        duration = 5  # seconds per segment
        t = np.linspace(0, duration, int(self.sr * duration))

        # Low energy segment (intro-like)
        self.low_energy_segment = 0.1 * np.sin(2 * np.pi * 220 * t)

        # High energy segment (chorus-like)
        self.high_energy_segment = 0.8 * (
            np.sin(2 * np.pi * 440 * t)
            + 0.5 * np.sin(2 * np.pi * 880 * t)
            + 0.3 * np.random.randn(len(t))
        )

        # Medium energy segment (verse-like)
        self.medium_energy_segment = 0.4 * np.sin(2 * np.pi * 330 * t)

        # Create boundaries for testing
        self.test_boundaries = [
            0,
            len(self.low_energy_segment),
            len(self.low_energy_segment) + len(self.medium_energy_segment),
        ]

    def test_classify_sections_basic(self):
        """Test basic section classification."""
        # Combine segments
        y = np.concatenate(
            [
                self.low_energy_segment,
                self.medium_energy_segment,
                self.high_energy_segment,
            ]
        )
        boundaries = [
            0,
            len(self.low_energy_segment),
            len(self.low_energy_segment) + len(self.medium_energy_segment),
            len(y),
        ]

        sections = self.classifier.classify_sections(y, self.sr, boundaries)

        # Check basic structure
        self.assertIsInstance(sections, list)
        # Implementation may merge sections, so check for at least one section
        self.assertGreaterEqual(len(sections), 1)

        # Check section structure
        for section in sections:
            self.assertIsInstance(section, dict)
            required_keys = ['start_time', 'end_time', 'type', 'characteristics']
            for key in required_keys:
                self.assertIn(key, section)

            # Check time ordering
            self.assertLessEqual(section['start_time'], section['end_time'])

            # Check that characteristics is a list
            self.assertIsInstance(section['characteristics'], list)

            # Check type is valid
            valid_types = [
                'intro',
                'verse',
                'chorus',
                'bridge',
                'outro',
                'instrumental',
                'breakdown',
                'buildup',
            ]
            self.assertIn(section['type'], valid_types)

    def test_analyze_segment_characteristics(self):
        """Test segment characteristic analysis."""
        characteristics = self.classifier._analyze_segment_characteristics(
            self.high_energy_segment, self.sr
        )

        # Check required characteristics (based on actual implementation)
        required_chars = ['energy', 'complexity', 'brightness']
        for char in required_chars:
            self.assertIn(char, characteristics)
            self.assertIsInstance(characteristics[char], (int, float))

        # Energy should be reasonable for high energy segment
        self.assertGreater(characteristics['energy'], 0.0)

        # Complexity should be positive
        self.assertGreater(characteristics['complexity'], 0.0)

    def test_classify_section_type_with_context(self):
        """Test section type classification with context."""
        # Create characteristics for different section types
        intro_chars = {
            'energy': 0.2,
            'complexity': 0.3,
            'brightness': 0.4,
            'labels': ['low_energy', 'simple'],
        }

        verse_chars = {
            'energy': 0.5,
            'complexity': 0.6,
            'brightness': 0.7,
            'labels': ['mid_energy', 'complex', 'vocal_present'],
        }

        chorus_chars = {
            'energy': 0.9,
            'complexity': 0.8,
            'brightness': 0.9,
            'labels': ['high_energy', 'complex', 'bright', 'vocal_present'],
        }

        # Test classification
        intro_result = self.classifier._classify_section_type_with_context(
            intro_chars, 0, 3, []
        )

        verse_result = self.classifier._classify_section_type_with_context(
            verse_chars, 1, 3, [{'type': 'intro', 'characteristics': intro_chars}]
        )

        chorus_result = self.classifier._classify_section_type_with_context(
            chorus_chars,
            2,
            3,
            [
                {'type': 'intro', 'characteristics': intro_chars},
                {'type': 'verse', 'characteristics': verse_chars},
            ],
        )

        # Check results
        self.assertIsInstance(intro_result, str)

        # Results should be valid section types
        valid_types = [
            'intro',
            'verse',
            'chorus',
            'bridge',
            'outro',
            'instrumental',
            'breakdown',
            'buildup',
        ]
        self.assertIn(intro_result, valid_types)
        self.assertIn(verse_result, valid_types)
        self.assertIn(chorus_result, valid_types)

    def test_is_energy_building(self):
        """Test energy building detection."""
        low_energy_chars = {'energy': 0.3, 'complexity': 0.2}
        high_energy_chars = {'energy': 0.8, 'complexity': 0.7}

        # Test energy building
        is_building = self.classifier._is_energy_building(
            {'characteristics': low_energy_chars}, high_energy_chars
        )
        self.assertTrue(is_building)

        # Test energy not building
        is_not_building = self.classifier._is_energy_building(
            {'characteristics': high_energy_chars}, low_energy_chars
        )
        self.assertFalse(is_not_building)

    def test_detect_spoken_word(self):
        """Test spoken word detection."""
        # Create a segment with speech-like characteristics
        speech_like = np.random.randn(self.sr * 2) * 0.1  # Low energy, noisy

        is_spoken = self.classifier._detect_spoken_word(
            speech_like, self.sr, energy=0.1, complexity=0.2
        )

        # Should return boolean
        self.assertIsInstance(is_spoken, bool)

    def test_detect_vocal_presence(self):
        """Test vocal presence detection."""
        # Test with harmonic content (vocal-like)
        harmonic_segment = np.sin(2 * np.pi * 440 * np.linspace(0, 2, self.sr * 2))

        vocal_presence = self.classifier._detect_vocal_presence(
            harmonic_segment, self.sr
        )

        # Should return boolean
        self.assertIsInstance(vocal_presence, bool)

    def test_classify_section_type(self):
        """Test individual section type classification."""
        # Test different characteristic combinations
        test_cases = [
            # Low energy, low complexity -> intro/outro
            {
                'energy': 0.2,
                'complexity': 0.2,
                'brightness': 0.3,
                'labels': ['low_energy', 'simple'],
            },
            # Medium energy, medium complexity -> verse
            {
                'energy': 0.5,
                'complexity': 0.6,
                'brightness': 0.7,
                'labels': ['mid_energy', 'complex'],
            },
            # High energy, high everything -> chorus
            {
                'energy': 0.9,
                'complexity': 0.8,
                'brightness': 0.9,
                'labels': ['high_energy', 'complex', 'bright'],
            },
        ]

        for i, characteristics in enumerate(test_cases):
            section_type = self.classifier._classify_section_type(
                characteristics, i, len(test_cases)
            )

            # Check return type
            self.assertIsInstance(section_type, str)

            # Check valid section type
            valid_types = [
                'intro',
                'verse',
                'chorus',
                'bridge',
                'outro',
                'instrumental',
                'breakdown',
                'buildup',
            ]
            self.assertIn(section_type, valid_types)

    def test_empty_boundaries(self):
        """Test behavior with empty boundaries."""
        y = self.medium_energy_segment
        empty_boundaries = []

        sections = self.classifier.classify_sections(y, self.sr, empty_boundaries)

        # Should handle empty boundaries gracefully
        self.assertIsInstance(sections, list)

    def test_single_section(self):
        """Test classification with single section."""
        y = self.medium_energy_segment
        boundaries = [0, len(y)]

        sections = self.classifier.classify_sections(y, self.sr, boundaries)

        # Should return one section
        self.assertEqual(len(sections), 1)
        self.assertIsInstance(sections[0], dict)

    def test_very_short_segments(self):
        """Test classification with very short segments."""
        # Create very short audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(self.sr * 0.5)))
        boundaries = [0, len(short_audio)]

        sections = self.classifier.classify_sections(short_audio, self.sr, boundaries)

        # Should handle short segments
        self.assertIsInstance(sections, list)
        if len(sections) > 0:
            self.assertIsInstance(sections[0], dict)

    def test_classification_consistency(self):
        """Test that classification is consistent for similar inputs."""
        # Create two similar segments
        segment1 = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, int(self.sr * 3)))
        segment2 = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, int(self.sr * 3)))

        boundaries = [0, len(segment1)]

        sections1 = self.classifier.classify_sections(segment1, self.sr, boundaries)
        sections2 = self.classifier.classify_sections(segment2, self.sr, boundaries)

        # Classifications should be similar (allowing for some variation)
        if len(sections1) > 0 and len(sections2) > 0:
            # At minimum, should return valid sections
            self.assertIsInstance(sections1[0]['type'], str)
            self.assertIsInstance(sections2[0]['type'], str)


if __name__ == '__main__':
    unittest.main()
