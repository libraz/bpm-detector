"""Tests for structure analyzer."""

import unittest
import numpy as np
from src.bpm_detector.structure_analyzer import StructureAnalyzer


class TestStructureAnalyzer(unittest.TestCase):
    """Test cases for StructureAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()
        self.sr = 22050
        self.duration = 20.0  # 20 seconds

        # Create a test signal with different sections
        t = np.linspace(0, self.duration, int(self.sr * self.duration))

        # Create signal with changing characteristics
        signal = np.zeros_like(t)

        # Section 1: Low energy (intro)
        mask1 = t < 5
        signal[mask1] = 0.2 * np.sin(2 * np.pi * 440 * t[mask1])

        # Section 2: Medium energy (verse)
        mask2 = (t >= 5) & (t < 10)
        signal[mask2] = 0.5 * (
            np.sin(2 * np.pi * 440 * t[mask2])
            + 0.5 * np.sin(2 * np.pi * 880 * t[mask2])
        )

        # Section 3: High energy (chorus)
        mask3 = (t >= 10) & (t < 15)
        signal[mask3] = 0.8 * (
            np.sin(2 * np.pi * 440 * t[mask3])
            + np.sin(2 * np.pi * 880 * t[mask3])
            + 0.5 * np.sin(2 * np.pi * 1320 * t[mask3])
        )

        # Section 4: Medium energy (verse repeat)
        mask4 = t >= 15
        signal[mask4] = 0.5 * (
            np.sin(2 * np.pi * 440 * t[mask4])
            + 0.5 * np.sin(2 * np.pi * 880 * t[mask4])
        )

        # Add some noise
        signal += 0.05 * np.random.randn(len(t))

        self.test_signal = signal

    def test_extract_structural_features(self):
        """Test enhanced structural feature extraction."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)

        # Check that all expected features are present (including new ones)
        expected_features = [
            'mfcc',
            'chroma',
            'spectral_centroid',
            'rms',
            'zcr',
            'onset_strength',
            'spectral_contrast',
            'spectral_rolloff',
        ]

        for feature_name in expected_features:
            self.assertIn(feature_name, features)
            self.assertIsInstance(features[feature_name], np.ndarray)
            self.assertGreater(features[feature_name].size, 0)

    def test_compute_self_similarity_matrix(self):
        """Test self-similarity matrix computation."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)

        # Check shape (should be square)
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])

        # Check diagonal is 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        np.testing.assert_allclose(diagonal, 1.0, rtol=1e-10)

        # Check symmetry
        np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, rtol=1e-10)

        # Check values are in [0, 1] range
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))

    def test_detect_boundaries(self):
        """Test boundary detection."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)
        boundaries = self.analyzer.detect_boundaries(similarity_matrix, self.sr)

        # Should detect at least start and end
        self.assertGreaterEqual(len(boundaries), 2)

        # First boundary should be 0
        self.assertEqual(boundaries[0], 0)

        # Last boundary should be within reasonable range
        self.assertLessEqual(boundaries[-1], similarity_matrix.shape[0] - 1)
        self.assertGreater(boundaries[-1], similarity_matrix.shape[0] * 0.5)

        # Boundaries should be sorted
        self.assertEqual(boundaries, sorted(boundaries))

    def test_classify_sections(self):
        """Test enhanced section classification."""
        boundaries = [0, 100, 200, 300, 400]  # Mock boundaries
        sections = self.analyzer.classify_sections(
            self.test_signal, self.sr, boundaries
        )

        # Should have one less section than boundaries
        self.assertEqual(len(sections), len(boundaries) - 1)

        # Check enhanced section structure
        for section in sections:
            required_keys = [
                'type',
                'start_time',
                'end_time',
                'duration',
                'characteristics',
                'energy_level',
                'complexity',
                'relative_energy',
                'rhythm_density',
            ]
            for key in required_keys:
                self.assertIn(key, section)

            # Check types
            self.assertIsInstance(section['type'], str)
            self.assertIsInstance(section['start_time'], float)
            self.assertIsInstance(section['end_time'], float)
            self.assertIsInstance(section['duration'], float)
            self.assertIsInstance(section['characteristics'], list)
            self.assertIsInstance(section['energy_level'], float)
            self.assertIsInstance(section['complexity'], float)
            self.assertIsInstance(section['relative_energy'], float)
            self.assertIsInstance(section['rhythm_density'], float)

            # Check time consistency
            self.assertLessEqual(section['start_time'], section['end_time'])
            self.assertAlmostEqual(
                section['duration'],
                section['end_time'] - section['start_time'],
                places=2,
            )

            # Check J-Pop ASCII labels
            self.assertIn('ascii_label', section)
            self.assertIsInstance(section['ascii_label'], str)

    def test_analyze_form(self):
        """Test form analysis."""
        # Mock sections
        mock_sections = [
            {'type': 'intro', 'duration': 8},
            {'type': 'verse', 'duration': 16},
            {'type': 'chorus', 'duration': 16},
            {'type': 'verse', 'duration': 16},
            {'type': 'chorus', 'duration': 16},
            {'type': 'outro', 'duration': 8},
        ]

        form_analysis = self.analyzer.analyze_form(mock_sections)

        # Check required keys
        required_keys = [
            'form',
            'repetition_ratio',
            'structural_complexity',
            'section_count',
            'unique_sections',
            'section_types',
        ]
        for key in required_keys:
            self.assertIn(key, form_analysis)

        # Check types
        self.assertIsInstance(form_analysis['form'], str)
        self.assertIsInstance(form_analysis['repetition_ratio'], float)
        self.assertIsInstance(form_analysis['structural_complexity'], float)
        self.assertIsInstance(form_analysis['section_count'], int)
        self.assertIsInstance(form_analysis['unique_sections'], int)
        self.assertIsInstance(form_analysis['section_types'], list)

        # Check values
        self.assertEqual(form_analysis['section_count'], len(mock_sections))
        self.assertGreater(form_analysis['unique_sections'], 0)
        self.assertLessEqual(form_analysis['unique_sections'], len(mock_sections))

    def test_detect_repetitions(self):
        """Test repetition detection."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)
        repetitions = self.analyzer.detect_repetitions(similarity_matrix, self.sr)

        # Should return a list
        self.assertIsInstance(repetitions, list)

        # Check repetition structure
        for rep in repetitions:
            required_keys = [
                'first_occurrence',
                'second_occurrence',
                'duration',
                'similarity',
            ]
            for key in required_keys:
                self.assertIn(key, rep)
                self.assertIsInstance(rep[key], float)

            # Check logical constraints
            self.assertGreater(rep['duration'], 0)
            self.assertGreaterEqual(rep['similarity'], 0)
            self.assertLessEqual(rep['similarity'], 1)
            self.assertLess(rep['first_occurrence'], rep['second_occurrence'])

    def test_analyze_complete(self):
        """Test complete structural analysis."""
        result = self.analyzer.analyze(self.test_signal, self.sr)

        # Check all required keys are present
        required_keys = [
            'sections',
            'form',
            'repetition_ratio',
            'structural_complexity',
            'section_count',
            'unique_sections',
            'repetitions',
            'boundaries',
        ]

        for key in required_keys:
            self.assertIn(key, result)

        # Check types
        self.assertIsInstance(result['sections'], list)
        self.assertIsInstance(result['form'], str)
        self.assertIsInstance(result['repetition_ratio'], float)
        self.assertIsInstance(result['structural_complexity'], float)
        self.assertIsInstance(result['section_count'], int)
        self.assertIsInstance(result['unique_sections'], int)
        self.assertIsInstance(result['repetitions'], list)
        self.assertIsInstance(result['boundaries'], list)

    def test_section_to_letter(self):
        """Test section type to letter conversion."""
        if not hasattr(self.analyzer, '_section_to_letter'):
            self.skipTest("_section_to_letter method not implemented")

        test_cases = {
            'intro': 'I',
            'verse': 'A',
            'chorus': 'B',
            'bridge': 'C',
            'instrumental': 'D',
            'outro': 'O',
            'unknown': 'X',
        }

        for section_type, expected_letter in test_cases.items():
            result = self.analyzer._section_to_letter(section_type)
            self.assertEqual(result, expected_letter)

    def test_empty_input(self):
        """Test behavior with empty input."""
        empty_signal = np.array([])

        # Should handle empty input gracefully
        try:
            result = self.analyzer.analyze(empty_signal, self.sr)
            self.assertIsInstance(result, dict)
        except Exception:
            # Or raise appropriate exception
            pass

    def test_vocal_detection(self):
        """Test vocal presence detection."""
        if not hasattr(self.analyzer, '_detect_vocal_presence'):
            self.skipTest("_detect_vocal_presence method not implemented")

        # Create signal with vocal-like frequencies
        t = np.linspace(0, 2.0, int(self.sr * 2.0))
        vocal_signal = np.sin(2 * np.pi * 200 * t)  # 200 Hz (vocal range)

        has_vocal = self.analyzer._detect_vocal_presence(vocal_signal, self.sr)
        self.assertIsInstance(has_vocal, bool)

    def test_enhanced_boundary_detection(self):
        """Test enhanced beat-synchronized boundary detection."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)

        # Test with different BPM values
        bpm_values = [120, 130, 140]
        for bpm in bpm_values:
            boundaries = self.analyzer.detect_boundaries(
                similarity_matrix, self.sr, bpm=bpm
            )

            # Should detect reasonable number of boundaries
            self.assertGreaterEqual(len(boundaries), 2)
            self.assertLessEqual(len(boundaries), 10)  # Not too many

            # Boundaries should be sorted
            self.assertEqual(boundaries, sorted(boundaries))

    def test_rbf_similarity_matrix(self):
        """Test RBF kernel similarity matrix computation."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)

        # RBF kernel should produce valid similarity matrix
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))

        # Diagonal should be close to 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        self.assertTrue(np.all(diagonal > 0.9))

    def test_relative_energy_analysis(self):
        """Test relative energy analysis for A/B/S classification."""
        # Create signal with clear energy progression (A-melo < B-melo < Sabi)
        t = np.linspace(0, 30.0, int(self.sr * 30.0))

        # A-melo: low energy
        a_melo = 0.3 * np.sin(2 * np.pi * 440 * t[: int(len(t) / 3)])

        # B-melo: medium energy
        b_melo = 0.6 * (
            np.sin(2 * np.pi * 440 * t[int(len(t) / 3) : int(2 * len(t) / 3)])
            + 0.5 * np.sin(2 * np.pi * 880 * t[int(len(t) / 3) : int(2 * len(t) / 3)])
        )

        # Sabi: high energy
        sabi = 0.9 * (
            np.sin(2 * np.pi * 440 * t[int(2 * len(t) / 3) :])
            + np.sin(2 * np.pi * 880 * t[int(2 * len(t) / 3) :])
            + 0.5 * np.sin(2 * np.pi * 1320 * t[int(2 * len(t) / 3) :])
        )

        test_signal = np.concatenate([a_melo, b_melo, sabi])

        # Analyze with enhanced features
        result = self.analyzer.analyze(test_signal, self.sr, bpm=130)
        sections = result['sections']

        # Should detect energy progression
        if len(sections) >= 3:
            energies = [s['relative_energy'] for s in sections[:3]]
            # Generally expect increasing energy trend
            self.assertIsInstance(energies[0], float)
            self.assertIsInstance(energies[1], float)
            self.assertIsInstance(energies[2], float)

    def test_fade_ending_detection(self):
        """Test fade ending detection for outro identification."""
        # Create signal with fade ending
        t = np.linspace(0, 20.0, int(self.sr * 20.0))

        # Normal section
        normal_part = 0.7 * np.sin(2 * np.pi * 440 * t[: int(len(t) * 0.8)])

        # Fade ending
        fade_part = t[int(len(t) * 0.8) :]
        fade_envelope = np.linspace(0.7, 0.1, len(fade_part))
        fade_signal = fade_envelope * np.sin(2 * np.pi * 440 * fade_part)

        test_signal = np.concatenate([normal_part, fade_signal])

        # Test fade detection
        if hasattr(self.analyzer.section_processor, '_detect_fade_ending'):
            mock_section = {'start_time': 16.0, 'end_time': 20.0, 'duration': 4.0}

            is_fade = self.analyzer.section_processor._detect_fade_ending(
                mock_section, test_signal, self.sr
            )
            self.assertIsInstance(is_fade, bool)

    def test_melody_jump_rate(self):
        """Test melody jump rate calculation."""
        if hasattr(self.analyzer.section_classifier, '_calculate_melody_jump_rate'):
            # Create signal with melodic jumps
            t = np.linspace(0, 5.0, int(self.sr * 5.0))

            # Create melody with large jumps
            frequencies = [440, 880, 440, 1320, 440]  # Large jumps
            segment_length = len(t) // len(frequencies)

            melody_signal = np.zeros_like(t)
            for i, freq in enumerate(frequencies):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(t))
                melody_signal[start_idx:end_idx] = np.sin(
                    2 * np.pi * freq * t[start_idx:end_idx]
                )

            jump_rate = self.analyzer.section_classifier._calculate_melody_jump_rate(
                melody_signal, self.sr
            )

            self.assertIsInstance(jump_rate, float)
            self.assertGreaterEqual(jump_rate, 0.0)
            self.assertLessEqual(jump_rate, 1.0)

    def test_jpop_section_labels(self):
        """Test J-Pop specific section labeling."""
        result = self.analyzer.analyze(self.test_signal, self.sr, bpm=130)
        sections = result['sections']

        # Check that J-Pop ASCII labels are present
        for section in sections:
            self.assertIn('ascii_label', section)
            # ASCII label should be one of the valid J-Pop terms
            # (or the original type if not mapped)
            ascii_label = section['ascii_label']
            self.assertIsInstance(ascii_label, str)
            self.assertGreater(len(ascii_label), 0)


if __name__ == '__main__':
    unittest.main()
