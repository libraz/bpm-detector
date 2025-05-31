"""Tests for melody analyzer module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.bpm_detector.melody_analyzer import MelodyAnalyzer


class TestMelodyAnalyzer(unittest.TestCase):
    """Test cases for MelodyAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MelodyAnalyzer(hop_length=512, fmin=80.0, fmax=2000.0)
        self.sr = 22050

        # Create test audio with melodic content
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create a simple melody (C-D-E-F-G)
        note_duration = duration / 5
        melody_freqs = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4-D4-E4-F4-G4

        self.melodic_audio = np.zeros_like(t)
        for i, freq in enumerate(melody_freqs):
            start_idx = int(i * note_duration * self.sr)
            end_idx = int((i + 1) * note_duration * self.sr)
            if end_idx <= len(t):
                self.melodic_audio[start_idx:end_idx] = np.sin(
                    2 * np.pi * freq * t[start_idx:end_idx]
                )

        # Create audio with wide melodic range
        wide_range_freqs = [130.81, 523.25]  # C3 to C5 (2 octaves)
        self.wide_range_audio = np.zeros_like(t)
        for i, freq in enumerate(wide_range_freqs):
            start_idx = int(i * (duration / 2) * self.sr)
            end_idx = int((i + 1) * (duration / 2) * self.sr)
            if end_idx <= len(t):
                self.wide_range_audio[start_idx:end_idx] = np.sin(
                    2 * np.pi * freq * t[start_idx:end_idx]
                )

        # Create audio with vibrato
        vibrato_freq = 440.0
        vibrato_rate = 5.0  # 5 Hz vibrato
        vibrato_depth = 10.0  # 10 Hz depth
        self.vibrato_audio = np.sin(
            2
            * np.pi
            * (vibrato_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
            * t
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        # Test with default parameters
        default_analyzer = MelodyAnalyzer()
        self.assertEqual(default_analyzer.hop_length, 512)
        self.assertEqual(default_analyzer.fmin, 80.0)
        self.assertEqual(default_analyzer.fmax, 2000.0)

        # Test with custom parameters
        custom_analyzer = MelodyAnalyzer(hop_length=256, fmin=100.0, fmax=1500.0)
        self.assertEqual(custom_analyzer.hop_length, 256)
        self.assertEqual(custom_analyzer.fmin, 100.0)
        self.assertEqual(custom_analyzer.fmax, 1500.0)

    @patch('librosa.pyin')
    def test_extract_melody(self, mock_pyin):
        """Test melody extraction."""
        # Mock librosa.pyin output
        mock_f0 = np.array([261.63, 293.66, 329.63, 349.23, 392.00])
        mock_voiced_flag = np.array([True, True, True, True, True])
        mock_voiced_prob = np.array([0.9, 0.8, 0.85, 0.9, 0.88])

        mock_pyin.return_value = (mock_f0, mock_voiced_flag, mock_voiced_prob)

        melody = self.analyzer.extract_melody(self.melodic_audio, self.sr)

        # Check required fields
        expected_fields = ['f0', 'voiced_flag', 'voiced_prob']
        for field in expected_fields:
            self.assertIn(field, melody)
            self.assertIsInstance(melody[field], np.ndarray)

        # Check that pyin was called with correct parameters
        mock_pyin.assert_called_once()
        call_args = mock_pyin.call_args
        self.assertEqual(call_args[1]['fmin'], self.analyzer.fmin)
        self.assertEqual(call_args[1]['fmax'], self.analyzer.fmax)

    def test_analyze_melodic_range(self):
        """Test melodic range analysis."""
        # Create mock melody data
        mock_melody = {
            'f0': np.array([261.63, 293.66, 329.63, 349.23, 392.00]),
            'voiced_flag': np.array([True, True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9, 0.88]),
        }

        range_analysis = self.analyzer.analyze_melodic_range(mock_melody)

        # Check required fields
        expected_fields = [
            'range_semitones',
            'range_octaves',
            'lowest_note',
            'highest_note',
            'vocal_range_classification',
        ]
        for field in expected_fields:
            self.assertIn(field, range_analysis)

        # Check types and ranges
        self.assertIsInstance(range_analysis['range_semitones'], (int, float))
        self.assertIsInstance(range_analysis['range_octaves'], (int, float))
        self.assertIsInstance(range_analysis['lowest_note'], (int, float))
        self.assertIsInstance(range_analysis['highest_note'], (int, float))
        self.assertIsInstance(range_analysis['vocal_range_classification'], str)

        # Check logical relationships
        self.assertLessEqual(
            range_analysis['lowest_note'], range_analysis['highest_note']
        )
        self.assertGreaterEqual(range_analysis['range_semitones'], 0)
        self.assertGreaterEqual(range_analysis['range_octaves'], 0)

    def test_extract_vocal_range(self):
        """Test vocal range extraction."""
        # Create notes with some non-vocal frequencies
        all_notes = np.array(
            [50, 100, 200, 300, 400, 500, 1000, 2000, 3000]
        )  # Mix of vocal and non-vocal

        vocal_notes = self.analyzer._extract_vocal_range(all_notes)

        # Should filter to vocal range (approximately 80-1000 Hz)
        self.assertTrue(np.all(vocal_notes >= 80))
        self.assertTrue(np.all(vocal_notes <= 1000))
        self.assertLessEqual(len(vocal_notes), len(all_notes))

    def test_analyze_melodic_direction(self):
        """Test melodic direction analysis."""
        # Create ascending melody
        ascending_melody = {
            'f0': np.array([261.63, 293.66, 329.63, 349.23, 392.00]),
            'voiced_flag': np.array([True, True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9, 0.88]),
        }

        direction_analysis = self.analyzer.analyze_melodic_direction(ascending_melody)

        # Check required fields
        expected_fields = [
            'overall_direction',
            'direction_changes',
            'step_sizes',
            'contour_complexity',
        ]
        for field in expected_fields:
            self.assertIn(field, direction_analysis)

        # Check types
        self.assertIsInstance(direction_analysis['overall_direction'], str)
        self.assertIsInstance(direction_analysis['direction_changes'], (int, float))
        self.assertIsInstance(direction_analysis['step_sizes'], dict)
        self.assertIsInstance(direction_analysis['contour_complexity'], (int, float))

        # Ascending melody should be detected as ascending
        self.assertEqual(direction_analysis['overall_direction'], 'ascending')

        # Test with descending melody
        descending_melody = {
            'f0': np.array([392.00, 349.23, 329.63, 293.66, 261.63]),
            'voiced_flag': np.array([True, True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9, 0.88]),
        }

        desc_direction = self.analyzer.analyze_melodic_direction(descending_melody)
        self.assertEqual(desc_direction['overall_direction'], 'descending')

    def test_analyze_interval_distribution(self):
        """Test interval distribution analysis."""
        # Create melody with known intervals
        melody_with_intervals = {
            'f0': np.array(
                [261.63, 293.66, 329.63, 261.63]
            ),  # C-D-E-C (whole step, whole step, major third down)
            'voiced_flag': np.array([True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9]),
        }

        interval_analysis = self.analyzer.analyze_interval_distribution(
            melody_with_intervals
        )

        # Check that it returns a dictionary
        self.assertIsInstance(interval_analysis, dict)

        # Should have interval categories
        expected_categories = ['small_intervals', 'medium_intervals', 'large_intervals']
        for category in expected_categories:
            if category in interval_analysis:
                self.assertIsInstance(interval_analysis[category], (int, float))
                self.assertGreaterEqual(interval_analysis[category], 0.0)
                self.assertLessEqual(interval_analysis[category], 1.0)

    def test_analyze_pitch_stability(self):
        """Test pitch stability analysis."""
        # Create stable melody (no vibrato)
        stable_melody = {
            'f0': np.array([440.0] * 100),  # Constant pitch
            'voiced_flag': np.array([True] * 100),
            'voiced_prob': np.array([0.9] * 100),
        }

        stability_analysis = self.analyzer.analyze_pitch_stability(stable_melody)

        # Check required fields
        expected_fields = [
            'pitch_stability',
            'vibrato_rate',
            'vibrato_extent',
            'pitch_drift',
        ]
        for field in expected_fields:
            self.assertIn(field, stability_analysis)
            self.assertIsInstance(stability_analysis[field], (int, float))

        # Stable pitch should have high stability
        self.assertGreater(stability_analysis['pitch_stability'], 0.8)

        # Create unstable melody with vibrato
        vibrato_f0 = 440.0 + 10 * np.sin(
            2 * np.pi * 5 * np.linspace(0, 2, 100)
        )  # 5 Hz vibrato
        unstable_melody = {
            'f0': vibrato_f0,
            'voiced_flag': np.array([True] * 100),
            'voiced_prob': np.array([0.9] * 100),
        }

        unstable_analysis = self.analyzer.analyze_pitch_stability(unstable_melody)

        # Unstable pitch should have lower stability
        self.assertLess(
            unstable_analysis['pitch_stability'], stability_analysis['pitch_stability']
        )

        # Should detect vibrato
        self.assertGreater(unstable_analysis['vibrato_rate'], 0)

    def test_detect_vibrato(self):
        """Test vibrato detection."""
        # Create signal with known vibrato
        vibrato_rate = 5.0  # 5 Hz
        vibrato_extent = 10.0  # 10 Hz extent
        f0_with_vibrato = 440.0 + vibrato_extent * np.sin(
            2 * np.pi * vibrato_rate * np.linspace(0, 2, 100)
        )

        detected_rate, detected_extent = self.analyzer._detect_vibrato(f0_with_vibrato)

        # Should detect vibrato parameters (with some tolerance)
        self.assertIsInstance(detected_rate, (int, float))
        self.assertIsInstance(detected_extent, (int, float))
        self.assertGreater(detected_rate, 0)
        self.assertGreater(detected_extent, 0)

        # Should be reasonably close to actual values
        self.assertLess(
            abs(detected_rate - vibrato_rate), 1.0
        )  # Within 1.0 Hz (reasonable precision)
        self.assertLess(
            abs(detected_extent - vibrato_extent), 10.0
        )  # Within 10 Hz (reasonable for extent)

    def test_calculate_pitch_drift(self):
        """Test pitch drift calculation."""
        # Create signal with drift
        drifting_f0 = np.linspace(440, 460, 100)  # 20 Hz drift over time

        drift = self.analyzer._calculate_pitch_drift(drifting_f0)

        self.assertIsInstance(drift, (int, float))
        self.assertGreater(drift, 0)

        # Test with stable pitch
        stable_f0 = np.array([440.0] * 100)
        stable_drift = self.analyzer._calculate_pitch_drift(stable_f0)

        # Drifting pitch should have higher drift than stable pitch
        self.assertGreater(drift, stable_drift)

    def test_empty_melody_handling(self):
        """Test handling of empty melody."""
        empty_melody = {
            'f0': np.array([]),
            'voiced_flag': np.array([]),
            'voiced_prob': np.array([]),
        }

        try:
            range_analysis = self.analyzer.analyze_melodic_range(empty_melody)
            # Should handle empty input gracefully
            self.assertIsInstance(range_analysis, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    def test_unvoiced_melody_handling(self):
        """Test handling of completely unvoiced melody."""
        unvoiced_melody = {
            'f0': np.array([0, 0, 0, 0, 0]),
            'voiced_flag': np.array([False, False, False, False, False]),
            'voiced_prob': np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        }

        range_analysis = self.analyzer.analyze_melodic_range(unvoiced_melody)

        # Should handle unvoiced input
        self.assertIsInstance(range_analysis, dict)

        # Range should be minimal for unvoiced content
        self.assertEqual(range_analysis['range_semitones'], 0)

    def test_single_note_melody(self):
        """Test handling of single note melody."""
        single_note_melody = {
            'f0': np.array([440.0]),
            'voiced_flag': np.array([True]),
            'voiced_prob': np.array([0.9]),
        }

        direction_analysis = self.analyzer.analyze_melodic_direction(single_note_melody)

        # Should handle single note gracefully
        self.assertIsInstance(direction_analysis, dict)
        self.assertEqual(direction_analysis['overall_direction'], 'static')
        self.assertEqual(direction_analysis['direction_changes'], 0)

    def test_vocal_range_classification(self):
        """Test vocal range classification."""
        # Test soprano range
        soprano_melody = {
            'f0': np.array([523.25, 587.33, 659.25, 698.46]),  # C5-D5-E5-F5
            'voiced_flag': np.array([True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9]),
        }

        soprano_analysis = self.analyzer.analyze_melodic_range(soprano_melody)
        classification = soprano_analysis['vocal_range_classification']

        # Should classify as soprano or similar high range
        self.assertIsInstance(classification, str)
        self.assertIn(classification.lower(), ['soprano', 'high', 'treble'])

    def test_large_interval_detection(self):
        """Test detection of large melodic intervals."""
        # Create melody with octave jump
        large_interval_melody = {
            'f0': np.array([261.63, 523.25]),  # C4 to C5 (octave)
            'voiced_flag': np.array([True, True]),
            'voiced_prob': np.array([0.9, 0.9]),
        }

        interval_analysis = self.analyzer.analyze_interval_distribution(
            large_interval_melody
        )

        # Should detect large intervals
        if 'large_intervals' in interval_analysis:
            self.assertGreater(interval_analysis['large_intervals'], 0.5)

    def test_melodic_complexity_assessment(self):
        """Test assessment of melodic complexity."""
        # Simple melody (few notes, small intervals)
        simple_melody = {
            'f0': np.array([261.63, 293.66, 261.63]),  # C-D-C
            'voiced_flag': np.array([True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.9]),
        }

        # Complex melody (many notes, large intervals, direction changes)
        complex_melody = {
            'f0': np.array(
                [261.63, 392.00, 220.00, 523.25, 174.61, 440.00]
            ),  # C4-G4-A3-C5-F3-A4
            'voiced_flag': np.array([True, True, True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9, 0.88, 0.87]),
        }

        simple_direction = self.analyzer.analyze_melodic_direction(simple_melody)
        complex_direction = self.analyzer.analyze_melodic_direction(complex_melody)

        # Complex melody should have more direction changes
        self.assertGreater(
            complex_direction['direction_changes'],
            simple_direction['direction_changes'],
        )

        # Complex melody should have higher contour complexity
        self.assertGreater(
            complex_direction['contour_complexity'],
            simple_direction['contour_complexity'],
        )


if __name__ == '__main__':
    unittest.main()
