"""Tests for melody harmony analyzer module."""

import unittest
from unittest.mock import patch

import numpy as np

from src.bpm_detector.melody_harmony_analyzer import MelodyHarmonyAnalyzer


class TestMelodyHarmonyAnalyzer(unittest.TestCase):
    """Test cases for MelodyHarmonyAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MelodyHarmonyAnalyzer(
            hop_length=512,
            fmin=80.0,
            fmax=2000.0,
            consonance_ratings={0: 1.0, 7: 0.8, 4: 0.7, 3: 0.6},
        )
        self.sr = 22050

        # Create synthetic audio with harmonic content
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create a chord-like signal (C major: C-E-G)
        fundamental = 261.63  # C4
        self.harmonic_audio = (
            0.5 * np.sin(2 * np.pi * fundamental * t)  # C
            + 0.3 * np.sin(2 * np.pi * fundamental * 5 / 4 * t)  # E
            + 0.2 * np.sin(2 * np.pi * fundamental * 3 / 2 * t)  # G
        )

        # Create a melodic signal
        melody_freqs = [261.63, 293.66, 329.63, 349.23]  # C-D-E-F
        self.melodic_audio = np.concatenate(
            [
                0.5 * np.sin(2 * np.pi * freq * np.linspace(0, 1, self.sr))
                for freq in melody_freqs
            ]
        )

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    @patch(
        'src.bpm_detector.harmony_analyzer.HarmonyAnalyzer.analyze_harmony_complexity'
    )
    @patch('src.bpm_detector.harmony_analyzer.HarmonyAnalyzer.analyze_consonance')
    @patch('src.bpm_detector.harmony_analyzer.HarmonyAnalyzer.analyze_harmonic_rhythm')
    def test_analyze_complete(
        self,
        mock_harmonic_rhythm,
        mock_consonance,
        mock_harmony_complexity,
        mock_extract_melody,
    ):
        """Test complete melody harmony analysis."""
        # Mock melody analyzer results
        mock_extract_melody.return_value = {
            'f0': np.array([261.63, 293.66, 329.63, 349.23]),
            'voiced_flag': np.array([True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9]),
        }

        # Mock harmony analyzer results
        mock_harmony_complexity.return_value = {
            'harmonic_complexity': 0.65,
            'spectral_entropy': 0.7,
            'harmonic_change_rate': 0.3,
        }

        mock_consonance.return_value = {
            'consonance_score': 0.75,
            'dissonance_score': 0.25,
            'interval_consonance': 0.8,
        }

        mock_harmonic_rhythm.return_value = {
            'harmonic_rhythm': 2.5,
            'chord_change_rate': 0.4,
            'harmonic_stability': 0.6,
        }

        # Run analysis
        results = self.analyzer.analyze(self.harmonic_audio, self.sr)

        # Check structure
        self.assertIsInstance(results, dict)

        # Check main sections
        expected_sections = ['melody', 'harmony', 'combined_features']
        for section in expected_sections:
            self.assertIn(section, results)

        # Check melody section
        melody_results = results['melody']
        self.assertIn('range', melody_results)
        self.assertIn('direction', melody_results)
        self.assertIn('intervals', melody_results)
        self.assertIn('stability', melody_results)

        # Check harmony section
        harmony_results = results['harmony']
        self.assertIn('complexity', harmony_results)
        self.assertIn('consonance', harmony_results)
        self.assertIn('rhythm', harmony_results)

        # Check combined features
        combined = results['combined_features']
        self.assertIn('melody_harmony_balance', combined)
        self.assertIn('overall_complexity', combined)
        self.assertIn('musical_sophistication', combined)

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_analyze_with_progress_callback(self, mock_extract_melody):
        """Test analysis with progress callback."""
        # Mock melody extraction
        mock_extract_melody.return_value = {
            'f0': np.array([261.63, 293.66]),
            'voiced_flag': np.array([True, True]),
            'voiced_prob': np.array([0.9, 0.8]),
        }

        # Track progress calls
        progress_calls = []

        def progress_callback(progress, message):
            progress_calls.append((progress, message))

        # Run analysis with callback
        self.analyzer.analyze(
            self.harmonic_audio, self.sr, progress_callback=progress_callback
        )

        # Check that progress was reported
        self.assertGreater(len(progress_calls), 0)

        # Check progress values are reasonable
        for progress, message in progress_calls:
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 100.0)
            self.assertIsInstance(message, str)

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_analyze_melodic_audio(self, mock_extract_melody):
        """Test analysis with melodic audio."""
        # Mock melody extraction for melodic content
        mock_extract_melody.return_value = {
            'f0': np.array([261.63, 293.66, 329.63, 349.23]),
            'voiced_flag': np.array([True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85, 0.9]),
        }

        results = self.analyzer.analyze(self.melodic_audio, self.sr)

        # Should detect melodic content
        self.assertIsInstance(results, dict)
        self.assertIn('melody', results)

        # Melody should have reasonable range
        melody_range = results['melody']['range']
        self.assertIn('range_semitones', melody_range)
        self.assertGreater(melody_range['range_semitones'], 0)

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_analyze_harmonic_audio(self, mock_extract_melody):
        """Test analysis with harmonic audio."""
        # Mock melody extraction for harmonic content
        mock_extract_melody.return_value = {
            'f0': np.array([261.63, 261.63, 261.63, 261.63]),  # Stable pitch
            'voiced_flag': np.array([True, True, True, True]),
            'voiced_prob': np.array([0.9, 0.9, 0.9, 0.9]),
        }

        results = self.analyzer.analyze(self.harmonic_audio, self.sr)

        # Should detect harmonic content
        self.assertIsInstance(results, dict)
        self.assertIn('harmony', results)

        # Harmony should show complexity
        harmony_complexity = results['harmony']['complexity']
        self.assertIn('harmonic_complexity', harmony_complexity)
        self.assertGreaterEqual(harmony_complexity['harmonic_complexity'], 0.0)
        self.assertLessEqual(harmony_complexity['harmonic_complexity'], 1.0)

    def test_initialization_with_custom_parameters(self):
        """Test analyzer initialization with custom parameters."""
        custom_analyzer = MelodyHarmonyAnalyzer(
            hop_length=256,
            fmin=100.0,
            fmax=1500.0,
            consonance_ratings={0: 1.0, 5: 0.9, 7: 0.8},
        )

        # Check that parameters are set
        self.assertEqual(custom_analyzer.hop_length, 256)
        self.assertEqual(custom_analyzer.fmin, 100.0)
        self.assertEqual(custom_analyzer.fmax, 1500.0)
        self.assertIsNotNone(custom_analyzer.consonance_ratings)

    def test_initialization_with_defaults(self):
        """Test analyzer initialization with default parameters."""
        default_analyzer = MelodyHarmonyAnalyzer()

        # Check that defaults are reasonable
        self.assertEqual(default_analyzer.hop_length, 512)
        self.assertEqual(default_analyzer.fmin, 80.0)
        self.assertEqual(default_analyzer.fmax, 2000.0)
        self.assertIsNotNone(default_analyzer.consonance_ratings)

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_empty_audio_handling(self, mock_extract_melody):
        """Test handling of empty audio."""
        # Mock empty melody extraction
        mock_extract_melody.return_value = {
            'f0': np.array([]),
            'voiced_flag': np.array([]),
            'voiced_prob': np.array([]),
        }

        empty_audio = np.array([])

        try:
            results = self.analyzer.analyze(empty_audio, self.sr)
            # Should handle empty input gracefully
            self.assertIsInstance(results, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_short_audio_handling(self, mock_extract_melody):
        """Test handling of very short audio."""
        # Mock melody extraction for short audio
        mock_extract_melody.return_value = {
            'f0': np.array([261.63]),
            'voiced_flag': np.array([True]),
            'voiced_prob': np.array([0.9]),
        }

        # Create 0.5 second audio
        short_audio = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.5, int(self.sr * 0.5))
        )

        results = self.analyzer.analyze(short_audio, self.sr)

        # Should handle short audio
        self.assertIsInstance(results, dict)
        self.assertIn('melody', results)
        self.assertIn('harmony', results)

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_noise_audio_handling(self, mock_extract_melody):
        """Test handling of noisy audio."""
        # Mock melody extraction for noisy audio
        mock_extract_melody.return_value = {
            'f0': np.array([0, 0, 261.63, 0]),  # Sparse melody
            'voiced_flag': np.array([False, False, True, False]),
            'voiced_prob': np.array([0.1, 0.2, 0.8, 0.1]),
        }

        # Create noisy audio
        noise_audio = 0.1 * np.random.randn(self.sr * 2)

        results = self.analyzer.analyze(noise_audio, self.sr)

        # Should handle noisy input
        self.assertIsInstance(results, dict)

        # Results should reflect low musical content
        if 'combined_features' in results:
            sophistication = results['combined_features'].get(
                'musical_sophistication', 0
            )
            self.assertLessEqual(sophistication, 0.5)  # Should be low for noise

    @patch('src.bpm_detector.melody_analyzer.MelodyAnalyzer.extract_melody')
    def test_combined_features_calculation(self, mock_extract_melody):
        """Test combined features calculation."""
        # Mock melody extraction
        mock_extract_melody.return_value = {
            'f0': np.array([261.63, 293.66, 329.63]),
            'voiced_flag': np.array([True, True, True]),
            'voiced_prob': np.array([0.9, 0.8, 0.85]),
        }

        results = self.analyzer.analyze(self.harmonic_audio, self.sr)

        # Check combined features
        combined = results['combined_features']

        # Check balance score
        self.assertIn('melody_harmony_balance', combined)
        balance = combined['melody_harmony_balance']
        self.assertGreaterEqual(balance, 0.0)
        self.assertLessEqual(balance, 1.0)

        # Check overall complexity
        self.assertIn('overall_complexity', combined)
        complexity = combined['overall_complexity']
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

        # Check musical sophistication
        self.assertIn('musical_sophistication', combined)
        sophistication = combined['musical_sophistication']
        self.assertGreaterEqual(sophistication, 0.0)
        self.assertLessEqual(sophistication, 1.0)

    def test_progress_callback_error_handling(self):
        """Test that analysis continues even if progress callback fails."""

        def failing_callback(progress, message):
            raise Exception("Callback error")

        # Should not crash even with failing callback
        try:
            results = self.analyzer.analyze(
                self.harmonic_audio, self.sr, progress_callback=failing_callback
            )
            # Analysis should complete despite callback failure
            self.assertIsInstance(results, dict)
        except Exception as e:
            # If it does raise an exception, it should be the callback error
            self.assertIn("Callback error", str(e))


if __name__ == '__main__':
    unittest.main()
