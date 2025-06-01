"""Tests for harmony analyzer module."""

import unittest

import numpy as np

from src.bpm_detector.harmony_analyzer import HarmonyAnalyzer


class TestHarmonyAnalyzer(unittest.TestCase):
    """Test cases for HarmonyAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = HarmonyAnalyzer(hop_length=512)
        self.sr = 22050

        # Create test audio with harmonic content
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create harmonic audio (C major chord with overtones)
        fundamental = 261.63  # C4
        self.harmonic_audio = (
            1.0 * np.sin(2 * np.pi * fundamental * t)  # Fundamental
            + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)  # 2nd harmonic
            + 0.3 * np.sin(2 * np.pi * fundamental * 3 * t)  # 3rd harmonic
            + 0.2 * np.sin(2 * np.pi * fundamental * 4 * t)  # 4th harmonic
            + 0.4 * np.sin(2 * np.pi * fundamental * 5 / 4 * t)  # Major third
            + 0.3 * np.sin(2 * np.pi * fundamental * 3 / 2 * t)  # Perfect fifth
        )

        # Create dissonant audio
        self.dissonant_audio = 0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.5 * np.sin(  # C
            2 * np.pi * 277.18 * t
        )  # C# (minor second)

        # Create simple sine wave
        self.simple_audio = np.sin(2 * np.pi * 440 * t)

    def test_initialization(self):
        """Test analyzer initialization."""
        # Test with default parameters
        default_analyzer = HarmonyAnalyzer()
        self.assertEqual(default_analyzer.hop_length, 512)
        self.assertIsNotNone(default_analyzer.consonance_ratings)

        # Test with custom parameters
        custom_consonance = {0: 1.0, 7: 0.8, 4: 0.7}
        custom_analyzer = HarmonyAnalyzer(hop_length=256, consonance_ratings=custom_consonance)
        self.assertEqual(custom_analyzer.hop_length, 256)
        self.assertEqual(custom_analyzer.consonance_ratings, custom_consonance)

    def test_analyze_harmony_complexity(self):
        """Test harmony complexity analysis."""
        # Test with harmonic audio
        harmonic_complexity = self.analyzer.analyze_harmony_complexity(self.harmonic_audio, self.sr)

        # Check required fields
        expected_fields = ['harmonic_complexity', 'spectral_entropy', 'harmonic_change_rate']
        for field in expected_fields:
            self.assertIn(field, harmonic_complexity)
            self.assertIsInstance(harmonic_complexity[field], (int, float))
            self.assertGreaterEqual(harmonic_complexity[field], 0.0)
            self.assertLessEqual(harmonic_complexity[field], 1.0)

        # Test with simple audio
        simple_complexity = self.analyzer.analyze_harmony_complexity(self.simple_audio, self.sr)

        # Harmonic audio should have higher complexity than simple sine wave
        self.assertGreater(harmonic_complexity['harmonic_complexity'], simple_complexity['harmonic_complexity'])

    def test_analyze_consonance(self):
        """Test consonance analysis."""
        # Test with consonant harmonic audio
        consonant_analysis = self.analyzer.analyze_consonance(self.harmonic_audio, self.sr)

        # Check required fields
        expected_fields = ['consonance_score', 'dissonance_score', 'interval_consonance']
        for field in expected_fields:
            self.assertIn(field, consonant_analysis)
            self.assertIsInstance(consonant_analysis[field], (int, float))
            self.assertGreaterEqual(consonant_analysis[field], 0.0)
            self.assertLessEqual(consonant_analysis[field], 1.0)

        # Test with dissonant audio
        dissonant_analysis = self.analyzer.analyze_consonance(self.dissonant_audio, self.sr)

        # Consonant audio should have higher consonance score
        self.assertGreater(consonant_analysis['consonance_score'], dissonant_analysis['consonance_score'])

        # Dissonant audio should have higher dissonance score
        self.assertGreater(dissonant_analysis['dissonance_score'], consonant_analysis['dissonance_score'])

    def test_analyze_harmonic_rhythm(self):
        """Test harmonic rhythm analysis."""
        # Create audio with changing harmony
        duration = 8.0
        t = np.linspace(0, duration, int(self.sr * duration))

        # First half: C major chord
        chord1_mask = t < duration / 2
        chord1 = np.zeros_like(t)
        chord1[chord1_mask] = (
            np.sin(2 * np.pi * 261.63 * t[chord1_mask])  # C
            + np.sin(2 * np.pi * 329.63 * t[chord1_mask])  # E
            + np.sin(2 * np.pi * 392.00 * t[chord1_mask])  # G
        )

        # Second half: F major chord
        chord2_mask = t >= duration / 2
        chord2 = np.zeros_like(t)
        chord2[chord2_mask] = (
            np.sin(2 * np.pi * 174.61 * t[chord2_mask])  # F
            + np.sin(2 * np.pi * 220.00 * t[chord2_mask])  # A
            + np.sin(2 * np.pi * 261.63 * t[chord2_mask])  # C
        )

        changing_harmony = chord1 + chord2

        harmonic_rhythm = self.analyzer.analyze_harmonic_rhythm(changing_harmony, self.sr)

        # Check required fields
        expected_fields = ['harmonic_rhythm', 'chord_change_rate', 'harmonic_stability']
        for field in expected_fields:
            self.assertIn(field, harmonic_rhythm)
            self.assertIsInstance(harmonic_rhythm[field], (int, float))
            self.assertGreaterEqual(harmonic_rhythm[field], 0.0)

        # Test with static harmony
        static_rhythm = self.analyzer.analyze_harmonic_rhythm(self.harmonic_audio, self.sr)

        # Changing harmony should have higher change rate
        self.assertGreater(harmonic_rhythm['chord_change_rate'], static_rhythm['chord_change_rate'])

    def test_consonance_ratings_effect(self):
        """Test effect of different consonance ratings."""
        # Create custom consonance ratings
        high_consonance_ratings = {0: 1.0, 7: 0.9, 4: 0.8, 3: 0.7}
        low_consonance_ratings = {0: 0.5, 7: 0.4, 4: 0.3, 3: 0.2}

        high_analyzer = HarmonyAnalyzer(consonance_ratings=high_consonance_ratings)
        low_analyzer = HarmonyAnalyzer(consonance_ratings=low_consonance_ratings)

        # Test with same audio
        high_consonance = high_analyzer.analyze_consonance(self.harmonic_audio, self.sr)
        low_consonance = low_analyzer.analyze_consonance(self.harmonic_audio, self.sr)

        # High consonance ratings should result in higher consonance scores
        self.assertGreater(high_consonance['consonance_score'], low_consonance['consonance_score'])

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])

        try:
            complexity = self.analyzer.analyze_harmony_complexity(empty_audio, self.sr)
            # Should handle empty input gracefully
            self.assertIsInstance(complexity, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    def test_short_audio_handling(self):
        """Test handling of very short audio."""
        # Create 0.1 second audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(self.sr * 0.1)))

        complexity = self.analyzer.analyze_harmony_complexity(short_audio, self.sr)

        # Should handle short audio
        self.assertIsInstance(complexity, dict)
        self.assertIn('harmonic_complexity', complexity)

    def test_noise_audio_handling(self):
        """Test handling of noisy audio."""
        # Create white noise
        noise_audio = 0.1 * np.random.randn(self.sr * 2)

        noise_analysis = self.analyzer.analyze_consonance(noise_audio, self.sr)

        # Should handle noise gracefully
        self.assertIsInstance(noise_analysis, dict)

        # Noise should have lower consonance than harmonic content
        # Note: Random noise can sometimes appear consonant due to statistical variation
        self.assertLess(noise_analysis['consonance_score'], 0.95)

    def test_silence_handling(self):
        """Test handling of silent audio."""
        silent_audio = np.zeros(self.sr * 2)

        silence_analysis = self.analyzer.analyze_harmony_complexity(silent_audio, self.sr)

        # Should handle silence gracefully
        self.assertIsInstance(silence_analysis, dict)

        # Silence should have minimal complexity
        self.assertLess(silence_analysis['harmonic_complexity'], 0.1)

    def test_different_hop_lengths(self):
        """Test analyzer with different hop lengths."""
        # Test with smaller hop length
        small_hop_analyzer = HarmonyAnalyzer(hop_length=256)
        small_hop_result = small_hop_analyzer.analyze_harmony_complexity(self.harmonic_audio, self.sr)

        # Test with larger hop length
        large_hop_analyzer = HarmonyAnalyzer(hop_length=1024)
        large_hop_result = large_hop_analyzer.analyze_harmony_complexity(self.harmonic_audio, self.sr)

        # Both should produce valid results
        self.assertIsInstance(small_hop_result, dict)
        self.assertIsInstance(large_hop_result, dict)

        # Both should have complexity scores
        self.assertIn('harmonic_complexity', small_hop_result)
        self.assertIn('harmonic_complexity', large_hop_result)

    def test_harmonic_change_detection(self):
        """Test detection of harmonic changes."""
        # Create audio with clear harmonic change
        duration = 6.0
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create three different chords
        chord_duration = duration / 3

        # C major
        mask1 = t < chord_duration
        # F major
        mask2 = (t >= chord_duration) & (t < 2 * chord_duration)
        # G major
        mask3 = t >= 2 * chord_duration

        changing_audio = np.zeros_like(t)

        # Add C major chord
        changing_audio[mask1] += (
            np.sin(2 * np.pi * 261.63 * t[mask1])
            + np.sin(2 * np.pi * 329.63 * t[mask1])
            + np.sin(2 * np.pi * 392.00 * t[mask1])
        )

        # Add F major chord
        changing_audio[mask2] += (
            np.sin(2 * np.pi * 174.61 * t[mask2])
            + np.sin(2 * np.pi * 220.00 * t[mask2])
            + np.sin(2 * np.pi * 261.63 * t[mask2])
        )

        # Add G major chord
        changing_audio[mask3] += (
            np.sin(2 * np.pi * 196.00 * t[mask3])
            + np.sin(2 * np.pi * 246.94 * t[mask3])
            + np.sin(2 * np.pi * 293.66 * t[mask3])
        )

        harmonic_rhythm = self.analyzer.analyze_harmonic_rhythm(changing_audio, self.sr)

        # Should detect harmonic changes
        self.assertGreater(harmonic_rhythm['chord_change_rate'], 0.5)
        self.assertLess(harmonic_rhythm['harmonic_stability'], 0.99)  # More realistic threshold

    def test_interval_consonance_calculation(self):
        """Test interval consonance calculation."""
        # Create perfect fifth (highly consonant)
        fifth_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, self.sr * 2)) + np.sin(
            2 * np.pi * 660 * np.linspace(0, 2, self.sr * 2)
        )  # 3:2 ratio

        # Create tritone (dissonant)
        tritone_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, self.sr * 2)) + np.sin(
            2 * np.pi * 622 * np.linspace(0, 2, self.sr * 2)
        )  # Tritone

        fifth_consonance = self.analyzer.analyze_consonance(fifth_audio, self.sr)
        tritone_consonance = self.analyzer.analyze_consonance(tritone_audio, self.sr)

        # Perfect fifth should be more consonant than tritone
        self.assertGreater(fifth_consonance['interval_consonance'], tritone_consonance['interval_consonance'])


if __name__ == '__main__':
    unittest.main()
