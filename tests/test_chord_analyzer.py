"""Tests for chord progression analyzer."""

import unittest
from unittest.mock import patch

import numpy as np

from src.bpm_detector.chord_analyzer import ChordProgressionAnalyzer


class TestChordProgressionAnalyzer(unittest.TestCase):
    """Test cases for ChordProgressionAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ChordProgressionAnalyzer()
        self.sr = 22050
        self.duration = 10.0  # 10 seconds

        # Create a simple test signal with known chord progression
        t = np.linspace(0, self.duration, int(self.sr * self.duration))

        # C major chord (C-E-G: 261.63, 329.63, 392.00 Hz)
        c_major = (
            0.3 * np.sin(2 * np.pi * 261.63 * t)
            + 0.3 * np.sin(2 * np.pi * 329.63 * t)
            + 0.3 * np.sin(2 * np.pi * 392.00 * t)
        )

        # Add some noise for realism
        self.test_signal = c_major + 0.05 * np.random.randn(len(t))

    def test_extract_chroma_features(self):
        """Test chroma feature extraction."""
        chroma = self.analyzer.extract_chroma_features(self.test_signal, self.sr)

        # Check shape
        self.assertEqual(chroma.shape[0], 12)  # 12 chroma bins
        self.assertGreater(chroma.shape[1], 0)  # Some time frames

        # Check that values are normalized (0-1 range)
        self.assertTrue(np.all(chroma >= 0))
        self.assertTrue(np.all(chroma <= 1))

    def test_detect_chords(self):
        """Test chord detection."""
        chroma = self.analyzer.extract_chroma_features(self.test_signal, self.sr)
        chords = self.analyzer.detect_chords(chroma)

        # Should detect at least one chord
        self.assertGreater(len(chords), 0)

        # Check chord format
        for chord in chords:
            self.assertEqual(len(chord), 4)  # (name, confidence, start, end)
            self.assertIsInstance(chord[0], str)  # chord name
            self.assertIsInstance(chord[1], float)  # confidence
            self.assertIsInstance(chord[2], int)  # start frame
            self.assertIsInstance(chord[3], int)  # end frame

    def test_analyze_progression(self):
        """Test chord progression analysis."""
        # Create mock chord list
        mock_chords = [('C', 0.8, 0, 10), ('Am', 0.7, 10, 20), ('F', 0.9, 20, 30), ('G', 0.8, 30, 40)]

        result = self.analyzer.analyze_progression(mock_chords)

        # Check required keys
        required_keys = [
            'main_progression',
            'progression_pattern',
            'harmonic_rhythm',
            'chord_complexity',
            'unique_chords',
            'chord_changes',
        ]
        for key in required_keys:
            self.assertIn(key, result)

        # Check types
        self.assertIsInstance(result['main_progression'], list)
        self.assertIsInstance(result['progression_pattern'], str)
        self.assertIsInstance(result['harmonic_rhythm'], float)
        self.assertIsInstance(result['chord_complexity'], float)
        self.assertIsInstance(result['unique_chords'], int)
        self.assertIsInstance(result['chord_changes'], int)

    def test_functional_analysis(self):
        """Test functional harmonic analysis."""
        chords = ['C', 'Am', 'F', 'G']
        key = 'C Major'

        roman_numerals = self.analyzer.functional_analysis(chords, key)

        # Should return roman numerals for each chord
        self.assertEqual(len(roman_numerals), len(chords))

        # Check expected roman numerals for C major
        expected = ['I', 'vi', 'IV', 'V']
        self.assertEqual(roman_numerals, expected)

    def test_detect_modulations(self):
        """Test modulation detection."""
        # Create mock chord progression with modulation
        mock_chords = [
            ('C', 0.8, 0, 10),
            ('Am', 0.7, 10, 20),
            ('F', 0.9, 20, 30),
            ('G', 0.8, 30, 40),
            ('D', 0.8, 40, 50),  # Modulation to G major
            ('Bm', 0.7, 50, 60),
            ('G', 0.9, 60, 70),
            ('A', 0.8, 70, 80),
        ]

        modulations = self.analyzer.detect_modulations(mock_chords, 'C Major', self.sr)

        # Should be a list
        self.assertIsInstance(modulations, list)

        # Each modulation should have required keys
        for mod in modulations:
            self.assertIn('time', mod)
            self.assertIn('from_key', mod)
            self.assertIn('to_key', mod)
            self.assertIn('confidence', mod)

    def test_analyze_complete(self):
        """Test complete chord analysis."""
        result = self.analyzer.analyze(self.test_signal, self.sr, 'C Major')

        # Check all required keys are present
        required_keys = [
            'chords',
            'main_progression',
            'progression_pattern',
            'harmonic_rhythm',
            'chord_complexity',
            'unique_chords',
            'chord_changes',
            'functional_analysis',
            'modulations',
            'substitute_chords_ratio',
        ]

        for key in required_keys:
            self.assertIn(key, result)

    def test_empty_input(self):
        """Test behavior with empty input."""
        empty_signal = np.array([])

        # Should handle empty input gracefully
        try:
            result = self.analyzer.analyze(empty_signal, self.sr)
            # Should return default values
            self.assertIsInstance(result, dict)
        except Exception:
            # Or raise appropriate exception
            pass

    def test_chord_template_matching(self):
        """Test chord template matching."""
        # Create a perfect C major chroma vector
        c_major_chroma = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)

        chord_name, confidence = self.analyzer._match_chord_template(c_major_chroma)

        # Should detect C major with high confidence
        self.assertEqual(chord_name, 'C')
        self.assertGreater(confidence, 0.5)

    def test_enhanced_chord_detection(self):
        """Test enhanced chord detection features."""
        # Create test audio with clear chord progression
        duration = 8.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))

        # Create C-Am-F-G progression
        chord_duration = duration / 4
        audio = np.zeros_like(t)

        # C major
        mask1 = t < chord_duration
        audio[mask1] = (
            np.sin(2 * np.pi * 261.63 * t[mask1])  # C
            + np.sin(2 * np.pi * 329.63 * t[mask1])  # E
            + np.sin(2 * np.pi * 392.00 * t[mask1])  # G
        )

        results = self.analyzer.analyze(audio, sr, key='C Major', bpm=120.0)

        # Check enhanced features
        self.assertIn('chord_changes', results)
        self.assertIn('functional_analysis', results)
        self.assertIn('modulations', results)

        # Check chord changes
        chord_changes = results['chord_changes']
        self.assertIsInstance(chord_changes, (int, float))
        self.assertGreaterEqual(chord_changes, 0)

    def test_chord_complexity_calculation(self):
        """Test chord complexity calculation."""
        # Test with different chord progressions
        simple_chords = [('C', 0.8, 0, 100), ('F', 0.8, 100, 200), ('G', 0.8, 200, 300)]

        simple_analysis = self.analyzer.analyze_progression(simple_chords)

        # Should have complexity score
        simple_complexity = simple_analysis.get('chord_complexity', 0)
        self.assertIsInstance(simple_complexity, (int, float))
        self.assertGreaterEqual(simple_complexity, 0.0)
        self.assertLessEqual(simple_complexity, 1.0)

    def test_substitute_chord_detection(self):
        """Test substitute chord detection."""
        # Test with chord progression that includes substitutes
        chord_names = ['C', 'A7', 'Dm', 'G7']  # A7 is a substitute for Am

        substitute_ratio = self.analyzer._calculate_substitute_ratio(chord_names, 'C Major')

        self.assertIsInstance(substitute_ratio, (int, float))
        self.assertGreaterEqual(substitute_ratio, 0.0)
        self.assertLessEqual(substitute_ratio, 1.0)

    def test_chord_clustering_and_merging(self):
        """Test chord clustering and merging functionality."""
        # Create chords with small gaps that should be merged
        fragmented_chords = [
            ('C', 0.9, 0, 90),
            ('C', 0.8, 95, 190),  # Small gap, should merge
            ('Am', 0.85, 200, 290),
            ('Am', 0.8, 295, 390),  # Small gap, should merge
        ]

        merged_chords = self.analyzer._merge_consecutive_chords(fragmented_chords)

        # Should have fewer chords after merging
        self.assertLessEqual(len(merged_chords), len(fragmented_chords))

        # Check that merged chords have reasonable durations
        for chord, confidence, start, end in merged_chords:
            self.assertLess(start, end)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_detect_chords_handles_short_c_and_am_chroma(self):
        """A chroma window shorter than the nominal bar window is still analyzed."""
        analyzer = ChordProgressionAnalyzer(hop_length=10)
        chord_vectors = {'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]}

        for expected_name, vector in chord_vectors.items():
            with self.subTest(chord=expected_name):
                chroma = np.repeat(np.asarray(vector, dtype=float)[:, None], 4, axis=1)
                chords = analyzer.detect_chords(chroma, bpm=120.0, sr=100)

                self.assertTrue(chords)
                self.assertEqual(chords[0][0], expected_name)
                self.assertEqual((chords[0][2], chords[0][3]), (0, 4))

    def test_detect_chords_includes_partial_tail_window(self):
        """The final partial window must contribute a chord detection."""
        analyzer = ChordProgressionAnalyzer(hop_length=10)
        c_major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
        a_minor = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=float)
        chroma = np.column_stack([*([c_major] * 20), *([a_minor] * 3)])

        chords = analyzer.detect_chords(chroma, bpm=120.0, sr=100)

        self.assertEqual([chord[0] for chord in chords], ['C', 'Am'])
        self.assertEqual((chords[0][2], chords[0][3]), (0, 20))
        self.assertEqual((chords[1][2], chords[1][3]), (20, 23))
        self.assertEqual(chords[-1][3], chroma.shape[1])

    def test_detect_chords_normalizes_overlapping_alternating_windows(self):
        """Overlapping evidence is clipped into chronological, bounded intervals."""
        analyzer = ChordProgressionAnalyzer(hop_length=10)
        chroma = np.ones((12, 100))
        alternating = [('C', 0.9), ('Am', 0.9)] * 5

        with patch.object(analyzer, '_match_chord_template', side_effect=alternating):
            chords = analyzer.detect_chords(chroma, bpm=120.0, sr=100)

        self.assertEqual(len(chords), len(alternating))
        self.assertTrue(all(0 <= start < end <= chroma.shape[1] for _, _, start, end in chords))
        self.assertTrue(all(previous[3] <= current[2] for previous, current in zip(chords, chords[1:])))
        self.assertEqual(chords[-1][3], chroma.shape[1])

        progression = analyzer.analyze_progression(chords, sr=100)
        self.assertAlmostEqual(progression['harmonic_rhythm'], 1.0)

    def test_detect_chords_handles_short_c_and_am_signals(self):
        """Short synthetic chord signals should reach the detector."""
        analyzer = ChordProgressionAnalyzer()
        sr = 22050
        t = np.arange(int(sr * 0.25)) / sr
        chord_frequencies = {'C': (261.63, 329.63, 392.0), 'Am': (220.0, 261.63, 329.63)}

        for expected_name, frequencies in chord_frequencies.items():
            with self.subTest(chord=expected_name):
                signal = sum(np.sin(2 * np.pi * frequency * t) for frequency in frequencies)
                chroma = analyzer.extract_chroma_features(signal, sr)
                chords = analyzer.detect_chords(chroma, bpm=120.0, sr=sr)

                self.assertTrue(chords)
                self.assertEqual(chords[0][0], expected_name)

    def test_functional_analysis_supports_flat_roots_and_compound_suffixes(self):
        """Flat spellings and major/minor suffixes map to the right degree."""
        chords = ['Ebmaj7', 'Bb7', 'Am7', 'Cmaj7']

        self.assertEqual(self.analyzer.functional_analysis(chords, 'C Major'), ['bIII', 'bVII', 'vi', 'I'])
        self.assertEqual(self.analyzer.functional_analysis(['Bbmaj7', 'Eb', 'F7'], 'Bb Major'), ['I', 'IV', 'V'])
        self.assertEqual(self.analyzer.functional_analysis(['Eb', 'Bb'], 'C Minor'), ['bIII', 'bVII'])

    def test_merge_consecutive_chords_uses_frame_gap(self):
        """A sample hop length must not bridge a large frame-index gap."""
        analyzer = ChordProgressionAnalyzer(hop_length=512)

        small_gap = [('C', 0.9, 0, 10), ('C', 0.8, 15, 20)]
        large_gap = [('C', 0.9, 0, 10), ('C', 0.8, 500, 510)]

        self.assertEqual(len(analyzer._merge_consecutive_chords(small_gap)), 1)
        self.assertEqual(len(analyzer._merge_consecutive_chords(large_gap)), 2)


if __name__ == '__main__':
    unittest.main()


def test_interval_normalization_drops_zero_length_segments():
    analyzer = ChordProgressionAnalyzer()
    source = [('C', 0.9, 0, 20), ('Am', 0.9, 0, 10)]
    result = analyzer._normalize_chord_intervals(source, 20)
    assert result
    assert all(start < end for _, _, start, end in result)


def test_invalid_interval_does_not_truncate_valid_chord():
    analyzer = ChordProgressionAnalyzer()
    result = analyzer._normalize_chord_intervals([('C', 0.9, 0, 20), ('Am', 0.9, 10, 10)], 20)
    assert result == [('C', 0.9, 0, 20)]
