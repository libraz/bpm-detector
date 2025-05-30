"""Tests for music theory module."""

import unittest
import numpy as np
from src.bpm_detector.music_theory import midi_to_note_name, classify_vocal_range


class TestMusicTheory(unittest.TestCase):
    """Test cases for music theory utilities."""

    def test_midi_to_note_name_basic(self):
        """Test basic MIDI to note name conversion."""
        # Test standard notes
        self.assertEqual(midi_to_note_name(60), 'C4')  # Middle C
        self.assertEqual(midi_to_note_name(69), 'A4')  # A440
        self.assertEqual(midi_to_note_name(72), 'C5')  # C above middle C
        
        # Test chromatic notes
        self.assertEqual(midi_to_note_name(61), 'C#4')  # C sharp
        self.assertEqual(midi_to_note_name(70), 'A#4')  # A sharp
        self.assertEqual(midi_to_note_name(59), 'B3')   # B below middle C

    def test_midi_to_note_name_edge_cases(self):
        """Test edge cases for MIDI to note name conversion."""
        # Test very low notes
        self.assertEqual(midi_to_note_name(0), 'C-1')
        self.assertEqual(midi_to_note_name(12), 'C0')
        self.assertEqual(midi_to_note_name(24), 'C1')
        
        # Test very high notes
        self.assertEqual(midi_to_note_name(108), 'C8')
        self.assertEqual(midi_to_note_name(120), 'C9')
        self.assertEqual(midi_to_note_name(127), 'G9')

    def test_midi_to_note_name_all_chromatic(self):
        """Test all chromatic notes in an octave."""
        expected_notes = [
            'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4',
            'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'
        ]
        
        for i, expected in enumerate(expected_notes):
            midi_note = 60 + i  # Starting from middle C
            self.assertEqual(midi_to_note_name(midi_note), expected)

    def test_midi_to_note_name_octave_consistency(self):
        """Test octave consistency across different octaves."""
        # Test C notes across octaves
        c_notes = [
            (0, 'C-1'), (12, 'C0'), (24, 'C1'), (36, 'C2'),
            (48, 'C3'), (60, 'C4'), (72, 'C5'), (84, 'C6'),
            (96, 'C7'), (108, 'C8')
        ]
        
        for midi, expected in c_notes:
            self.assertEqual(midi_to_note_name(midi), expected)

    def test_midi_to_note_name_float_input(self):
        """Test MIDI to note name with float input."""
        # Test with float values (should round to nearest integer)
        self.assertEqual(midi_to_note_name(60.0), 'C4')
        self.assertEqual(midi_to_note_name(60.4), 'C4')  # Should round down
        self.assertEqual(midi_to_note_name(60.6), 'C#4')  # Should round up
        self.assertEqual(midi_to_note_name(69.5), 'A#4')  # Should round up

    def test_midi_to_note_name_negative_input(self):
        """Test MIDI to note name with negative input."""
        # Test negative MIDI numbers
        self.assertEqual(midi_to_note_name(-12), 'C-2')
        self.assertEqual(midi_to_note_name(-1), 'B-2')
        self.assertEqual(midi_to_note_name(-24), 'C-3')

    def test_classify_vocal_range_soprano(self):
        """Test soprano range classification."""
        # Typical soprano range: C4 (MIDI 60) to C6 (MIDI 84)
        soprano_classification = classify_vocal_range(60, 84)
        self.assertEqual(soprano_classification, 'Soprano')
        
        # High soprano
        high_soprano = classify_vocal_range(65, 86)  # F4 to D6
        self.assertEqual(high_soprano, 'Soprano')

    def test_classify_vocal_range_alto(self):
        """Test alto range classification."""
        # Typical alto range: G3 (MIDI 55) to G5 (MIDI 79)
        alto_classification = classify_vocal_range(55, 79)
        self.assertEqual(alto_classification, 'Alto')
        
        # Mezzo-soprano (overlaps with alto)
        mezzo_classification = classify_vocal_range(57, 81)  # A3 to A5
        self.assertIn(mezzo_classification, ['Alto', 'Mezzo-Soprano'])

    def test_classify_vocal_range_tenor(self):
        """Test tenor range classification."""
        # Typical tenor range: C3 (MIDI 48) to C5 (MIDI 72)
        tenor_classification = classify_vocal_range(48, 72)
        self.assertEqual(tenor_classification, 'Tenor')
        
        # High tenor (may be classified as Alto due to overlap)
        high_tenor = classify_vocal_range(50, 74)  # D3 to D5
        self.assertIn(high_tenor, ['Tenor', 'Alto'])

    def test_classify_vocal_range_bass(self):
        """Test bass range classification."""
        # Typical bass range: E2 (MIDI 40) to E4 (MIDI 64)
        bass_classification = classify_vocal_range(40, 64)
        self.assertEqual(bass_classification, 'Bass')
        
        # Baritone (overlaps with bass)
        baritone_classification = classify_vocal_range(43, 67)  # G2 to G4
        self.assertIn(baritone_classification, ['Bass', 'Baritone'])

    def test_classify_vocal_range_extreme_ranges(self):
        """Test classification of extreme vocal ranges."""
        # Very wide range (should classify as instrumental)
        very_wide = classify_vocal_range(30, 90)  # 5 octaves
        self.assertEqual(very_wide, 'Instrumental (Wide Range)')
        
        # Very narrow range
        very_narrow = classify_vocal_range(69, 70)  # A4 to A#4 (semitone)
        self.assertIsInstance(very_narrow, str)

    def test_classify_vocal_range_non_vocal_frequencies(self):
        """Test classification with non-vocal frequencies."""
        # Very low frequencies (sub-bass)
        very_low = classify_vocal_range(20, 35)
        self.assertEqual(very_low, 'Instrumental (Extreme Range)')
        
        # Very high frequencies (above vocal range)
        very_high = classify_vocal_range(100, 120)
        self.assertEqual(very_high, 'Instrumental (Extreme Range)')

    def test_classify_vocal_range_edge_cases(self):
        """Test edge cases for vocal range classification."""
        # Same frequency for lowest and highest
        same_freq = classify_vocal_range(440.00, 440.00)
        self.assertIsInstance(same_freq, str)
        
        # Reversed frequencies (highest < lowest)
        try:
            reversed_freq = classify_vocal_range(880.00, 440.00)
            self.assertIsInstance(reversed_freq, str)
        except ValueError:
            # It's acceptable to raise an error for invalid input
            pass

    def test_classify_vocal_range_boundary_cases(self):
        """Test boundary cases between vocal ranges."""
        # Boundary between bass and tenor
        bass_tenor_boundary = classify_vocal_range(45, 69)  # A2 to A4
        self.assertIn(bass_tenor_boundary, ['Bass', 'Tenor', 'Baritone'])
        
        # Boundary between alto and soprano
        alto_soprano_boundary = classify_vocal_range(55, 81)  # G3 to A5
        self.assertIn(alto_soprano_boundary, ['Alto', 'Soprano', 'Mezzo-Soprano'])

    def test_classify_vocal_range_instrumental_ranges(self):
        """Test classification with instrumental frequency ranges."""
        # Piano range (A0 to C8) - should be instrumental
        piano_range = classify_vocal_range(21, 108)
        self.assertEqual(piano_range, 'Instrumental (Wide Range)')
        
        # Guitar range (E2 to E6) - should be instrumental
        guitar_range = classify_vocal_range(40, 88)
        self.assertEqual(guitar_range, 'Instrumental (Wide Range)')
        
        # Violin range (G3 to E7) - should be instrumental
        violin_range = classify_vocal_range(55, 100)
        self.assertEqual(violin_range, 'Instrumental (Wide Range)')

    def test_frequency_to_midi_consistency(self):
        """Test consistency between frequency and MIDI conversions."""
        # Test known frequency-MIDI pairs
        known_pairs = [
            (261.63, 60),   # C4
            (440.00, 69),   # A4
            (523.25, 72),   # C5
            (880.00, 81),   # A5
        ]
        
        for freq, expected_midi in known_pairs:
            # Convert frequency to MIDI (approximate)
            calculated_midi = 69 + 12 * np.log2(freq / 440.0)
            
            # Should be close to expected MIDI note
            self.assertAlmostEqual(calculated_midi, expected_midi, places=1)
            
            # Convert back to note name
            note_name = midi_to_note_name(round(calculated_midi))
            self.assertIsInstance(note_name, str)

    def test_note_name_format_consistency(self):
        """Test note name format consistency."""
        # Test that all note names follow expected format
        for midi in range(0, 128):
            note_name = midi_to_note_name(midi)
            
            # Should be a string
            self.assertIsInstance(note_name, str)
            
            # Should have at least 2 characters (note + octave)
            self.assertGreaterEqual(len(note_name), 2)
            
            # Should start with a note letter
            self.assertIn(note_name[0], 'ABCDEFG')
            
            # Should end with octave number (possibly negative)
            self.assertTrue(note_name[-1].isdigit() or note_name[-2:] == '-1')

    def test_vocal_range_classification_consistency(self):
        """Test vocal range classification consistency."""
        # Test that classification is consistent for overlapping ranges
        test_ranges = [
            (41, 65),   # Bass range
            (45, 69),   # Bass-Tenor overlap
            (48, 72),   # Tenor range
            (53, 77),   # Tenor-Alto overlap
            (57, 81),   # Alto range
            (60, 84),   # Alto-Soprano overlap
        ]
        
        for low, high in test_ranges:
            classification = classify_vocal_range(low, high)
            
            # Should return a valid classification
            self.assertIsInstance(classification, str)
            self.assertGreater(len(classification), 0)
            
            # Should be one of the expected vocal ranges
            expected_ranges = [
                'Bass', 'Baritone', 'Tenor', 'Alto',
                'Mezzo-Soprano', 'Soprano', 'Instrumental',
                'Unknown'
            ]
            # Allow for variations in naming
            is_valid = any(expected in classification for expected in expected_ranges)
            self.assertTrue(is_valid, f"Unexpected classification: {classification}")


if __name__ == '__main__':
    unittest.main()