"""Tests for instrument classifier module."""

import unittest

import numpy as np

from src.bpm_detector.instrument_classifier import InstrumentClassifier


class TestInstrumentClassifier(unittest.TestCase):
    """Test cases for InstrumentClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = InstrumentClassifier(hop_length=512, n_fft=2048)
        self.sr = 22050

        # Create test audio
        duration = 2  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_classify_instruments(self):
        """Test instrument classification."""
        instruments = self.classifier.classify_instruments(self.test_audio, self.sr)

        # Should return a list of instrument detections
        self.assertIsInstance(instruments, list)

        # Each detection should have required fields
        for instrument in instruments:
            self.assertIsInstance(instrument, dict)
            required_fields = ['instrument', 'confidence', 'prominence']
            for field in required_fields:
                self.assertIn(field, instrument)

            # Check field types and ranges
            self.assertIsInstance(instrument['instrument'], str)
            self.assertIsInstance(instrument['confidence'], (int, float))
            self.assertIsInstance(instrument['prominence'], (int, float))

            self.assertGreaterEqual(instrument['confidence'], 0.0)
            self.assertLessEqual(instrument['confidence'], 1.0)
            self.assertGreaterEqual(instrument['prominence'], 0.0)
            self.assertLessEqual(instrument['prominence'], 1.0)

    def test_filter_redundant_instruments(self):
        """Test redundant instrument filtering."""
        # Create test instruments with overlaps
        test_instruments = [
            {'instrument': 'piano', 'confidence': 0.9, 'prominence': 0.8},
            {'instrument': 'piano', 'confidence': 0.7, 'prominence': 0.6},  # Duplicate
            {'instrument': 'guitar', 'confidence': 0.8, 'prominence': 0.7},
            {'instrument': 'drums', 'confidence': 0.6, 'prominence': 0.5},
        ]

        filtered = self.classifier._filter_redundant_instruments(test_instruments)

        # Should remove duplicate piano entry
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(test_instruments))

        # Should keep the higher confidence piano
        piano_entries = [inst for inst in filtered if inst['instrument'] == 'piano']
        if len(piano_entries) > 0:
            self.assertEqual(len(piano_entries), 1)
            self.assertEqual(piano_entries[0]['confidence'], 0.9)

    def test_calculate_instrument_confidence(self):
        """Test instrument confidence calculation."""
        # Create test magnitude spectrum
        freqs = np.linspace(0, self.sr / 2, 1024)
        magnitude = np.exp(-freqs / 1000)  # Decaying spectrum

        # Test piano frequency range
        confidence = self.classifier._calculate_instrument_confidence(
            magnitude, freqs, freq_range=(200, 2000), instrument='piano'
        )

        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_instrument_prominence(self):
        """Test instrument prominence calculation."""
        # Create test magnitude spectrum
        freqs = np.linspace(0, self.sr / 2, 1024)
        magnitude = np.ones_like(freqs)
        magnitude[100:200] = 5.0  # Peak in specific range

        prominence = self.classifier._calculate_instrument_prominence(
            magnitude, freqs, freq_range=(freqs[100], freqs[200])
        )

        self.assertIsInstance(prominence, (int, float))
        self.assertGreaterEqual(prominence, 0.0)
        self.assertLessEqual(prominence, 1.0)

        # Should be high due to the peak in the frequency range
        self.assertGreater(prominence, 0.5)

    def test_instrument_ranges(self):
        """Test that instrument ranges are properly defined."""
        self.assertIsInstance(self.classifier.INSTRUMENT_RANGES, dict)

        for instrument, freq_range in self.classifier.INSTRUMENT_RANGES.items():
            self.assertIsInstance(instrument, str)
            self.assertIsInstance(freq_range, tuple)
            self.assertEqual(len(freq_range), 2)
            self.assertLess(freq_range[0], freq_range[1])  # Low < High


if __name__ == '__main__':
    unittest.main()
