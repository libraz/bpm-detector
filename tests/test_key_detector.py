"""Tests for key detector module."""

import unittest
from unittest.mock import patch

import numpy as np

from src.bpm_detector.key_detector import KeyDetector


class TestKeyDetector(unittest.TestCase):
    """Test cases for KeyDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = KeyDetector(hop_length=512)
        self.sr = 22050

        # Create test audio for C major
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # C major scale: C-D-E-F-G-A-B-C
        c_major_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        self.c_major_audio = np.zeros_like(t)

        note_duration = duration / len(c_major_freqs)
        for i, freq in enumerate(c_major_freqs):
            start_idx = int(i * note_duration * self.sr)
            end_idx = int((i + 1) * note_duration * self.sr)
            if end_idx <= len(t):
                self.c_major_audio[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

        # Create test audio for A minor (relative minor of C major)
        a_minor_freqs = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00]
        self.a_minor_audio = np.zeros_like(t)

        for i, freq in enumerate(a_minor_freqs):
            start_idx = int(i * note_duration * self.sr)
            end_idx = int((i + 1) * note_duration * self.sr)
            if end_idx <= len(t):
                self.a_minor_audio[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    def test_initialization(self):
        """Test detector initialization."""
        # Test with default parameters
        default_detector = KeyDetector()
        self.assertEqual(default_detector.hop_length, 512)

        # Test with custom parameters
        custom_detector = KeyDetector(hop_length=256)
        self.assertEqual(custom_detector.hop_length, 256)

    @patch('librosa.feature.chroma_stft')
    def test_detect_key_c_major(self, mock_chroma):
        """Test key detection for C major."""
        # Mock chroma features for C major
        # C major should have strong C, E, G components
        mock_chroma_data = np.zeros((12, 100))
        mock_chroma_data[0, :] = 1.0  # C
        mock_chroma_data[4, :] = 0.8  # E
        mock_chroma_data[7, :] = 0.6  # G
        mock_chroma.return_value = mock_chroma_data

        result = self.detector.detect_key(self.c_major_audio, self.sr, _feature_backend='stft')

        # Check result structure
        self.assertIsInstance(result, dict)
        required_fields = ['key', 'mode', 'confidence']
        for field in required_fields:
            self.assertIn(field, result)

        # Check types
        self.assertIsInstance(result['key'], str)
        self.assertIsInstance(result['mode'], str)
        self.assertIsInstance(result['confidence'], (int, float))

        # Check confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 100.0)

        # Should detect C major
        self.assertEqual(result['key'], 'C')
        self.assertEqual(result['mode'], 'Major')

    @patch('librosa.feature.chroma_stft')
    def test_detect_key_a_minor(self, mock_chroma):
        """Test key detection for A minor."""
        # Mock chroma features for A minor
        # A minor should have strong A, C, E components
        mock_chroma_data = np.zeros((12, 100))
        mock_chroma_data[9, :] = 1.0  # A
        mock_chroma_data[0, :] = 0.8  # C
        mock_chroma_data[4, :] = 0.6  # E
        mock_chroma.return_value = mock_chroma_data

        result = self.detector.detect_key(self.a_minor_audio, self.sr, _feature_backend='stft')

        # Should detect A minor
        self.assertEqual(result['key'], 'A')
        self.assertEqual(result['mode'], 'Minor')

    def test_apply_audio_filters(self):
        """Test audio filtering."""
        filtered_audio = self.detector._apply_audio_filters(self.c_major_audio, self.sr)

        # Should return filtered audio of same length
        self.assertEqual(len(filtered_audio), len(self.c_major_audio))
        self.assertIsInstance(filtered_audio, np.ndarray)

    @patch('librosa.feature.chroma_stft')
    def test_extract_chroma_features(self, mock_chroma):
        """Test chroma feature extraction."""
        # Mock librosa chroma output
        mock_chroma_data = np.random.rand(12, 100)
        mock_chroma.return_value = mock_chroma_data

        chroma = self.detector._extract_chroma_features(self.c_major_audio, self.sr, backend='stft')

        # Should return chroma features
        self.assertIsInstance(chroma, np.ndarray)
        self.assertEqual(chroma.shape[0], 12)  # 12 pitch classes

        # Check that librosa was called correctly
        mock_chroma.assert_called_once()

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])

        try:
            result = self.detector.detect_key(empty_audio, self.sr, _feature_backend='stft')
            # Should handle empty input gracefully
            self.assertIsInstance(result, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    def test_short_audio_handling(self):
        """Test handling of very short audio."""
        # Create 0.1 second audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(self.sr * 0.1)))

        result = self.detector.detect_key(short_audio, self.sr, _feature_backend='stft')

        # Should handle short audio
        self.assertIsInstance(result, dict)
        self.assertIn('key', result)
        self.assertIn('mode', result)
        self.assertIn('confidence', result)

    def test_noise_audio_handling(self):
        """Test handling of noisy audio."""
        # Create white noise
        noise_audio = 0.1 * np.random.randn(self.sr * 2)

        result = self.detector.detect_key(noise_audio, self.sr, _feature_backend='stft')

        # Should handle noise gracefully
        self.assertIsInstance(result, dict)

        # Confidence should be low for noise
        self.assertLess(result['confidence'], 50.0)

    def test_silence_handling(self):
        """Test handling of silent audio."""
        silent_audio = np.zeros(self.sr * 2)

        result = self.detector.detect_key(silent_audio, self.sr, _feature_backend='stft')

        # Should handle silence gracefully
        self.assertIsInstance(result, dict)

        # Confidence should be very low for silence
        self.assertLess(result['confidence'], 30.0)

    def test_key_detection_with_hints(self):
        """Test key detection with external hints."""
        # Test with correct hint
        result_with_hint = self.detector.detect_key(
            self.c_major_audio, self.sr, external_key_hint='C Major', _feature_backend='stft'
        )

        # Test without hint
        result_without_hint = self.detector.detect_key(self.c_major_audio, self.sr, _feature_backend='stft')

        # Both should return valid results
        self.assertIsInstance(result_with_hint, dict)
        self.assertIsInstance(result_without_hint, dict)

        # Hint should potentially improve confidence
        if result_with_hint['key'] == 'C' and result_with_hint['mode'] == 'Major':
            self.assertGreaterEqual(
                result_with_hint['confidence'], result_without_hint['confidence'] * 0.9  # Allow some tolerance
            )

    def test_different_hop_lengths(self):
        """Test detector with different hop lengths."""
        # Test with smaller hop length
        small_hop_detector = KeyDetector(hop_length=256)
        small_hop_result = small_hop_detector.detect_key(self.c_major_audio, self.sr, _feature_backend='stft')

        # Test with larger hop length
        large_hop_detector = KeyDetector(hop_length=1024)
        large_hop_result = large_hop_detector.detect_key(self.c_major_audio, self.sr, _feature_backend='stft')

        # Both should produce valid results
        self.assertIsInstance(small_hop_result, dict)
        self.assertIsInstance(large_hop_result, dict)

        # Both should detect the same key (C major)
        self.assertEqual(small_hop_result['key'], 'C')
        self.assertEqual(large_hop_result['key'], 'C')

    def test_relative_major_minor_detection(self):
        """Test detection of relative major/minor keys."""
        # C major and A minor share the same notes
        c_major_result = self.detector.detect_key(self.c_major_audio, self.sr, _feature_backend='stft')
        a_minor_result = self.detector.detect_key(self.a_minor_audio, self.sr, _feature_backend='stft')

        # Should detect different modes
        self.assertNotEqual(
            (c_major_result['key'], c_major_result['mode']), (a_minor_result['key'], a_minor_result['mode'])
        )

        # Should have reasonable confidence for both
        self.assertGreater(c_major_result['confidence'], 30.0)
        self.assertGreater(a_minor_result['confidence'], 30.0)

    def test_detect_method_compatibility(self):
        """Test the detect method for backward compatibility."""
        # detect method doesn't support _feature_backend, so we test it as-is
        key, confidence = self.detector.detect(self.c_major_audio, self.sr)

        self.assertIsInstance(key, str)
        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)


if __name__ == '__main__':
    unittest.main()
