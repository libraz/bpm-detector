"""Tests for key validation module."""

import unittest
import numpy as np
from src.bpm_detector.key_validation import KeyValidator, JPOPKeyDetector


class TestKeyValidator(unittest.TestCase):
    """Test cases for KeyValidator."""

    def test_validate_relative_keys(self):
        """Test relative key validation."""
        # Test C major vs A minor validation
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # C
        chroma_mean[4] = 0.8  # E
        chroma_mean[7] = 0.6  # G
        chroma_mean[9] = 0.7  # A
        
        validated_key, validated_mode, confidence = KeyValidator.validate_relative_keys(
            'C', 'Major', 0.8, chroma_mean
        )
        
        # Should return validation results
        self.assertIsInstance(validated_key, str)
        self.assertIsInstance(validated_mode, str)
        self.assertIsInstance(confidence, (int, float))

    def test_analyze_major_tendency(self):
        """Test major key tendency analysis."""
        # Create chroma with major characteristics
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # Tonic
        chroma_mean[4] = 0.8  # Major third
        chroma_mean[7] = 0.6  # Perfect fifth
        
        major_tendency = KeyValidator._analyze_major_tendency(chroma_mean, 0)  # C major
        
        self.assertIsInstance(major_tendency, (int, float))
        self.assertGreaterEqual(major_tendency, 0.0)
        self.assertLessEqual(major_tendency, 1.0)

    def test_analyze_minor_tendency(self):
        """Test minor key tendency analysis."""
        # Create chroma with minor characteristics
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # Tonic
        chroma_mean[3] = 0.8  # Minor third
        chroma_mean[7] = 0.6  # Perfect fifth
        
        minor_tendency = KeyValidator._analyze_minor_tendency(chroma_mean, 0)  # C minor
        
        self.assertIsInstance(minor_tendency, (int, float))
        self.assertGreaterEqual(minor_tendency, 0.0)
        self.assertLessEqual(minor_tendency, 1.0)


class TestJPOPKeyDetector(unittest.TestCase):
    """Test cases for JPOPKeyDetector."""

    def test_detect_jpop_keys(self):
        """Test J-pop specific key detection."""
        # Create mock correlations
        correlations = [0.5] * 24  # 12 major + 12 minor keys
        correlations[0] = 0.9  # High correlation for C major
        
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # Strong C
        
        jpop_result = JPOPKeyDetector.detect_jpop_keys(chroma_mean, correlations, enable_jpop=True)
        
        # Should return key detection result
        self.assertIsInstance(jpop_result, tuple)
        self.assertEqual(len(jpop_result), 3)  # (key, mode, confidence)
        
        key, mode, confidence = jpop_result
        self.assertIsInstance(key, str)
        self.assertIsInstance(mode, str)
        self.assertIsInstance(confidence, (int, float))

    def test_gsharp_minor_detection(self):
        """Test G# minor specific detection."""
        # Create chroma pattern for G# minor
        chroma_mean = np.zeros(12)
        chroma_mean[8] = 1.0   # G#
        chroma_mean[6] = 0.8   # F#
        chroma_mean[4] = 0.7   # E
        chroma_mean[1] = 0.6   # C#
        
        correlations = [0.3] * 24
        key_names = [f"Note{i} Major" for i in range(12)] + [f"Note{i} Minor" for i in range(12)]
        
        result = JPOPKeyDetector.detect_jpop_keys(chroma_mean, correlations, key_names=key_names)
        
        key, mode, confidence = result
        self.assertIsInstance(key, str)
        self.assertIsInstance(mode, str)
        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)


if __name__ == '__main__':
    unittest.main()