"""Tests for chord analysis module."""

import unittest
import numpy as np
from src.bpm_detector.chord_analysis import ChordProgressionAnalyzer


class TestChordProgressionAnalyzer(unittest.TestCase):
    """Test cases for ChordProgressionAnalyzer."""

    def test_validate_key_with_chord_analysis(self):
        """Test key validation with chord analysis."""
        # Create mock chroma mean for C major
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # C
        chroma_mean[4] = 0.8  # E
        chroma_mean[7] = 0.6  # G
        
        # Test validation
        validated_key, validated_mode, confidence = ChordProgressionAnalyzer.validate_key_with_chord_analysis(
            chroma_mean, 'C', 'Major', 0.8
        )
        
        # Should return validated results
        self.assertIsInstance(validated_key, str)
        self.assertIsInstance(validated_mode, str)
        self.assertIsInstance(confidence, (int, float))
        
        # Confidence should be reasonable
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_chord_driven_key_estimation(self):
        """Test chord-driven key estimation."""
        # Create chroma with clear chord implications
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # C
        chroma_mean[4] = 0.8  # E
        chroma_mean[7] = 0.6  # G
        
        key, mode, confidence = ChordProgressionAnalyzer.chord_driven_key_estimation(chroma_mean)
        
        # Should return estimation results
        self.assertIsInstance(key, str)
        self.assertIsInstance(mode, str)
        self.assertIsInstance(confidence, (int, float))
        
        # Confidence should be reasonable
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_analyze_minor_chord_progression(self):
        """Test minor chord progression analysis."""
        # Create chroma suggesting minor key
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # Tonic
        chroma_mean[3] = 0.8  # Minor third
        chroma_mean[7] = 0.6  # Perfect fifth
        
        minor_score = ChordProgressionAnalyzer._analyze_minor_chord_progression(chroma_mean, 0)
        
        self.assertIsInstance(minor_score, (int, float))
        self.assertGreaterEqual(minor_score, 0.0)

    def test_analyze_major_chord_progression(self):
        """Test major chord progression analysis."""
        # Create chroma suggesting major key
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 1.0  # Tonic
        chroma_mean[4] = 0.8  # Major third
        chroma_mean[7] = 0.6  # Perfect fifth
        
        major_score = ChordProgressionAnalyzer._analyze_major_chord_progression(chroma_mean, 0)
        
        self.assertIsInstance(major_score, (int, float))
        self.assertGreaterEqual(major_score, 0.0)


if __name__ == '__main__':
    unittest.main()