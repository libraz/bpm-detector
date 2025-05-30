"""Tests for effects detector module."""

import unittest
import numpy as np
from src.bpm_detector.effects_detector import EffectsDetector


class TestEffectsDetector(unittest.TestCase):
    """Test cases for EffectsDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = EffectsDetector(hop_length=512)
        self.sr = 22050
        
        # Create test audio
        duration = 2  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_analyze_effects_usage(self):
        """Test effects usage analysis."""
        effects = self.detector.analyze_effects_usage(self.test_audio, self.sr)
        
        # Check required effects
        expected_effects = ['reverb', 'distortion', 'chorus', 'compression']
        for effect in expected_effects:
            self.assertIn(effect, effects)
            self.assertIsInstance(effects[effect], (int, float))
            self.assertGreaterEqual(effects[effect], 0.0)
            self.assertLessEqual(effects[effect], 1.0)

    def test_detect_reverb(self):
        """Test reverb detection."""
        # Test with dry audio
        dry_reverb = self.detector._detect_reverb(self.test_audio, self.sr)
        
        # Create audio with simulated reverb (add delayed copies)
        reverb_audio = self.test_audio.copy()
        delay_samples = int(0.1 * self.sr)  # 100ms delay
        if len(reverb_audio) > delay_samples:
            reverb_audio[delay_samples:] += 0.3 * reverb_audio[:-delay_samples]
        
        wet_reverb = self.detector._detect_reverb(reverb_audio, self.sr)
        
        # Both should be valid scores
        self.assertIsInstance(dry_reverb, (int, float))
        self.assertIsInstance(wet_reverb, (int, float))
        self.assertGreaterEqual(dry_reverb, 0.0)
        self.assertLessEqual(dry_reverb, 1.0)
        self.assertGreaterEqual(wet_reverb, 0.0)
        self.assertLessEqual(wet_reverb, 1.0)
        
        # Reverb audio should have higher reverb score
        self.assertGreater(wet_reverb, dry_reverb)

    def test_detect_distortion(self):
        """Test distortion detection."""
        # Test with clean audio
        clean_distortion = self.detector._detect_distortion(self.test_audio, self.sr)
        
        # Create distorted audio (clipping)
        distorted_audio = np.clip(3.0 * self.test_audio, -1.0, 1.0)
        dirty_distortion = self.detector._detect_distortion(distorted_audio, self.sr)
        
        # Both should be valid scores
        self.assertIsInstance(clean_distortion, (int, float))
        self.assertIsInstance(dirty_distortion, (int, float))
        self.assertGreaterEqual(clean_distortion, 0.0)
        self.assertLessEqual(clean_distortion, 1.0)
        self.assertGreaterEqual(dirty_distortion, 0.0)
        self.assertLessEqual(dirty_distortion, 1.0)
        
        # Distorted audio should have higher distortion score
        self.assertGreater(dirty_distortion, clean_distortion)

    def test_detect_chorus(self):
        """Test chorus detection."""
        chorus_score = self.detector._detect_chorus(self.test_audio, self.sr)
        
        self.assertIsInstance(chorus_score, (int, float))
        self.assertGreaterEqual(chorus_score, 0.0)
        self.assertLessEqual(chorus_score, 1.0)

    def test_detect_compression(self):
        """Test compression detection."""
        # Create more dynamic audio with varying amplitude
        t = np.linspace(0, 2, int(self.sr * 2))
        dynamic_audio = np.sin(2 * np.pi * 440 * t) * (0.1 + 0.9 * np.sin(2 * np.pi * 0.5 * t))
        dynamic_compression = self.detector._detect_compression(dynamic_audio)
        
        # Create heavily compressed audio (very uniform amplitude)
        compressed_audio = np.tanh(5.0 * dynamic_audio) * 0.8
        compressed_compression = self.detector._detect_compression(compressed_audio)
        
        # Both should be valid scores
        self.assertIsInstance(dynamic_compression, (int, float))
        self.assertIsInstance(compressed_compression, (int, float))
        self.assertGreaterEqual(dynamic_compression, 0.0)
        self.assertLessEqual(dynamic_compression, 1.0)
        self.assertGreaterEqual(compressed_compression, 0.0)
        self.assertLessEqual(compressed_compression, 1.0)
        
        # Compressed audio should have higher compression score
        self.assertGreater(compressed_compression, dynamic_compression)

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])
        
        try:
            effects = self.detector.analyze_effects_usage(empty_audio, self.sr)
            self.assertIsInstance(effects, dict)
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass

    def test_short_audio_handling(self):
        """Test handling of very short audio."""
        short_audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(self.sr * 0.1)))
        
        effects = self.detector.analyze_effects_usage(short_audio, self.sr)
        
        # Should handle short audio
        self.assertIsInstance(effects, dict)
        self.assertIn('reverb', effects)
        self.assertIn('compression', effects)


if __name__ == '__main__':
    unittest.main()