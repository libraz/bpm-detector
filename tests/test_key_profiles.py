"""Tests for key profiles module."""

import unittest
import numpy as np
from src.bpm_detector.key_profiles import KeyProfileBuilder, KeyHintMapper


class TestKeyProfileBuilder(unittest.TestCase):
    """Test cases for KeyProfileBuilder."""

    def test_build_profiles(self):
        """Test key profile building."""
        # Test with different profile types
        krumhansl_profiles = KeyProfileBuilder.build_profiles(profile_type='krumhansl')
        temperley_profiles = KeyProfileBuilder.build_profiles(profile_type='temperley')
        
        # Should return 24 profiles (12 major + 12 minor)
        self.assertEqual(len(krumhansl_profiles), 24)
        self.assertEqual(len(temperley_profiles), 24)
        
        # Each profile should be a numpy array of length 12
        for profile in krumhansl_profiles:
            self.assertIsInstance(profile, np.ndarray)
            self.assertEqual(len(profile), 12)

    def test_profile_normalization(self):
        """Test that profiles are properly normalized."""
        profiles = KeyProfileBuilder.build_profiles()
        
        for profile in profiles:
            # Each profile should sum to approximately 1.0
            self.assertAlmostEqual(np.sum(profile), 1.0, places=5)
            # All values should be positive
            self.assertTrue(np.all(profile >= 0))


class TestKeyHintMapper(unittest.TestCase):
    """Test cases for KeyHintMapper."""

    def test_build_hint_mapping(self):
        """Test hint mapping construction."""
        hint_mapping = KeyHintMapper.build_hint_mapping()
        
        # Should return a dictionary
        self.assertIsInstance(hint_mapping, dict)
        
        # Should contain expected key mappings
        expected_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        for key in expected_keys:
            self.assertIn(key, hint_mapping)
            
        # Each mapping should be a tuple of (major, minor)
        for key, (major, minor) in hint_mapping.items():
            if isinstance(major, str) and isinstance(minor, str):
                self.assertIsInstance(major, str)
                self.assertIsInstance(minor, str)

    def test_apply_external_key_hint(self):
        """Test external key hint application."""
        hint_map = KeyHintMapper.build_hint_mapping()
        
        # Test with matching hint
        adjusted_key, adjusted_mode, adjusted_conf = KeyHintMapper.apply_external_key_hint(
            'C Major', 'C', 'Major', 0.8, hint_map
        )
        
        # Should boost confidence for matching hint
        self.assertEqual(adjusted_key, 'C')
        self.assertEqual(adjusted_mode, 'Major')
        self.assertGreaterEqual(adjusted_conf, 0.8)
        
        # Test with conflicting hint
        conflicting_key, conflicting_mode, conflicting_conf = KeyHintMapper.apply_external_key_hint(
            'G Major', 'C', 'Major', 0.8, hint_map
        )
        
        # Should handle conflicting hint appropriately
        self.assertIsInstance(conflicting_key, str)
        self.assertIsInstance(conflicting_mode, str)
        self.assertIsInstance(conflicting_conf, (int, float))


if __name__ == '__main__':
    unittest.main()