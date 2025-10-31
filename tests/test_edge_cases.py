"""Tests for edge cases and cross-cutting concerns.

This module contains tests for edge cases that don't fit neatly into
a single analyzer module. Future edge case tests should be added here.
"""

import unittest

import numpy as np

from src.bpm_detector.chord_analyzer import ChordProgressionAnalyzer
from src.bpm_detector.dynamics_analyzer import DynamicsAnalyzer


class TestSampleRateParameters(unittest.TestCase):
    """Test cases for sample rate parameter fixes across multiple analyzers."""

    def setUp(self):
        """Set up test fixtures."""
        self.chord_analyzer = ChordProgressionAnalyzer()
        self.dynamics_analyzer = DynamicsAnalyzer()

    def test_chord_detect_chords_accepts_sr(self):
        """Test that detect_chords accepts sr parameter."""
        # Create minimal test data
        chroma = np.random.rand(12, 100)

        # Should accept sr parameter without error
        try:
            result = self.chord_analyzer.detect_chords(chroma, bpm=120.0, sr=44100)
            self.assertIsInstance(result, list)
        except TypeError as e:
            if 'sr' in str(e):
                self.fail("detect_chords should accept sr parameter")

    def test_chord_analyze_progression_accepts_sr(self):
        """Test that analyze_progression accepts sr parameter."""
        # Create minimal test data
        chords = [('C', 0.9, 0, 10), ('G', 0.8, 10, 20)]

        # Should accept sr parameter without error
        try:
            result = self.chord_analyzer.analyze_progression(chords, sr=44100)
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if 'sr' in str(e):
                self.fail("analyze_progression should accept sr parameter")

    def test_dynamics_generate_energy_profile_accepts_sr(self):
        """Test that generate_energy_profile accepts sr parameter."""
        # Create minimal test data
        energy_features = {'rms': np.random.rand(100), 'onset_strength': np.random.rand(100)}

        # Should accept sr parameter without error
        try:
            result = self.dynamics_analyzer.generate_energy_profile(energy_features, sr=44100)
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if 'sr' in str(e):
                self.fail("generate_energy_profile should accept sr parameter")

    def test_dynamics_detect_climax_points_accepts_sr(self):
        """Test that detect_climax_points accepts sr parameter."""
        # Create minimal test data
        energy_features = {'rms': np.random.rand(100), 'onset_strength': np.random.rand(100)}

        # Should accept sr parameter without error
        try:
            result = self.dynamics_analyzer.detect_climax_points(energy_features, sr=44100)
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if 'sr' in str(e):
                self.fail("detect_climax_points should accept sr parameter")

    def test_dynamics_detect_dynamic_events_accepts_sr(self):
        """Test that detect_dynamic_events accepts sr parameter."""
        # Create minimal test data
        energy_features = {'rms': np.random.rand(100), 'onset_strength': np.random.rand(100)}

        # Should accept sr parameter without error
        try:
            result = self.dynamics_analyzer.detect_dynamic_events(energy_features, sr=44100)
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if 'sr' in str(e):
                self.fail("detect_dynamic_events should accept sr parameter")

    def test_different_sample_rates_produce_different_results(self):
        """Test that different sample rates produce different time calculations."""
        # Create test data with known duration
        rms_length = 100

        energy_features = {'rms': np.ones(rms_length), 'onset_strength': np.ones(rms_length)}

        # Generate energy profile with different sample rates
        profile_22050 = self.dynamics_analyzer.generate_energy_profile(energy_features, window_size=1.0, sr=22050)
        profile_44100 = self.dynamics_analyzer.generate_energy_profile(energy_features, window_size=1.0, sr=44100)

        # Time points should be different due to different sample rates
        if len(profile_22050['time_points']) > 0 and len(profile_44100['time_points']) > 0:
            # The last time point should be different
            self.assertNotEqual(
                profile_22050['time_points'][-1],
                profile_44100['time_points'][-1],
                "Different sample rates should produce different time points",
            )


if __name__ == '__main__':
    unittest.main()
