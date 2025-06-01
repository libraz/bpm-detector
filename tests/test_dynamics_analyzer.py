"""Tests for dynamics analyzer module."""

import unittest

import numpy as np

from src.bpm_detector.dynamics_analyzer import DynamicsAnalyzer


class TestDynamicsAnalyzer(unittest.TestCase):
    """Test cases for DynamicsAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DynamicsAnalyzer(hop_length=512, frame_size=2048)
        self.sr = 22050

        # Create test audio with varying dynamics
        duration = 10  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create audio with dynamic changes
        # Quiet beginning, loud middle, quiet end
        envelope = np.concatenate(
            [
                np.linspace(0.1, 0.1, len(t) // 3),  # Quiet start
                np.linspace(0.1, 0.9, len(t) // 6),  # Build up
                np.linspace(0.9, 0.9, len(t) // 3),  # Loud middle
                np.linspace(0.9, 0.1, len(t) // 6),  # Fade out
            ]
        )

        # Ensure envelope matches audio length
        if len(envelope) != len(t):
            envelope = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(envelope)), envelope)

        self.dynamic_audio = envelope * np.sin(2 * np.pi * 440 * t)

        # Create constant level audio for comparison
        self.constant_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_extract_energy_features(self):
        """Test energy feature extraction."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)

        # Check required features
        expected_features = ['rms', 'spectral_energy', 'zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            self.assertGreater(len(features[feature]), 0)

        # Check that RMS varies for dynamic audio
        rms_std = np.std(features['rms'])
        self.assertGreater(rms_std, 0.01)  # Should have some variation

    def test_calculate_dynamic_range(self):
        """Test dynamic range calculation."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        dynamic_range = self.analyzer.calculate_dynamic_range(features['rms'])

        # Check required fields
        expected_fields = ['dynamic_range_db', 'crest_factor', 'rms_std', 'peak_to_average']
        for field in expected_fields:
            self.assertIn(field, dynamic_range)
            self.assertIsInstance(dynamic_range[field], (int, float))

        # Dynamic range should be positive
        self.assertGreater(dynamic_range['dynamic_range_db'], 0)

        # Crest factor should be >= 1
        self.assertGreaterEqual(dynamic_range['crest_factor'], 1.0)

        # Compare with constant audio
        const_features = self.analyzer.extract_energy_features(self.constant_audio, self.sr)
        const_range = self.analyzer.calculate_dynamic_range(const_features['rms'])

        # Dynamic audio should have larger dynamic range
        self.assertGreater(dynamic_range['dynamic_range_db'], const_range['dynamic_range_db'])

    def test_analyze_loudness(self):
        """Test loudness analysis."""
        loudness = self.analyzer.analyze_loudness(self.dynamic_audio, self.sr)

        # Check required fields
        expected_fields = ['perceived_loudness', 'peak_loudness', 'average_loudness', 'loudness_range']
        for field in expected_fields:
            self.assertIn(field, loudness)
            self.assertIsInstance(loudness[field], (int, float))

        # Loudness values should be reasonable
        self.assertGreater(loudness['perceived_loudness'], 0)
        self.assertGreater(loudness['peak_loudness'], 0)
        self.assertGreater(loudness['average_loudness'], 0)
        self.assertGreaterEqual(loudness['loudness_range'], 0)

        # Peak should be >= average
        self.assertGreaterEqual(loudness['peak_loudness'], loudness['average_loudness'])

    def test_calculate_perceived_loudness(self):
        """Test perceived loudness calculation."""
        perceived = self.analyzer._calculate_perceived_loudness(self.dynamic_audio, self.sr)

        # Should return a positive number
        self.assertIsInstance(perceived, (int, float))
        self.assertGreater(perceived, 0)

        # Louder audio should have higher perceived loudness
        loud_audio = 2.0 * self.dynamic_audio
        loud_perceived = self.analyzer._calculate_perceived_loudness(loud_audio, self.sr)
        self.assertGreater(loud_perceived, perceived)

    def test_generate_energy_profile(self):
        """Test energy profile generation."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        profile = self.analyzer.generate_energy_profile(features, window_size=1.0)

        # Check required fields
        expected_fields = ['time_points', 'energy_curve', 'smoothed_energy', 'energy_derivative']
        for field in expected_fields:
            self.assertIn(field, profile)
            self.assertIsInstance(profile[field], np.ndarray)

        # All arrays should have same length
        lengths = [len(profile[field]) for field in expected_fields]
        self.assertTrue(all(length == lengths[0] for length in lengths))

        # Time points should be increasing
        self.assertTrue(np.all(np.diff(profile['time_points']) > 0))

    def test_detect_climax_points(self):
        """Test climax point detection."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        climax_points = self.analyzer.detect_climax_points(features, prominence_threshold=0.1)

        # Check structure
        self.assertIsInstance(climax_points, dict)

        expected_fields = ['climax_times', 'climax_energies', 'main_climax', 'climax_count']
        for field in expected_fields:
            self.assertIn(field, climax_points)

        # Climax times and energies should be arrays
        self.assertIsInstance(climax_points['climax_times'], np.ndarray)
        self.assertIsInstance(climax_points['climax_energies'], np.ndarray)

        # Should have same number of times and energies
        self.assertEqual(len(climax_points['climax_times']), len(climax_points['climax_energies']))

        # Main climax should be a time value
        if climax_points['climax_count'] > 0:
            self.assertIsInstance(climax_points['main_climax'], (int, float))
            self.assertGreaterEqual(climax_points['main_climax'], 0)

    def test_analyze_tension_curve(self):
        """Test tension curve analysis."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        tension = self.analyzer.analyze_tension_curve(features, window_size=1.0)

        # Check required fields
        expected_fields = ['tension_curve', 'tension_peaks', 'tension_valleys', 'average_tension', 'tension_variance']
        for field in expected_fields:
            self.assertIn(field, tension)

        # Tension curve should be an array
        self.assertIsInstance(tension['tension_curve'], np.ndarray)
        self.assertGreater(len(tension['tension_curve']), 0)

        # Peaks and valleys should be arrays
        self.assertIsInstance(tension['tension_peaks'], np.ndarray)
        self.assertIsInstance(tension['tension_valleys'], np.ndarray)

        # Statistics should be numbers
        self.assertIsInstance(tension['average_tension'], (int, float))
        self.assertIsInstance(tension['tension_variance'], (int, float))

        # Variance should be non-negative
        self.assertGreaterEqual(tension['tension_variance'], 0)

    def test_analyze_energy_distribution(self):
        """Test energy distribution analysis."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        distribution = self.analyzer.analyze_energy_distribution(features)

        # Check required fields
        expected_fields = [
            'low_energy_ratio',
            'mid_energy_ratio',
            'high_energy_ratio',
            'energy_concentration',
            'energy_spread',
        ]
        for field in expected_fields:
            self.assertIn(field, distribution)
            self.assertIsInstance(distribution[field], (int, float))

        # Ratios should sum to approximately 1
        total_ratio = (
            distribution['low_energy_ratio'] + distribution['mid_energy_ratio'] + distribution['high_energy_ratio']
        )
        self.assertAlmostEqual(total_ratio, 1.0, places=2)

        # All ratios should be between 0 and 1
        for ratio_field in ['low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio']:
            self.assertGreaterEqual(distribution[ratio_field], 0.0)
            self.assertLessEqual(distribution[ratio_field], 1.0)

    def test_detect_dynamic_events(self):
        """Test dynamic event detection."""
        features = self.analyzer.extract_energy_features(self.dynamic_audio, self.sr)
        events = self.analyzer.detect_dynamic_events(features, threshold=0.2)

        # Check structure
        self.assertIsInstance(events, dict)

        expected_fields = ['sudden_increases', 'sudden_decreases', 'sustained_peaks', 'quiet_sections']
        for field in expected_fields:
            self.assertIn(field, events)
            self.assertIsInstance(events[field], list)

        # Each event should have time and magnitude
        for event_type in expected_fields:
            for event in events[event_type]:
                self.assertIsInstance(event, dict)
                self.assertIn('time', event)
                self.assertIn('magnitude', event)
                self.assertIsInstance(event['time'], (int, float))
                self.assertIsInstance(event['magnitude'], (int, float))

    def test_analyze_complete(self):
        """Test complete dynamics analysis."""
        results = self.analyzer.analyze(self.dynamic_audio, self.sr)

        # Check main structure
        self.assertIsInstance(results, dict)

        expected_sections = [
            'dynamic_range',
            'loudness',
            'energy_profile',
            'climax_points',
            'tension_curve',
            'energy_distribution',
            'dynamic_events',
            'overall_energy',
            'energy_variance',
        ]
        for section in expected_sections:
            self.assertIn(section, results)

        # Check that each section has expected structure
        self.assertIsInstance(results['dynamic_range'], dict)
        self.assertIsInstance(results['loudness'], dict)
        self.assertIsInstance(results['energy_profile'], list)
        self.assertIsInstance(results['climax_points'], list)
        self.assertIsInstance(results['tension_curve'], list)
        self.assertIsInstance(results['energy_distribution'], dict)
        self.assertIsInstance(results['dynamic_events'], list)
        self.assertIsInstance(results['overall_energy'], (int, float))
        self.assertIsInstance(results['energy_variance'], (int, float))

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])

        try:
            results = self.analyzer.analyze(empty_audio, self.sr)
            # Should handle empty input gracefully
            self.assertIsInstance(results, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    def test_short_audio_handling(self):
        """Test handling of very short audio."""
        # Create 0.5 second audio
        short_audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(self.sr * 0.5)))

        results = self.analyzer.analyze(short_audio, self.sr)

        # Should handle short audio
        self.assertIsInstance(results, dict)
        self.assertIn('dynamic_range', results)
        self.assertIn('loudness', results)

    def test_constant_audio_dynamics(self):
        """Test dynamics analysis on constant level audio."""
        results = self.analyzer.analyze(self.constant_audio, self.sr)

        # Should have minimal dynamic range
        dynamic_range = results['dynamic_range']['dynamic_range_db']
        self.assertLess(dynamic_range, 10.0)  # Should be small for constant audio

        # Should have low energy variance
        energy_variance = results['energy_variance']
        self.assertLess(energy_variance, 0.1)

    def test_silence_handling(self):
        """Test handling of silent audio."""
        silent_audio = np.zeros(self.sr * 2)  # 2 seconds of silence

        results = self.analyzer.analyze(silent_audio, self.sr)

        # Should handle silence gracefully
        self.assertIsInstance(results, dict)

        # Loudness should be very low
        loudness = results['loudness']['average_loudness']
        self.assertLess(loudness, 0.01)

    def test_different_frame_sizes(self):
        """Test analyzer with different frame sizes."""
        # Test with smaller frame size
        small_frame_analyzer = DynamicsAnalyzer(hop_length=256, frame_size=1024)
        results_small = small_frame_analyzer.analyze(self.dynamic_audio, self.sr)

        # Test with larger frame size
        large_frame_analyzer = DynamicsAnalyzer(hop_length=1024, frame_size=4096)
        results_large = large_frame_analyzer.analyze(self.dynamic_audio, self.sr)

        # Both should produce valid results
        self.assertIsInstance(results_small, dict)
        self.assertIsInstance(results_large, dict)

        # Should have different temporal resolution
        small_profile_len = len(results_small['energy_profile'])
        large_profile_len = len(results_large['energy_profile'])

        # Both should have energy profiles
        self.assertGreater(small_profile_len, 0)
        self.assertGreater(large_profile_len, 0)

    def test_climax_detection_accuracy(self):
        """Test that climax detection finds the actual peak."""
        # Create audio with clear peak in the middle
        t = np.linspace(0, 5, int(self.sr * 5))
        peak_time = 2.5  # Peak at 2.5 seconds

        # Gaussian envelope centered at peak_time
        envelope = np.exp(-((t - peak_time) ** 2) / (2 * 0.5**2))
        peak_audio = envelope * np.sin(2 * np.pi * 440 * t)

        results = self.analyzer.analyze(peak_audio, self.sr)

        # Climax points should be detected
        climax_points = results['climax_points']
        if len(climax_points) > 0:
            # Should detect at least one climax point
            main_climax_time = climax_points[0]['time']
            # Should be within 1 second of the actual peak
            self.assertLess(abs(main_climax_time - peak_time), 1.0)


if __name__ == '__main__':
    unittest.main()
