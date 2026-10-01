"""Tests for timbre features module."""

import unittest

import numpy as np

from src.bpm_detector.timbre_features import TimbreFeatureExtractor


class TestTimbreFeatureExtractor(unittest.TestCase):
    """Test cases for TimbreFeatureExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = TimbreFeatureExtractor(hop_length=512, n_fft=2048)
        self.sr = 22050

        # Create test audio
        duration = 2  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_extract_timbral_features(self):
        """Test timbral feature extraction."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)

        # Check required features
        expected_features = [
            'spectral_centroid',
            'spectral_contrast',
            'mfcc',
            'spectral_rolloff',
            'zero_crossing_rate',
            'chroma',
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            self.assertGreater(features[feature].shape[0], 0)

    def test_analyze_brightness(self):
        """Test brightness analysis."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)
        brightness = self.extractor.analyze_brightness(features['spectral_centroid'], self.sr)

        self.assertIsInstance(brightness, (int, float))
        self.assertGreaterEqual(brightness, 0.0)
        self.assertLessEqual(brightness, 1.0)

    def test_analyze_roughness(self):
        """Test roughness analysis."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)
        roughness = self.extractor.analyze_roughness(features['spectral_contrast'])

        self.assertIsInstance(roughness, (int, float))
        self.assertGreaterEqual(roughness, 0.0)
        self.assertLessEqual(roughness, 1.0)

    def test_analyze_warmth(self):
        """Test warmth analysis."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)
        warmth = self.extractor.analyze_warmth(features['mfcc'])

        self.assertIsInstance(warmth, (int, float))
        self.assertGreaterEqual(warmth, 0.0)
        self.assertLessEqual(warmth, 1.0)

    def test_analyze_density(self):
        """Test density analysis."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)
        density = self.extractor.analyze_density(features)

        self.assertIsInstance(density, (int, float))
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)

    def test_analyze_texture(self):
        """Test texture analysis."""
        features = self.extractor.extract_timbral_features(self.test_audio, self.sr)
        texture = self.extractor.analyze_texture(features)

        # Check required fields
        expected_fields = ['spectral_complexity', 'harmonic_richness', 'temporal_stability', 'timbral_consistency']
        for field in expected_fields:
            self.assertIn(field, texture)
            self.assertIsInstance(texture[field], (int, float))
            self.assertGreaterEqual(texture[field], 0.0)
            self.assertLessEqual(texture[field], 1.0)


if __name__ == '__main__':
    unittest.main()


def test_constant_contrast_roughness_stays_in_range():
    extractor = TimbreFeatureExtractor()
    for contrast in (np.zeros((7, 20)), np.ones((7, 20))):
        roughness = extractor.analyze_roughness(contrast)
        assert np.isfinite(roughness)
        assert 0 <= roughness <= 1


def test_texture_uses_rms_time_axis():
    extractor = TimbreFeatureExtractor()
    steady = extractor.analyze_texture({'rms': np.ones((1, 10))})
    changing = extractor.analyze_texture({'rms': np.array([[0.0, 1.0] * 5])})
    assert changing['temporal_stability'] < steady['temporal_stability']


def test_timbral_features_support_low_sample_rates():
    from src.bpm_detector.timbre_analyzer import TimbreAnalyzer

    for sr in (8000, 16000, 22050):
        y = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        features = TimbreFeatureExtractor().extract_timbral_features(y, sr)
        assert np.isfinite(features['spectral_contrast']).all()
        result = TimbreAnalyzer().analyze(y, sr)
        assert 0 <= result['roughness'] <= 1
