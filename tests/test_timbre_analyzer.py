"""Tests for timbre analyzer module."""

import unittest
import numpy as np
from src.bpm_detector.timbre_analyzer import TimbreAnalyzer


class TestTimbreAnalyzer(unittest.TestCase):
    """Test cases for TimbreAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TimbreAnalyzer(hop_length=512, n_fft=2048)
        self.sr = 22050

        # Create test audio with different timbral characteristics
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))

        # Bright sound (high frequencies)
        self.bright_audio = (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.4 * np.sin(2 * np.pi * 880 * t)
            + 0.3 * np.sin(2 * np.pi * 1760 * t)
        )

        # Warm sound (low frequencies)
        self.warm_audio = (
            0.5 * np.sin(2 * np.pi * 220 * t)
            + 0.3 * np.sin(2 * np.pi * 330 * t)
            + 0.2 * np.sin(2 * np.pi * 440 * t)
        )

        # Rough sound (with noise)
        self.rough_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(
            len(t)
        )

        # Dense sound (many harmonics)
        harmonics = [440 * (i + 1) for i in range(8)]
        self.dense_audio = sum(
            (0.8 / (i + 1)) * np.sin(2 * np.pi * freq * t)
            for i, freq in enumerate(harmonics)
        )

    def test_extract_timbral_features(self):
        """Test timbral feature extraction."""
        features = self.analyzer.extract_timbral_features(self.bright_audio, self.sr)

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

        # MFCC should have multiple coefficients
        self.assertGreater(features['mfcc'].shape[0], 1)

    def test_analyze_brightness(self):
        """Test brightness analysis."""
        # Extract features for bright audio
        bright_features = self.analyzer.extract_timbral_features(
            self.bright_audio, self.sr
        )
        bright_score = self.analyzer.analyze_brightness(
            bright_features['spectral_centroid'], self.sr
        )

        # Extract features for warm audio
        warm_features = self.analyzer.extract_timbral_features(self.warm_audio, self.sr)
        warm_score = self.analyzer.analyze_brightness(
            warm_features['spectral_centroid'], self.sr
        )

        # Brightness should be a float between 0 and 1
        self.assertIsInstance(bright_score, (int, float))
        self.assertGreaterEqual(bright_score, 0.0)
        self.assertLessEqual(bright_score, 1.0)

        self.assertIsInstance(warm_score, (int, float))
        self.assertGreaterEqual(warm_score, 0.0)
        self.assertLessEqual(warm_score, 1.0)

        # Bright audio should have higher brightness score
        self.assertGreater(bright_score, warm_score)

    def test_analyze_roughness(self):
        """Test roughness analysis."""
        # Extract features for rough audio
        rough_features = self.analyzer.extract_timbral_features(
            self.rough_audio, self.sr
        )
        rough_score = self.analyzer.analyze_roughness(
            rough_features['spectral_contrast']
        )

        # Extract features for smooth audio
        smooth_features = self.analyzer.extract_timbral_features(
            self.bright_audio, self.sr
        )
        smooth_score = self.analyzer.analyze_roughness(
            smooth_features['spectral_contrast']
        )

        # Roughness should be a float between 0 and 1
        self.assertIsInstance(rough_score, (int, float))
        self.assertGreaterEqual(rough_score, 0.0)
        self.assertLessEqual(rough_score, 1.0)

        self.assertIsInstance(smooth_score, (int, float))
        self.assertGreaterEqual(smooth_score, 0.0)
        self.assertLessEqual(smooth_score, 1.0)

        # Rough audio should have higher roughness score
        self.assertGreater(rough_score, smooth_score)

    def test_analyze_warmth(self):
        """Test warmth analysis."""
        # Extract features for warm audio
        warm_features = self.analyzer.extract_timbral_features(self.warm_audio, self.sr)
        warm_score = self.analyzer.analyze_warmth(warm_features['mfcc'])

        # Extract features for bright audio
        bright_features = self.analyzer.extract_timbral_features(
            self.bright_audio, self.sr
        )
        bright_score = self.analyzer.analyze_warmth(bright_features['mfcc'])

        # Warmth should be a float between 0 and 1
        self.assertIsInstance(warm_score, (int, float))
        self.assertGreaterEqual(warm_score, 0.0)
        self.assertLessEqual(warm_score, 1.0)

        self.assertIsInstance(bright_score, (int, float))
        self.assertGreaterEqual(bright_score, 0.0)
        self.assertLessEqual(bright_score, 1.0)

        # Warm audio should have higher warmth score
        self.assertGreater(warm_score, bright_score)

    def test_analyze_density(self):
        """Test density analysis."""
        # Extract features for dense audio
        dense_features = self.analyzer.extract_timbral_features(
            self.dense_audio, self.sr
        )
        dense_score = self.analyzer.analyze_density(dense_features)

        # Extract features for simple audio
        simple_features = self.analyzer.extract_timbral_features(
            self.bright_audio, self.sr
        )
        simple_score = self.analyzer.analyze_density(simple_features)

        # Density should be a float between 0 and 1
        self.assertIsInstance(dense_score, (int, float))
        self.assertGreaterEqual(dense_score, 0.0)
        self.assertLessEqual(dense_score, 1.0)

        self.assertIsInstance(simple_score, (int, float))
        self.assertGreaterEqual(simple_score, 0.0)
        self.assertLessEqual(simple_score, 1.0)

        # Dense audio should have higher density score
        self.assertGreater(dense_score, simple_score)

    def test_classify_instruments(self):
        """Test instrument classification."""
        instruments = self.analyzer.classify_instruments(self.bright_audio, self.sr)

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

        filtered = self.analyzer._filter_redundant_instruments(test_instruments)

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
        confidence = self.analyzer._calculate_instrument_confidence(
            magnitude, freqs, freq_range=(200, 2000), spectral_shape='harmonic'
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

        prominence = self.analyzer._calculate_instrument_prominence(
            magnitude, freqs, freq_range=(freqs[100], freqs[200])
        )

        self.assertIsInstance(prominence, (int, float))
        self.assertGreaterEqual(prominence, 0.0)
        self.assertLessEqual(prominence, 1.0)

        # Should be high due to the peak in the frequency range
        self.assertGreater(prominence, 0.5)

    def test_analyze_effects_usage(self):
        """Test effects usage analysis."""
        effects = self.analyzer.analyze_effects_usage(self.bright_audio, self.sr)

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
        dry_reverb = self.analyzer._detect_reverb(self.bright_audio, self.sr)

        # Create audio with simulated reverb (add delayed copies)
        reverb_audio = self.bright_audio.copy()
        delay_samples = int(0.1 * self.sr)  # 100ms delay
        if len(reverb_audio) > delay_samples:
            reverb_audio[delay_samples:] += 0.3 * reverb_audio[:-delay_samples]

        wet_reverb = self.analyzer._detect_reverb(reverb_audio, self.sr)

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
        clean_distortion = self.analyzer._detect_distortion(self.bright_audio, self.sr)

        # Create distorted audio (clipping)
        distorted_audio = np.clip(3.0 * self.bright_audio, -1.0, 1.0)
        dirty_distortion = self.analyzer._detect_distortion(distorted_audio, self.sr)

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
        chorus_score = self.analyzer._detect_chorus(self.bright_audio, self.sr)

        self.assertIsInstance(chorus_score, (int, float))
        self.assertGreaterEqual(chorus_score, 0.0)
        self.assertLessEqual(chorus_score, 1.0)

    def test_detect_compression(self):
        """Test compression detection."""
        # Test with dynamic audio
        dynamic_compression = self.analyzer._detect_compression(self.bright_audio)

        # Create compressed audio (reduce dynamic range)
        compressed_audio = np.tanh(2.0 * self.bright_audio)
        compressed_compression = self.analyzer._detect_compression(compressed_audio)

        # Both should be valid scores
        self.assertIsInstance(dynamic_compression, (int, float))
        self.assertIsInstance(compressed_compression, (int, float))
        self.assertGreaterEqual(dynamic_compression, 0.0)
        self.assertLessEqual(dynamic_compression, 1.0)
        self.assertGreaterEqual(compressed_compression, 0.0)
        self.assertLessEqual(compressed_compression, 1.0)

        # Compressed audio should have higher compression score
        self.assertGreater(compressed_compression, dynamic_compression)

    def test_analyze_texture(self):
        """Test texture analysis."""
        features = self.analyzer.extract_timbral_features(self.dense_audio, self.sr)
        texture = self.analyzer.analyze_texture(features)

        # Check required fields
        expected_fields = [
            'spectral_complexity',
            'harmonic_richness',
            'temporal_stability',
            'timbral_consistency',
        ]
        for field in expected_fields:
            self.assertIn(field, texture)
            self.assertIsInstance(texture[field], (int, float))
            self.assertGreaterEqual(texture[field], 0.0)
            self.assertLessEqual(texture[field], 1.0)

    def test_analyze_complete(self):
        """Test complete timbre analysis."""
        results = self.analyzer.analyze(self.bright_audio, self.sr)

        # Check main structure
        self.assertIsInstance(results, dict)

        expected_sections = [
            'spectral_features',
            'brightness',
            'roughness',
            'warmth',
            'density',
            'dominant_instruments',
            'effects_usage',
            'texture',
        ]
        for section in expected_sections:
            self.assertIn(section, results)

        # Check specific values
        self.assertIsInstance(results['brightness'], (int, float))
        self.assertIsInstance(results['roughness'], (int, float))
        self.assertIsInstance(results['warmth'], (int, float))
        self.assertIsInstance(results['density'], (int, float))
        self.assertIsInstance(results['dominant_instruments'], list)
        self.assertIsInstance(results['effects_usage'], dict)
        self.assertIsInstance(results['texture'], dict)

    def test_analyze_with_progress_callback(self):
        """Test analysis with progress callback."""
        progress_calls = []

        def progress_callback(progress, message):
            progress_calls.append((progress, message))

        self.analyzer.analyze(
            self.bright_audio, self.sr, progress_callback=progress_callback
        )

        # Check that progress was reported
        self.assertGreater(len(progress_calls), 0)

        # Check progress values
        for progress, message in progress_calls:
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 100.0)
            self.assertIsInstance(message, str)

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])

        try:
            results = self.analyzer.analyze(empty_audio, self.sr)
            self.assertIsInstance(results, dict)
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass

    def test_short_audio_handling(self):
        """Test handling of very short audio."""
        short_audio = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.5, int(self.sr * 0.5))
        )

        results = self.analyzer.analyze(short_audio, self.sr)

        # Should handle short audio
        self.assertIsInstance(results, dict)
        self.assertIn('brightness', results)
        self.assertIn('warmth', results)

    def test_silence_handling(self):
        """Test handling of silent audio."""
        silent_audio = np.zeros(self.sr * 2)

        results = self.analyzer.analyze(silent_audio, self.sr)

        # Should handle silence gracefully
        self.assertIsInstance(results, dict)

        # Brightness should be low for silence
        self.assertLess(results['brightness'], 0.1)

    def test_different_parameters(self):
        """Test analyzer with different parameters."""
        custom_analyzer = TimbreAnalyzer(hop_length=256, n_fft=4096)
        results = custom_analyzer.analyze(self.bright_audio, self.sr)

        # Should produce valid results with different parameters
        self.assertIsInstance(results, dict)
        self.assertIn('brightness', results)
        self.assertIn('texture', results)

    def test_instrument_classification_consistency(self):
        """Test that instrument classification is reasonably consistent."""
        # Run classification multiple times on same audio
        results1 = self.analyzer.classify_instruments(self.bright_audio, self.sr)
        results2 = self.analyzer.classify_instruments(self.bright_audio, self.sr)

        # Should return same number of instruments
        self.assertEqual(len(results1), len(results2))

        # Should have similar instrument types (allowing for some variation)
        if len(results1) > 0 and len(results2) > 0:
            instruments1 = {inst['instrument'] for inst in results1}
            instruments2 = {inst['instrument'] for inst in results2}
            # At least some overlap expected
            self.assertGreater(len(instruments1.intersection(instruments2)), 0)


if __name__ == '__main__':
    unittest.main()
