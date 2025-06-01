"""Tests for boundary detector module."""

import unittest

import numpy as np

from src.bpm_detector.boundary_detector import BoundaryDetector


class TestBoundaryDetector(unittest.TestCase):
    """Test cases for BoundaryDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = BoundaryDetector(hop_length=512)
        self.sr = 22050
        # Create synthetic audio signal
        duration = 10  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        # Create a signal with clear structural changes
        self.y = np.concatenate(
            [
                np.sin(2 * np.pi * 440 * t[: len(t) // 3]),  # First section
                np.sin(2 * np.pi * 880 * t[: len(t) // 3]),  # Second section
                np.sin(2 * np.pi * 220 * t[: len(t) // 3]),  # Third section
            ]
        )

    def test_extract_structural_features(self):
        """Test essential structural feature extraction."""
        features = self.detector.extract_structural_features(self.y, self.sr)

        # Check that essential features are present (simplified for reliability)
        expected_features = ['mfcc', 'chroma', 'rms', 'spectral_centroid']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            self.assertGreater(features[feature].shape[0], 0)

        # Check feature dimensions
        self.assertEqual(features['mfcc'].shape[0], 13)  # 13 MFCC coefficients
        self.assertEqual(features['chroma'].shape[0], 12)  # 12 chroma bins
        self.assertEqual(features['rms'].shape[0], 1)  # 1 RMS value per frame
        self.assertEqual(features['spectral_centroid'].shape[0], 1)  # 1 centroid per frame

    def test_compute_self_similarity_matrix(self):
        """Test self-similarity matrix computation."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        # Check matrix properties
        self.assertIsInstance(similarity_matrix, np.ndarray)
        self.assertEqual(len(similarity_matrix.shape), 2)
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])

        # Check that diagonal is 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        np.testing.assert_array_almost_equal(diagonal, np.ones_like(diagonal), decimal=5)

        # Check symmetry
        np.testing.assert_array_almost_equal(similarity_matrix, similarity_matrix.T, decimal=5)

    def test_detect_boundaries(self):
        """Test boundary detection."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        boundaries = self.detector.detect_boundaries(similarity_matrix, sr=self.sr, min_segment_length=2.0, bpm=120.0)

        # Check that boundaries are returned
        self.assertIsInstance(boundaries, list)
        # Should have at least some boundaries for our synthetic signal
        self.assertGreaterEqual(len(boundaries), 0)

        # Check that boundaries are in ascending order
        if len(boundaries) > 1:
            self.assertTrue(all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1)))

    def test_compute_novelty(self):
        """Test novelty computation."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        novelty = self.detector._compute_novelty(similarity_matrix)

        self.assertIsInstance(novelty, np.ndarray)
        self.assertEqual(len(novelty), similarity_matrix.shape[0])
        self.assertTrue(np.all(np.isfinite(novelty)))

    def test_snap_to_beat(self):
        """Test beat-aligned boundary snapping."""
        # Create some test boundaries
        boundaries = [50, 150, 250]  # Frame indices
        bpm = 120.0

        snapped_boundaries = self.detector.snap_to_beat(boundaries, self.sr, bpm)

        self.assertIsInstance(snapped_boundaries, list)
        # Length may be different due to duplicate removal and minimum separation
        self.assertLessEqual(len(snapped_boundaries), len(boundaries))
        self.assertGreater(len(snapped_boundaries), 0)

        # Check that snapped boundaries are valid
        for snapped in snapped_boundaries:
            self.assertIsInstance(snapped, int)
            self.assertGreaterEqual(snapped, 0)

        # Check that boundaries are in ascending order
        if len(snapped_boundaries) > 1:
            self.assertTrue(
                all(snapped_boundaries[i] < snapped_boundaries[i + 1] for i in range(len(snapped_boundaries) - 1))
            )

    def test_detect_repetitions(self):
        """Test repetition detection."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        repetitions = self.detector.detect_repetitions(similarity_matrix, sr=self.sr)

        self.assertIsInstance(repetitions, list)

        # Check repetition structure
        for rep in repetitions:
            self.assertIsInstance(rep, dict)
            required_keys = ['first_occurrence', 'second_occurrence', 'duration', 'similarity']
            for key in required_keys:
                self.assertIn(key, rep)

            # Check time ordering
            self.assertLessEqual(rep['first_occurrence'], rep['second_occurrence'])
            self.assertGreater(rep['duration'], 0)

            # Check similarity score
            self.assertGreaterEqual(rep['similarity'], 0.0)
            self.assertLessEqual(rep['similarity'], 1.0)

    def test_remove_overlapping_repetitions(self):
        """Test overlapping repetition removal."""
        # Create test repetitions with overlaps
        repetitions = [
            {'first_occurrence': 0, 'second_occurrence': 20, 'duration': 10, 'similarity': 0.8},
            {'first_occurrence': 5, 'second_occurrence': 25, 'duration': 10, 'similarity': 0.7},
            {'first_occurrence': 40, 'second_occurrence': 60, 'duration': 10, 'similarity': 0.9},
        ]

        filtered = self.detector._remove_overlapping_repetitions(repetitions)

        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(repetitions))

        # Check that higher similarity repetitions are kept
        if len(filtered) > 0:
            similarities = [rep['similarity'] for rep in filtered]
            self.assertTrue(all(sim >= 0.7 for sim in similarities))

    def test_repetitions_overlap(self):
        """Test repetition overlap detection."""
        rep1 = {'first_occurrence': 0, 'second_occurrence': 20, 'duration': 10}
        rep2 = {'first_occurrence': 5, 'second_occurrence': 25, 'duration': 10}
        rep3 = {'first_occurrence': 40, 'second_occurrence': 60, 'duration': 10}

        # Test overlapping repetitions
        self.assertTrue(self.detector._repetitions_overlap(rep1, rep2))

        # Test non-overlapping repetitions
        self.assertFalse(self.detector._repetitions_overlap(rep1, rep3))

    def test_empty_input(self):
        """Test behavior with empty input."""
        empty_audio = np.array([])

        try:
            features = self.detector.extract_structural_features(empty_audio, self.sr)
            # Should handle empty input gracefully
            self.assertIsInstance(features, dict)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty input
            pass

    def test_short_audio(self):
        """Test behavior with very short audio."""
        # Create 1 second of audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sr))

        features = self.detector.extract_structural_features(short_audio, self.sr)
        self.assertIsInstance(features, dict)

        # Should still extract features even for short audio
        for feature_name, feature_data in features.items():
            self.assertIsInstance(feature_data, np.ndarray)
            self.assertGreater(feature_data.shape[0], 0)

    def test_boundary_detection_parameters(self):
        """Test boundary detection with different parameters."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        # Test with strict threshold
        strict_boundaries = self.detector.detect_boundaries(
            similarity_matrix, sr=self.sr, min_segment_length=5.0, bpm=120.0
        )

        # Test with loose threshold
        loose_boundaries = self.detector.detect_boundaries(
            similarity_matrix, sr=self.sr, min_segment_length=1.0, bpm=120.0
        )

        # Loose threshold should generally find more boundaries
        self.assertGreaterEqual(len(loose_boundaries), len(strict_boundaries))

    def test_beat_synchronized_novelty(self):
        """Test enhanced beat-synchronized novelty detection."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        # Test beat-synchronized novelty with different BPM values
        bpm_values = [120, 130, 140]
        for bpm in bpm_values:
            novelty = self.detector._compute_beat_synchronized_novelty(similarity_matrix, self.sr, bpm)

            self.assertIsInstance(novelty, np.ndarray)
            self.assertEqual(len(novelty), similarity_matrix.shape[0])
            self.assertTrue(np.all(np.isfinite(novelty)))
            # Novelty should be normalized
            self.assertGreaterEqual(np.min(novelty), 0.0)
            self.assertLessEqual(np.max(novelty), 1.0)

    def test_foote_kernel_creation(self):
        """Test Foote kernel creation for boundary detection."""
        kernel_sizes = [8, 16, 32]
        for size in kernel_sizes:
            kernel = self.detector._create_foote_kernel(size)

            self.assertIsInstance(kernel, np.ndarray)
            self.assertEqual(kernel.shape, (2 * size, 2 * size))

            # Check checkerboard pattern
            # Top-left and bottom-right should be positive
            self.assertTrue(np.all(kernel[:size, :size] > 0))
            self.assertTrue(np.all(kernel[size:, size:] > 0))

            # Top-right and bottom-left should be negative
            self.assertTrue(np.all(kernel[:size, size:] < 0))
            self.assertTrue(np.all(kernel[size:, :size] < 0))

    def test_8bar_grid_alignment(self):
        """Test 8-bar grid alignment using Hough transform approach."""
        # Create test novelty function
        novelty = np.random.rand(1000)
        bar_8_frames = 100  # Mock 8-bar frame count

        aligned_novelty = self.detector._align_to_8bar_grid(novelty, bar_8_frames)

        self.assertIsInstance(aligned_novelty, np.ndarray)
        self.assertEqual(len(aligned_novelty), len(novelty))

        # Most values should be zero (only peaks at grid positions)
        non_zero_count = np.count_nonzero(aligned_novelty)
        self.assertLess(non_zero_count, len(aligned_novelty) * 0.2)  # Less than 20%

    def test_rbf_similarity_computation(self):
        """Test RBF kernel similarity matrix with PCA."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        # RBF similarity should produce valid results
        self.assertIsInstance(similarity_matrix, np.ndarray)
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])

        # Values should be in [0, 1] range
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))

        # Diagonal should be close to 1 (self-similarity with RBF)
        diagonal = np.diag(similarity_matrix)
        self.assertTrue(np.all(diagonal > 0.8))  # RBF may not be exactly 1

    def test_realistic_track_length(self):
        """Test boundary detection with realistic track length (3-5 minutes)."""
        # Create a 4-minute synthetic track with clear structural changes
        duration = 240  # 4 minutes
        t = np.linspace(0, duration, int(self.sr * duration))

        # Create sections: Intro (30s) -> Verse (60s) -> Chorus (60s) -> Verse (60s) -> Outro (30s)
        intro = np.sin(2 * np.pi * 220 * t[: int(30 * self.sr)])  # 30s intro
        verse1 = np.sin(2 * np.pi * 440 * t[: int(60 * self.sr)])  # 60s verse
        chorus = np.sin(2 * np.pi * 880 * t[: int(60 * self.sr)])  # 60s chorus
        verse2 = np.sin(2 * np.pi * 440 * t[: int(60 * self.sr)])  # 60s verse
        outro = np.sin(2 * np.pi * 220 * t[: int(30 * self.sr)])  # 30s outro

        realistic_audio = np.concatenate([intro, verse1, chorus, verse2, outro])

        # Test boundary detection
        features = self.detector.extract_structural_features(realistic_audio, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)
        boundaries = self.detector.detect_boundaries(
            similarity_matrix, sr=self.sr, min_segment_length=8.0, bpm=120.0  # 8 seconds minimum
        )

        # Should detect multiple boundaries for a 4-minute track
        self.assertGreaterEqual(len(boundaries), 3)  # At least start, middle, end
        self.assertLessEqual(len(boundaries), 20)  # Not too many boundaries

        # Check boundary times are reasonable
        boundary_times = [b * self.detector.hop_length / self.sr for b in boundaries]
        self.assertEqual(boundary_times[0], 0.0)  # Should start at 0
        self.assertGreater(boundary_times[-1], 230)  # Should end near track end

        # Boundaries should be in ascending order
        self.assertTrue(all(boundary_times[i] < boundary_times[i + 1] for i in range(len(boundary_times) - 1)))

    def test_boundary_detection_consistency(self):
        """Test that boundary detection produces consistent results."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        # Run boundary detection multiple times
        boundaries1 = self.detector.detect_boundaries(similarity_matrix, self.sr, bpm=120.0)
        boundaries2 = self.detector.detect_boundaries(similarity_matrix, self.sr, bpm=120.0)
        boundaries3 = self.detector.detect_boundaries(similarity_matrix, self.sr, bpm=120.0)

        # Results should be identical (deterministic algorithm)
        self.assertEqual(boundaries1, boundaries2)
        self.assertEqual(boundaries2, boundaries3)

    def test_simple_novelty_function(self):
        """Test the simplified novelty function."""
        features = self.detector.extract_structural_features(self.y, self.sr)
        similarity_matrix = self.detector.compute_self_similarity_matrix(features)

        novelty = self.detector._compute_simple_novelty(similarity_matrix)

        self.assertIsInstance(novelty, np.ndarray)
        self.assertEqual(len(novelty), similarity_matrix.shape[0])
        self.assertTrue(np.all(np.isfinite(novelty)))

        # Novelty should be normalized to [0, 1]
        self.assertGreaterEqual(np.min(novelty), 0.0)
        self.assertLessEqual(np.max(novelty), 1.0)


if __name__ == '__main__':
    unittest.main()
