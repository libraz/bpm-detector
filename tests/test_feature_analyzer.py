"""Regression tests for feature analysis."""

import unittest
from unittest.mock import patch

import numpy as np

from src.bpm_detector.feature_analyzer import FeatureAnalyzer


class TestFeatureAnalyzer(unittest.TestCase):
    """Test tempo handling in segment feature extraction."""

    def test_segment_analysis_preserves_ndarray_tempo(self):
        """The tempo returned as a one-element ndarray must be preserved."""
        sr = 22050
        clicks = np.zeros(sr * 2, dtype=np.float32)
        clicks[:: int(sr * 60 / 150)] = 1.0

        with patch(
            'src.bpm_detector.feature_analyzer.librosa.beat.beat_track',
            return_value=(np.array([150.0]), np.array([0, 16, 32])),
        ):
            characteristics = FeatureAnalyzer().analyze_segment_characteristics(clicks, sr)

        self.assertEqual(characteristics['tempo'], 150.0)


if __name__ == '__main__':
    unittest.main()
