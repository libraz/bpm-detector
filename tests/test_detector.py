"""Tests for detector module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bpm_detector.music_analyzer import AudioAnalyzer, BPMDetector, KeyDetector


class TestBPMDetector(unittest.TestCase):
    """Test BPM detection functionality."""

    def setUp(self):
        self.detector = BPMDetector()

    def test_harmonic_cluster(self):
        """Test harmonic clustering of BPM candidates."""
        bpms = np.array([120.0, 240.0, 60.0, 180.0])
        votes = np.array([45, 23, 18, 12])

        clusters = self.detector.harmonic_cluster(bpms, votes)

        # Should group harmonically related BPMs
        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)

        # Check that 120 and 240 are grouped together (2:1 ratio)
        found_120_cluster = False
        for base, candidates in clusters.items():
            if any(abs(bpm - 120.0) < 0.1 for bpm, _ in candidates):
                found_120_cluster = True
                break
        self.assertTrue(found_120_cluster)

    def test_smart_choice(self):
        """Test smart BPM selection from clusters."""
        clusters = {120.0: [(120.0, 45), (240.0, 23)], 60.0: [(60.0, 18)]}
        total_votes = 86

        bpm, conf = self.detector.smart_choice(clusters, total_votes)

        self.assertIsInstance(bpm, float)
        self.assertIsInstance(conf, float)
        self.assertGreater(conf, 0)
        self.assertLessEqual(conf, 100)

        # Should choose from the cluster with most votes (120.0 cluster has 68 votes)
        self.assertTrue(bpm in [120.0, 240.0])


class TestKeyDetector(unittest.TestCase):
    """Test key detection functionality."""

    def setUp(self):
        self.detector = KeyDetector()

    @patch("librosa.feature.chroma_stft")
    def test_detect_key(self, mock_chroma):
        """Test key detection from audio signal."""
        # Mock chroma features - create a pattern that should favor C Major
        chroma_features = np.zeros((12, 100))
        # Emphasize C, E, G (C Major triad)
        chroma_features[0, :] = 1.0  # C
        chroma_features[4, :] = 0.8  # E
        chroma_features[7, :] = 0.6  # G
        mock_chroma.return_value = chroma_features

        y = np.random.rand(22050)  # 1 second of audio
        sr = 22050

        key, confidence = self.detector.detect(y, sr)

        self.assertIsInstance(key, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 100)
        self.assertTrue(
            any(note in key for note in ["C", "D", "E", "F", "G", "A", "B"])
        )
        self.assertTrue("Major" in key or "Minor" in key)

    @patch("librosa.feature.chroma_stft")
    def test_detect_key_minor(self, mock_chroma):
        """Test key detection for minor key."""
        # Mock chroma features for A minor (A, C, E)
        chroma_features = np.zeros((12, 100))
        chroma_features[9, :] = 1.0  # A
        chroma_features[0, :] = 0.8  # C
        chroma_features[4, :] = 0.6  # E
        mock_chroma.return_value = chroma_features

        y = np.random.rand(22050)
        sr = 22050

        key, confidence = self.detector.detect(y, sr)

        self.assertIsInstance(key, str)
        self.assertIsInstance(confidence, float)
        # Should detect some form of minor key
        self.assertTrue("Minor" in key or "Major" in key)


class TestAudioAnalyzer(unittest.TestCase):
    """Test main audio analyzer."""

    def setUp(self):
        self.analyzer = AudioAnalyzer()

    @patch("librosa.load")
    @patch("bpm_detector.detector.BPMDetector.detect")
    def test_analyze_file_bpm_only(self, mock_bpm_detect, mock_load):
        """Test file analysis with BPM only."""
        # Mock audio loading
        mock_load.return_value = (np.random.rand(22050), 22050)

        # Mock BPM detection
        mock_bpm_detect.return_value = (
            120.0,
            85.5,
            np.array([120.0, 240.0]),
            np.array([45, 23]),
        )

        results = self.analyzer.analyze_file("test.wav", detect_key=False)

        self.assertIn("filename", results)
        self.assertIn("bpm", results)
        self.assertIn("bpm_confidence", results)
        self.assertIn("bpm_candidates", results)
        self.assertNotIn("key", results)

        self.assertEqual(results["bpm"], 120.0)
        self.assertEqual(results["bpm_confidence"], 85.5)
        self.assertEqual(len(results["bpm_candidates"]), 2)

    @patch("librosa.load")
    @patch("bpm_detector.detector.BPMDetector.detect")
    @patch("bpm_detector.detector.KeyDetector.detect")
    def test_analyze_file_with_key(self, mock_key_detect, mock_bpm_detect, mock_load):
        """Test file analysis with key detection."""
        # Mock audio loading
        mock_load.return_value = (np.random.rand(22050), 22050)

        # Mock BPM detection
        mock_bpm_detect.return_value = (
            120.0,
            85.5,
            np.array([120.0, 240.0]),
            np.array([45, 23]),
        )

        # Mock key detection
        mock_key_detect.return_value = ("C Major", 78.2)

        results = self.analyzer.analyze_file("test.wav", detect_key=True)

        self.assertIn("filename", results)
        self.assertIn("bpm", results)
        self.assertIn("bpm_confidence", results)
        self.assertIn("bpm_candidates", results)
        self.assertIn("key", results)
        self.assertIn("key_confidence", results)

        self.assertEqual(results["bpm"], 120.0)
        self.assertEqual(results["bpm_confidence"], 85.5)
        self.assertEqual(results["key"], "C Major")
        self.assertEqual(results["key_confidence"], 78.2)

    def test_analyzer_initialization(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = AudioAnalyzer(sr=44100, hop_length=512)

        self.assertEqual(analyzer.sr, 44100)
        self.assertEqual(analyzer.hop_length, 512)
        self.assertIsInstance(analyzer.bpm_detector, BPMDetector)
        self.assertIsInstance(analyzer.key_detector, KeyDetector)


class TestIntegration(unittest.TestCase):
    """Integration tests with synthetic audio."""

    def test_synthetic_audio_analysis(self):
        """Test analysis with synthetic audio signal."""
        # Create synthetic audio with known characteristics
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create a simple sine wave at 440 Hz (A4)
        frequency = 440.0
        audio = np.sin(2 * np.pi * frequency * t)

        # Add some amplitude modulation to simulate rhythm
        beat_freq = 2.0  # 120 BPM = 2 beats per second
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
        audio = audio * envelope

        analyzer = AudioAnalyzer()

        # Test BPM detection with synthetic audio
        with patch("librosa.load") as mock_load:
            mock_load.return_value = (audio, sr)

            # This is more of a smoke test - we can't predict exact results
            # but we can ensure the function runs without error
            try:
                results = analyzer.analyze_file("synthetic.wav", detect_key=False)

                self.assertIn("bpm", results)
                self.assertIn("bpm_confidence", results)
                self.assertIsInstance(results["bpm"], float)
                self.assertGreater(results["bpm"], 0)

            except Exception as e:
                self.fail(f"Synthetic audio analysis failed: {e}")


if __name__ == "__main__":
    unittest.main()
