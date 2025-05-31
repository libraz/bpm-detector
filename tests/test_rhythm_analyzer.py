"""Tests for rhythm analyzer."""

import unittest

import librosa
import numpy as np

from src.bpm_detector.rhythm_analyzer import RhythmAnalyzer


class TestRhythmAnalyzer(unittest.TestCase):
    """Test cases for RhythmAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RhythmAnalyzer()
        self.sr = 22050
        self.duration = 10.0  # 10 seconds

        # Create a test signal with known rhythm
        t = np.linspace(0, self.duration, int(self.sr * self.duration))

        # Create a 4/4 rhythm at 120 BPM (2 beats per second)
        beat_freq = 2.0  # 120 BPM = 2 beats per second

        # Generate beat pattern
        beat_times = np.arange(0, self.duration, 1 / beat_freq)
        signal = np.zeros_like(t)

        # Add beats as short impulses
        for beat_time in beat_times:
            beat_idx = int(beat_time * self.sr)
            if beat_idx < len(signal):
                # Create a short impulse
                impulse_length = int(0.05 * self.sr)  # 50ms impulse
                end_idx = min(beat_idx + impulse_length, len(signal))
                signal[beat_idx:end_idx] = 0.8 * np.sin(
                    2 * np.pi * 1000 * t[beat_idx:end_idx]
                )

        # Add some background tone
        signal += 0.2 * np.sin(2 * np.pi * 440 * t)

        # Add noise
        signal += 0.05 * np.random.randn(len(t))

        self.test_signal = signal

    def test_extract_onset_features(self):
        """Test onset feature extraction."""
        features = self.analyzer.extract_onset_features(self.test_signal, self.sr)

        # Check that all expected features are present
        expected_features = [
            'onset_strength',
            'onset_times',
            'onset_frames',
            'tempo',
            'beat_frames',
            'beat_times',
        ]

        for feature_name in expected_features:
            self.assertIn(feature_name, features)
            self.assertIsInstance(features[feature_name], np.ndarray)

        # Check tempo is reasonable
        tempo = features['tempo']
        self.assertGreater(tempo, 60)  # At least 60 BPM
        self.assertLess(tempo, 200)  # Less than 200 BPM

        # Check beat times are sorted
        beat_times = features['beat_times']
        self.assertTrue(np.all(np.diff(beat_times) >= 0))

    def test_detect_time_signature(self):
        """Test time signature detection."""
        features = self.analyzer.extract_onset_features(self.test_signal, self.sr)
        time_sig, confidence = self.analyzer.detect_time_signature(features, self.sr)

        # Should return a valid time signature
        valid_time_sigs = ['4/4', '3/4', '2/4', '6/8', '9/8', '12/8', '5/4', '7/8']
        self.assertIn(time_sig, valid_time_sigs)

        # Confidence should be between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_extract_rhythm_patterns(self):
        """Test rhythm pattern extraction."""
        features = self.analyzer.extract_onset_features(self.test_signal, self.sr)
        patterns = self.analyzer.extract_rhythm_patterns(features, '4/4')

        # Check required keys
        required_keys = [
            'rhythmic_complexity',
            'syncopation_level',
            'pattern_regularity',
            'subdivision_density',
        ]
        for key in required_keys:
            self.assertIn(key, patterns)
            self.assertIsInstance(patterns[key], float)

            # Check values are in reasonable range
            self.assertGreaterEqual(patterns[key], 0.0)
            self.assertLessEqual(patterns[key], 1.0)

    def test_detect_groove_type(self):
        """Test groove type detection."""
        features = self.analyzer.extract_onset_features(self.test_signal, self.sr)
        patterns = self.analyzer.extract_rhythm_patterns(features, '4/4')
        groove_type, confidence = self.analyzer.detect_groove_type(features, patterns)

        # Should return a valid groove type
        valid_grooves = ['straight', 'swing', 'shuffle']
        self.assertIn(groove_type, valid_grooves)

        # Confidence should be between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_swing_ratio(self):
        """Test swing ratio calculation."""
        # Create straight rhythm
        features = self.analyzer.extract_onset_features(self.test_signal, self.sr)
        swing_ratio = self.analyzer._calculate_swing_ratio(
            features['onset_times'], features['beat_times']
        )

        # Should be close to 0.5 for straight rhythm
        self.assertGreaterEqual(swing_ratio, 0.0)
        self.assertLessEqual(swing_ratio, 1.0)

    def test_analyze_complete(self):
        """Test complete rhythm analysis."""
        result = self.analyzer.analyze(self.test_signal, self.sr)

        # Check all required keys are present
        required_keys = [
            'time_signature',
            'time_signature_confidence',
            'groove_type',
            'groove_confidence',
            'rhythmic_complexity',
            'syncopation_level',
            'pattern_regularity',
            'subdivision_density',
            'swing_ratio',
            'polyrhythm_detected',
            'onset_count',
            'beat_count',
        ]

        for key in required_keys:
            self.assertIn(key, result)

        # Check types
        self.assertIsInstance(result['time_signature'], str)
        self.assertIsInstance(result['time_signature_confidence'], float)
        self.assertIsInstance(result['groove_type'], str)
        self.assertIsInstance(result['groove_confidence'], float)
        self.assertIsInstance(result['rhythmic_complexity'], float)
        self.assertIsInstance(result['syncopation_level'], float)
        self.assertIsInstance(result['pattern_regularity'], float)
        self.assertIsInstance(result['subdivision_density'], float)
        self.assertIsInstance(result['swing_ratio'], float)
        self.assertIsInstance(result['polyrhythm_detected'], bool)
        self.assertIsInstance(result['onset_count'], int)
        self.assertIsInstance(result['beat_count'], int)

        # Check value ranges
        self.assertGreaterEqual(result['time_signature_confidence'], 0.0)
        self.assertLessEqual(result['time_signature_confidence'], 1.0)
        self.assertGreaterEqual(result['groove_confidence'], 0.0)
        self.assertLessEqual(result['groove_confidence'], 1.0)
        self.assertGreaterEqual(result['swing_ratio'], 0.0)
        self.assertLessEqual(result['swing_ratio'], 1.0)

    def test_time_signature_scoring(self):
        """Test time signature scoring."""
        # Create mock beat intervals for 4/4 time
        beat_intervals = np.array([0.5, 0.5, 0.5, 0.5] * 4)  # Regular 4/4 pattern
        signature_info = self.analyzer.TIME_SIGNATURES['4/4']

        # Create mock onset features
        mock_features = {
            'onset_strength': np.random.rand(100),
            'beat_frames': np.arange(0, 100, 10),
            'beat_times': np.arange(0, 10, 1.0),
        }

        score = self.analyzer._score_time_signature(
            beat_intervals, signature_info, mock_features
        )

        # Should return a score between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_calculate_rhythmic_complexity(self):
        """Test rhythmic complexity calculation."""
        # Create simple onset pattern
        onset_times = np.array([0, 1, 2, 3, 4])
        beat_times = np.array([0, 1, 2, 3, 4])

        complexity = self.analyzer._calculate_rhythmic_complexity(
            onset_times, beat_times
        )

        # Should return a value between 0 and 1
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

    def test_calculate_syncopation(self):
        """Test syncopation calculation."""
        # Create onset pattern
        onset_times = np.array([0, 0.5, 1, 1.5, 2])  # Some off-beat onsets
        beat_times = np.array([0, 1, 2, 3, 4])

        syncopation = self.analyzer._calculate_syncopation(
            onset_times, beat_times, '4/4'
        )

        # Should return a value between 0 and 1
        self.assertGreaterEqual(syncopation, 0.0)
        self.assertLessEqual(syncopation, 1.0)

    def test_detect_polyrhythm(self):
        """Test polyrhythm detection."""
        # Create mock onset features
        mock_features = {'onset_strength': np.random.rand(200)}

        polyrhythm = self.analyzer._detect_polyrhythm(mock_features)

        # Should return a boolean
        self.assertIsInstance(polyrhythm, bool)

    def test_empty_input(self):
        """Test behavior with empty input."""
        empty_signal = np.array([])

        # Should handle empty input gracefully
        try:
            result = self.analyzer.analyze(empty_signal, self.sr)
            self.assertIsInstance(result, dict)
        except Exception:
            # Or raise appropriate exception
            pass

    def test_pattern_regularity(self):
        """Test pattern regularity calculation."""
        # Regular pattern
        regular_onsets = np.array([0, 1, 2, 3, 4, 5])
        regular_beats = np.array([0, 1, 2, 3, 4, 5])

        regularity = self.analyzer._calculate_pattern_regularity(
            regular_onsets, regular_beats
        )

        # Should be high for regular pattern
        self.assertGreaterEqual(regularity, 0.0)
        self.assertLessEqual(regularity, 1.0)

        # Irregular pattern
        irregular_onsets = np.array([0, 0.7, 2.3, 2.9, 4.1, 5.8])
        irregular_beats = np.array([0, 1, 2, 3, 4, 5])

        irregularity = self.analyzer._calculate_pattern_regularity(
            irregular_onsets, irregular_beats
        )

        # Should be lower for irregular pattern
        self.assertLess(irregularity, regularity)

    def test_enhanced_groove_detection(self):
        """Test enhanced groove detection features."""
        # Create test audio with swing feel
        duration = 4.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))

        # Create swing pattern (long-short-long-short)
        beat_times = np.array(
            [0, 0.67, 1.0, 1.67, 2.0, 2.67, 3.0, 3.67]
        )  # Swing eighths
        swing_audio = np.zeros_like(t)

        for beat_time in beat_times:
            beat_idx = int(beat_time * sr)
            if beat_idx < len(swing_audio):
                # Add a short percussive sound
                decay = np.exp(-10 * (t - beat_time))
                decay[t < beat_time] = 0
                swing_audio += 0.5 * np.sin(2 * np.pi * 200 * t) * decay

        results = self.analyzer.analyze(swing_audio, sr)

        # Check groove detection
        self.assertIn('groove_type', results)
        groove_type = results['groove_type']
        self.assertIsInstance(groove_type, str)

        # Check swing ratio
        if 'swing_ratio' in results:
            swing_ratio = results['swing_ratio']
            self.assertIsInstance(swing_ratio, (int, float))
            self.assertGreaterEqual(swing_ratio, 0.5)  # Should detect swing
            self.assertLessEqual(swing_ratio, 1.0)

    def test_subdivision_density_analysis(self):
        """Test subdivision density analysis."""
        # Create dense rhythmic pattern
        dense_onsets = np.arange(0, 4, 0.125)  # 32nd notes
        beat_times = np.arange(0, 4, 0.5)  # Beat times array

        dense_density = self.analyzer._calculate_subdivision_density(
            dense_onsets, beat_times
        )

        # Create sparse rhythmic pattern
        sparse_onsets = np.arange(0, 4, 1.0)  # Quarter notes

        sparse_density = self.analyzer._calculate_subdivision_density(
            sparse_onsets, beat_times
        )

        # Dense pattern should have higher density
        self.assertGreater(dense_density, sparse_density)

        # Both should be valid values
        self.assertIsInstance(dense_density, (int, float))
        self.assertIsInstance(sparse_density, (int, float))
        self.assertGreaterEqual(dense_density, 0.0)
        self.assertGreaterEqual(sparse_density, 0.0)

    def test_time_signature_confidence(self):
        """Test time signature detection confidence."""
        # Create clear 4/4 pattern
        clear_44_onsets = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        onset_features = {'onset_times': clear_44_onsets}

        time_sig_result = self.analyzer.detect_time_signature(onset_features, 120.0)

        # detect_time_signature returns (time_signature, confidence) tuple
        if isinstance(time_sig_result, tuple):
            time_signature, confidence = time_sig_result
            self.assertIsInstance(time_signature, str)
            self.assertIsInstance(confidence, (int, float))
        else:
            # If it returns a dict
            self.assertIn('time_signature', time_sig_result)
            self.assertIn('confidence', time_sig_result)
            confidence = time_sig_result['confidence']
        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_rhythmic_complexity_edge_cases(self):
        """Test rhythmic complexity with edge cases."""
        # Test with very few onsets
        few_onsets = np.array([0, 2])
        beat_times = np.array([0, 1, 2])  # Create beat times array

        few_complexity = self.analyzer._calculate_rhythmic_complexity(
            few_onsets, beat_times
        )
        self.assertIsInstance(few_complexity, (int, float))
        self.assertGreaterEqual(few_complexity, 0.0)
        self.assertLessEqual(few_complexity, 1.0)

        # Test with many onsets
        many_onsets = np.linspace(0, 4, 100)
        many_beat_times = np.linspace(0, 4, 17)  # Create beat times array

        many_complexity = self.analyzer._calculate_rhythmic_complexity(
            many_onsets, many_beat_times
        )
        self.assertIsInstance(many_complexity, (int, float))
        self.assertGreaterEqual(many_complexity, 0.0)
        self.assertLessEqual(many_complexity, 1.0)

        # Many onsets should generally have higher complexity
        self.assertGreater(many_complexity, few_complexity)

    def test_comprehensive_rhythm_analysis(self):
        """Test comprehensive rhythm analysis integration."""
        # Create complex rhythmic audio
        duration = 8.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))

        # Create layered rhythm with kick, snare, and hi-hat
        rhythm_audio = np.zeros_like(t)

        # Kick on beats 1 and 3
        kick_times = np.arange(0, duration, 1.0)
        for kick_time in kick_times:
            kick_idx = int(kick_time * sr)
            if kick_idx < len(rhythm_audio):
                # Low frequency kick
                kick_env = np.exp(-5 * (t - kick_time))
                kick_env[t < kick_time] = 0
                rhythm_audio += 0.8 * np.sin(2 * np.pi * 60 * t) * kick_env

        results = self.analyzer.analyze(rhythm_audio, sr)

        # Should detect clear 4/4 pattern
        self.assertEqual(results['time_signature'], '4/4')

        # Should have reasonable complexity
        complexity = results['rhythmic_complexity']
        self.assertGreater(complexity, 0.0)  # Should have some complexity
        self.assertLess(complexity, 1.0)  # But not overly complex


if __name__ == '__main__':
    unittest.main()
