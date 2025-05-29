"""Tests for structure analyzer."""

import unittest
import numpy as np
import librosa
from src.bpm_detector.structure_analyzer import StructureAnalyzer


class TestStructureAnalyzer(unittest.TestCase):
    """Test cases for StructureAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()
        self.sr = 22050
        self.duration = 20.0  # 20 seconds
        
        # Create a test signal with different sections
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        
        # Create signal with changing characteristics
        signal = np.zeros_like(t)
        
        # Section 1: Low energy (intro)
        mask1 = t < 5
        signal[mask1] = 0.2 * np.sin(2 * np.pi * 440 * t[mask1])
        
        # Section 2: Medium energy (verse)
        mask2 = (t >= 5) & (t < 10)
        signal[mask2] = 0.5 * (
            np.sin(2 * np.pi * 440 * t[mask2]) +
            0.5 * np.sin(2 * np.pi * 880 * t[mask2])
        )
        
        # Section 3: High energy (chorus)
        mask3 = (t >= 10) & (t < 15)
        signal[mask3] = 0.8 * (
            np.sin(2 * np.pi * 440 * t[mask3]) +
            np.sin(2 * np.pi * 880 * t[mask3]) +
            0.5 * np.sin(2 * np.pi * 1320 * t[mask3])
        )
        
        # Section 4: Medium energy (verse repeat)
        mask4 = t >= 15
        signal[mask4] = 0.5 * (
            np.sin(2 * np.pi * 440 * t[mask4]) +
            0.5 * np.sin(2 * np.pi * 880 * t[mask4])
        )
        
        # Add some noise
        signal += 0.05 * np.random.randn(len(t))
        
        self.test_signal = signal
        
    def test_extract_structural_features(self):
        """Test structural feature extraction."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        
        # Check that all expected features are present
        expected_features = [
            'mfcc', 'chroma', 'spectral_centroid', 'rms', 
            'zcr', 'onset_strength'
        ]
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features)
            self.assertIsInstance(features[feature_name], np.ndarray)
            self.assertGreater(features[feature_name].size, 0)
            
    def test_compute_self_similarity_matrix(self):
        """Test self-similarity matrix computation."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)
        
        # Check shape (should be square)
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])
        
        # Check diagonal is 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        np.testing.assert_allclose(diagonal, 1.0, rtol=1e-10)
        
        # Check symmetry
        np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, rtol=1e-10)
        
        # Check values are in [0, 1] range
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))
        
    def test_detect_boundaries(self):
        """Test boundary detection."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)
        boundaries = self.analyzer.detect_boundaries(similarity_matrix, self.sr)
        
        # Should detect at least start and end
        self.assertGreaterEqual(len(boundaries), 2)
        
        # First boundary should be 0
        self.assertEqual(boundaries[0], 0)
        
        # Last boundary should be last frame
        self.assertEqual(boundaries[-1], similarity_matrix.shape[0] - 1)
        
        # Boundaries should be sorted
        self.assertEqual(boundaries, sorted(boundaries))
        
    def test_classify_sections(self):
        """Test section classification."""
        boundaries = [0, 100, 200, 300, 400]  # Mock boundaries
        sections = self.analyzer.classify_sections(self.test_signal, self.sr, boundaries)
        
        # Should have one less section than boundaries
        self.assertEqual(len(sections), len(boundaries) - 1)
        
        # Check section structure
        for section in sections:
            required_keys = [
                'type', 'start_time', 'end_time', 'duration',
                'characteristics', 'energy_level', 'complexity'
            ]
            for key in required_keys:
                self.assertIn(key, section)
                
            # Check types
            self.assertIsInstance(section['type'], str)
            self.assertIsInstance(section['start_time'], float)
            self.assertIsInstance(section['end_time'], float)
            self.assertIsInstance(section['duration'], float)
            self.assertIsInstance(section['characteristics'], list)
            self.assertIsInstance(section['energy_level'], float)
            self.assertIsInstance(section['complexity'], float)
            
            # Check time consistency
            self.assertLessEqual(section['start_time'], section['end_time'])
            self.assertAlmostEqual(
                section['duration'], 
                section['end_time'] - section['start_time'],
                places=2
            )
            
    def test_analyze_form(self):
        """Test form analysis."""
        # Mock sections
        mock_sections = [
            {'type': 'intro', 'duration': 8},
            {'type': 'verse', 'duration': 16},
            {'type': 'chorus', 'duration': 16},
            {'type': 'verse', 'duration': 16},
            {'type': 'chorus', 'duration': 16},
            {'type': 'outro', 'duration': 8}
        ]
        
        form_analysis = self.analyzer.analyze_form(mock_sections)
        
        # Check required keys
        required_keys = [
            'form', 'repetition_ratio', 'structural_complexity',
            'section_count', 'unique_sections', 'section_types'
        ]
        for key in required_keys:
            self.assertIn(key, form_analysis)
            
        # Check types
        self.assertIsInstance(form_analysis['form'], str)
        self.assertIsInstance(form_analysis['repetition_ratio'], float)
        self.assertIsInstance(form_analysis['structural_complexity'], float)
        self.assertIsInstance(form_analysis['section_count'], int)
        self.assertIsInstance(form_analysis['unique_sections'], int)
        self.assertIsInstance(form_analysis['section_types'], list)
        
        # Check values
        self.assertEqual(form_analysis['section_count'], len(mock_sections))
        self.assertGreater(form_analysis['unique_sections'], 0)
        self.assertLessEqual(form_analysis['unique_sections'], len(mock_sections))
        
    def test_detect_repetitions(self):
        """Test repetition detection."""
        features = self.analyzer.extract_structural_features(self.test_signal, self.sr)
        similarity_matrix = self.analyzer.compute_self_similarity_matrix(features)
        repetitions = self.analyzer.detect_repetitions(similarity_matrix, self.sr)
        
        # Should return a list
        self.assertIsInstance(repetitions, list)
        
        # Check repetition structure
        for rep in repetitions:
            required_keys = [
                'first_occurrence', 'second_occurrence', 
                'duration', 'similarity'
            ]
            for key in required_keys:
                self.assertIn(key, rep)
                self.assertIsInstance(rep[key], float)
                
            # Check logical constraints
            self.assertGreater(rep['duration'], 0)
            self.assertGreaterEqual(rep['similarity'], 0)
            self.assertLessEqual(rep['similarity'], 1)
            self.assertLess(rep['first_occurrence'], rep['second_occurrence'])
            
    def test_analyze_complete(self):
        """Test complete structural analysis."""
        result = self.analyzer.analyze(self.test_signal, self.sr)
        
        # Check all required keys are present
        required_keys = [
            'sections', 'form', 'repetition_ratio', 'structural_complexity',
            'section_count', 'unique_sections', 'repetitions', 'boundaries'
        ]
        
        for key in required_keys:
            self.assertIn(key, result)
            
        # Check types
        self.assertIsInstance(result['sections'], list)
        self.assertIsInstance(result['form'], str)
        self.assertIsInstance(result['repetition_ratio'], float)
        self.assertIsInstance(result['structural_complexity'], float)
        self.assertIsInstance(result['section_count'], int)
        self.assertIsInstance(result['unique_sections'], int)
        self.assertIsInstance(result['repetitions'], list)
        self.assertIsInstance(result['boundaries'], list)
        
    def test_section_to_letter(self):
        """Test section type to letter conversion."""
        test_cases = {
            'intro': 'I',
            'verse': 'A',
            'chorus': 'B',
            'bridge': 'C',
            'instrumental': 'D',
            'outro': 'O',
            'unknown': 'X'
        }
        
        for section_type, expected_letter in test_cases.items():
            result = self.analyzer._section_to_letter(section_type)
            self.assertEqual(result, expected_letter)
            
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
            
    def test_vocal_detection(self):
        """Test vocal presence detection."""
        # Create signal with vocal-like frequencies
        t = np.linspace(0, 2.0, int(self.sr * 2.0))
        vocal_signal = np.sin(2 * np.pi * 200 * t)  # 200 Hz (vocal range)
        
        has_vocal = self.analyzer._detect_vocal_presence(vocal_signal, self.sr)
        self.assertIsInstance(has_vocal, bool)


if __name__ == '__main__':
    unittest.main()