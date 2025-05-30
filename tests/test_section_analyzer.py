"""Tests for section analyzer module."""

import unittest
import numpy as np
from src.bpm_detector.section_analyzer import SectionAnalyzer


class TestSectionAnalyzer(unittest.TestCase):
    """Test cases for SectionAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SectionAnalyzer(hop_length=512)
        self.sr = 22050
        
        # Create test sections
        self.test_sections = [
            {
                'start_time': 0.0, 'end_time': 10.0, 'type': 'intro',
                'confidence': 0.8, 'characteristics': {'energy': 0.2, 'spectral_complexity': 0.3}
            },
            {
                'start_time': 10.0, 'end_time': 30.0, 'type': 'verse',
                'confidence': 0.9, 'characteristics': {'energy': 0.5, 'spectral_complexity': 0.6}
            },
            {
                'start_time': 30.0, 'end_time': 50.0, 'type': 'chorus',
                'confidence': 0.95, 'characteristics': {'energy': 0.9, 'spectral_complexity': 0.8}
            },
        ]
        
        # Create synthetic audio for testing
        duration = 50  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_refine_section_labels_with_spectral_analysis(self):
        """Test section label refinement with spectral analysis."""
        refined = self.analyzer.refine_section_labels_with_spectral_analysis(
            self.test_audio, self.sr, self.test_sections.copy()
        )
        
        # Check that refinement preserves structure
        self.assertIsInstance(refined, list)
        self.assertEqual(len(refined), len(self.test_sections))
        
        # Check that sections still have required fields
        for section in refined:
            self.assertIn('type', section)
            self.assertIn('confidence', section)
            self.assertIn('start_time', section)
            self.assertIn('end_time', section)
            self.assertIn('spectral_flux', section)

    def test_classify_instrumental_subtype(self):
        """Test instrumental subtype classification."""
        # Create test section with instrumental characteristics
        instrumental_section = {
            'type': 'instrumental',
            'characteristics': {
                'energy': 0.7,
                'spectral_complexity': 0.8,
                'harmonic_content': 0.6,
                'rhythmic_density': 0.9
            }
        }
        
        spectral_features = {
            'spectral_centroid': np.array([2000.0, 2100.0, 1900.0]),
            'spectral_rolloff': np.array([4000.0, 4200.0, 3800.0]),
            'mfcc': np.random.randn(13, 3)
        }
        
        subtype = self.analyzer.classify_instrumental_subtype(
            instrumental_section, spectral_features
        )
        
        # Should return a valid subtype
        self.assertIsInstance(subtype, str)
        valid_subtypes = ['solo', 'breakdown', 'buildup', 'interlude']
        self.assertIn(subtype, valid_subtypes)

    def test_enhance_outro_detection(self):
        """Test enhanced outro detection."""
        # Create sections with potential outro at the end
        test_sections = [
            {'start_time': 0.0, 'end_time': 30.0, 'type': 'verse', 'energy_level': 0.5},
            {'start_time': 30.0, 'end_time': 45.0, 'type': 'chorus', 'energy_level': 0.8},
            {'start_time': 45.0, 'end_time': 50.0, 'type': 'verse', 'energy_level': 0.3},  # Potential outro
        ]
        
        enhanced = self.analyzer.enhance_outro_detection(test_sections, self.test_audio, self.sr)
        
        # Should preserve structure
        self.assertIsInstance(enhanced, list)
        self.assertEqual(len(enhanced), len(test_sections))

    def test_detect_fade_ending(self):
        """Test fade ending detection."""
        # Create a test section
        test_section = {
            'start_time': 40.0,
            'end_time': 50.0,
            'duration': 10.0
        }
        
        # Create audio with fade out
        fade_audio = self.test_audio.copy()
        fade_start = int(45.0 * self.sr)
        fade_audio[fade_start:] *= np.linspace(1.0, 0.1, len(fade_audio) - fade_start)
        
        is_fade = self.analyzer.detect_fade_ending(test_section, fade_audio, self.sr)
        
        # Should return a boolean
        self.assertIsInstance(is_fade, bool)

    def test_detect_harmonic_resolution(self):
        """Test harmonic resolution detection."""
        test_section = {
            'start_time': 40.0,
            'end_time': 50.0
        }
        
        has_resolution = self.analyzer.detect_harmonic_resolution(test_section, self.test_audio, self.sr)
        
        # Should return a boolean
        self.assertIsInstance(has_resolution, bool)

    def test_detect_chorus_hooks(self):
        """Test chorus hook detection."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'energy_level': 0.4, 'brightness': 0.5},
            {'type': 'verse', 'start_time': 15.0, 'end_time': 23.0, 'duration': 8.0,
             'energy_level': 0.7, 'brightness': 0.7},  # Should become chorus (hook pattern)
            {'type': 'bridge', 'start_time': 23.0, 'end_time': 38.0, 'duration': 15.0,
             'energy_level': 0.6, 'brightness': 0.6},
        ]
        
        processed = self.analyzer.detect_chorus_hooks(test_sections.copy())
        
        # Check that hook detection works
        self.assertIsInstance(processed, list)
        self.assertEqual(len(processed), len(test_sections))

    def test_analyze_form(self):
        """Test song form analysis."""
        form_analysis = self.analyzer.analyze_form(self.test_sections)
        
        # Check required fields
        required_fields = ['form', 'section_count', 'total_duration', 'structure_complexity']
        for field in required_fields:
            self.assertIn(field, form_analysis)
        
        # Check form string
        self.assertIsInstance(form_analysis['form'], str)
        self.assertGreater(len(form_analysis['form']), 0)
        
        # Check section count
        self.assertIsInstance(form_analysis['section_count'], int)
        self.assertGreater(form_analysis['section_count'], 0)
        
        # Check total duration
        self.assertIsInstance(form_analysis['total_duration'], (int, float))
        self.assertGreater(form_analysis['total_duration'], 0)
        
        # Check complexity score
        self.assertIsInstance(form_analysis['structure_complexity'], (int, float))
        self.assertGreaterEqual(form_analysis['structure_complexity'], 0.0)
        self.assertLessEqual(form_analysis['structure_complexity'], 1.0)

    def test_section_to_letter(self):
        """Test section to letter conversion."""
        # Test standard mappings
        self.assertEqual(self.analyzer._section_to_letter('intro'), 'I')
        self.assertEqual(self.analyzer._section_to_letter('verse'), 'A')
        self.assertEqual(self.analyzer._section_to_letter('chorus'), 'B')
        self.assertEqual(self.analyzer._section_to_letter('bridge'), 'C')
        self.assertEqual(self.analyzer._section_to_letter('outro'), 'O')
        
        # Test unknown section type
        unknown_letter = self.analyzer._section_to_letter('unknown_type')
        self.assertIsInstance(unknown_letter, str)
        self.assertEqual(len(unknown_letter), 1)

    def test_calculate_structural_complexity(self):
        """Test structural complexity calculation."""
        complexity = self.analyzer._calculate_structural_complexity(self.test_sections)
        
        # Should return a float between 0 and 1
        self.assertIsInstance(complexity, (int, float))
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
        
        # Test with simple structure (should be less complex)
        simple_sections = [
            {'type': 'verse', 'start_time': 0, 'end_time': 30},
            {'type': 'chorus', 'start_time': 30, 'end_time': 60},
            {'type': 'verse', 'start_time': 60, 'end_time': 90},
            {'type': 'chorus', 'start_time': 90, 'end_time': 120}
        ]
        simple_complexity = self.analyzer._calculate_structural_complexity(simple_sections)
        
        # Complex structure should have higher complexity
        self.assertGreaterEqual(complexity, simple_complexity)

    def test_summarize_sections(self):
        """Test section summarization."""
        summary = self.analyzer.summarize_sections(self.test_sections)
        
        # Should return a string
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        
        # Should contain section information
        self.assertIn('intro', summary.lower())
        self.assertIn('verse', summary.lower())
        self.assertIn('chorus', summary.lower())

    def test_calculate_energy_scale(self):
        """Test energy scale calculation."""
        energy_scale = self.analyzer.calculate_energy_scale(self.test_audio)
        
        # Should return a dictionary with energy statistics
        self.assertIsInstance(energy_scale, dict)
        
        expected_keys = ['min_energy', 'max_energy', 'mean_energy', 'energy_range']
        for key in expected_keys:
            self.assertIn(key, energy_scale)
            self.assertIsInstance(energy_scale[key], (int, float))
        
        # Check logical relationships
        self.assertLessEqual(energy_scale['min_energy'], energy_scale['mean_energy'])
        self.assertLessEqual(energy_scale['mean_energy'], energy_scale['max_energy'])
        self.assertGreaterEqual(energy_scale['energy_range'], 0)

    def test_empty_sections_handling(self):
        """Test behavior with empty sections list."""
        # Test form analysis with empty sections
        form_analysis = self.analyzer.analyze_form([])
        self.assertEqual(form_analysis['section_count'], 0)
        self.assertEqual(form_analysis['total_duration'], 0.0)
        
        # Test summarization with empty sections
        summary = self.analyzer.summarize_sections([])
        self.assertIsInstance(summary, str)
        self.assertIn('0 sections', summary)

    def test_characteristics_list_format_handling(self):
        """Test handling of characteristics in list format."""
        section_with_list_characteristics = {
            'type': 'instrumental',
            'characteristics': [0.5, 0.6, 0.7, 0.8]  # List format instead of dict
        }
        
        # Should handle list format gracefully
        subtype = self.analyzer.classify_instrumental_subtype(section_with_list_characteristics)
        self.assertIsInstance(subtype, str)
        valid_subtypes = ['solo', 'breakdown', 'buildup', 'interlude']
        self.assertIn(subtype, valid_subtypes)


if __name__ == '__main__':
    unittest.main()