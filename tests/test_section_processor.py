"""Tests for section processor module."""

import unittest

import numpy as np

from src.bpm_detector.section_processor import SectionProcessor


class TestSectionProcessor(unittest.TestCase):
    """Test cases for SectionProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SectionProcessor(hop_length=512)
        self.sr = 22050

        # Create test sections
        self.test_sections = [
            {
                'start_time': 0.0,
                'end_time': 10.0,
                'type': 'intro',
                'confidence': 0.8,
                'characteristics': {'energy': 0.2, 'spectral_complexity': 0.3},
            },
            {
                'start_time': 10.0,
                'end_time': 30.0,
                'type': 'verse',
                'confidence': 0.9,
                'characteristics': {'energy': 0.5, 'spectral_complexity': 0.6},
            },
            {
                'start_time': 30.0,
                'end_time': 50.0,
                'type': 'chorus',
                'confidence': 0.95,
                'characteristics': {'energy': 0.9, 'spectral_complexity': 0.8},
            },
            {
                'start_time': 50.0,
                'end_time': 70.0,
                'type': 'verse',
                'confidence': 0.85,
                'characteristics': {'energy': 0.5, 'spectral_complexity': 0.6},
            },
            {
                'start_time': 70.0,
                'end_time': 90.0,
                'type': 'chorus',
                'confidence': 0.9,
                'characteristics': {'energy': 0.9, 'spectral_complexity': 0.8},
            },
            {
                'start_time': 90.0,
                'end_time': 110.0,
                'type': 'bridge',
                'confidence': 0.7,
                'characteristics': {'energy': 0.6, 'spectral_complexity': 0.7},
            },
            {
                'start_time': 110.0,
                'end_time': 130.0,
                'type': 'chorus',
                'confidence': 0.92,
                'characteristics': {'energy': 0.9, 'spectral_complexity': 0.8},
            },
            {
                'start_time': 130.0,
                'end_time': 140.0,
                'type': 'outro',
                'confidence': 0.75,
                'characteristics': {'energy': 0.2, 'spectral_complexity': 0.3},
            },
        ]

        # Create synthetic audio for testing
        duration = 140  # seconds
        t = np.linspace(0, duration, int(self.sr * duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_post_process_sections_basic(self):
        """Test basic section post-processing."""
        processed = self.processor.post_process_sections(
            self.test_sections.copy(), total_duration=140.0, merge_threshold=5.0
        )

        # Check basic structure
        self.assertIsInstance(processed, list)
        self.assertGreater(len(processed), 0)

        # Check section structure
        for section in processed:
            self.assertIsInstance(section, dict)
            required_keys = ['start_time', 'end_time', 'type', 'confidence']
            for key in required_keys:
                self.assertIn(key, section)

            # Check time ordering
            self.assertLessEqual(section['start_time'], section['end_time'])

            # Check confidence range
            self.assertGreaterEqual(section['confidence'], 0.0)
            self.assertLessEqual(section['confidence'], 1.0)

    def test_smart_merge_types(self):
        """Test smart section type merging."""
        # Test merging similar types
        merged_type = self.processor._smart_merge_types('verse', 'verse', 20.0, 15.0)
        self.assertEqual(merged_type, 'verse')

        # Test merging short intro with verse
        merged_type = self.processor._smart_merge_types('intro', 'verse', 3.0, 20.0)
        self.assertEqual(merged_type, 'verse')

        # Test merging short outro with chorus
        merged_type = self.processor._smart_merge_types('chorus', 'outro', 25.0, 4.0)
        self.assertEqual(merged_type, 'chorus')

        # Test merging different types with similar durations
        merged_type = self.processor._smart_merge_types('verse', 'bridge', 20.0, 18.0)
        self.assertIn(merged_type, ['verse', 'bridge'])

    def test_refine_section_labels_with_spectral_analysis(self):
        """Test section label refinement with spectral analysis."""
        refined = self.processor.refine_section_labels_with_spectral_analysis(
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

    def test_classify_instrumental_subtype(self):
        """Test instrumental subtype classification."""
        # Create test section with instrumental characteristics
        instrumental_section = {
            'type': 'instrumental',
            'characteristics': {
                'energy': 0.7,
                'spectral_complexity': 0.8,
                'harmonic_content': 0.6,
                'rhythmic_density': 0.9,
            },
        }

        spectral_features = {
            'spectral_centroid': np.array([2000.0, 2100.0, 1900.0]),
            'spectral_rolloff': np.array([4000.0, 4200.0, 3800.0]),
            'mfcc': np.random.randn(13, 3),
        }

        subtype = self.processor._classify_instrumental_subtype(instrumental_section, spectral_features)

        # Should return a valid subtype
        self.assertIsInstance(subtype, str)
        valid_subtypes = ['solo', 'breakdown', 'buildup', 'interlude']
        self.assertIn(subtype, valid_subtypes)

    def test_analyze_form(self):
        """Test song form analysis."""
        form_analysis = self.processor.analyze_form(self.test_sections)

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
        self.assertEqual(self.processor._section_to_letter('intro'), 'I')
        self.assertEqual(self.processor._section_to_letter('verse'), 'A')
        self.assertEqual(self.processor._section_to_letter('chorus'), 'B')
        self.assertEqual(self.processor._section_to_letter('bridge'), 'C')
        self.assertEqual(self.processor._section_to_letter('outro'), 'O')

        # Test unknown section type
        unknown_letter = self.processor._section_to_letter('unknown_type')
        self.assertIsInstance(unknown_letter, str)
        self.assertEqual(len(unknown_letter), 1)

    def test_calculate_structural_complexity(self):
        """Test structural complexity calculation."""
        complexity = self.processor._calculate_structural_complexity(self.test_sections)

        # Should return a float between 0 and 1
        self.assertIsInstance(complexity, (int, float))
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

        # Test with simple structure (should be less complex)
        simple_sections = [
            {'type': 'verse', 'start_time': 0, 'end_time': 30},
            {'type': 'chorus', 'start_time': 30, 'end_time': 60},
            {'type': 'verse', 'start_time': 60, 'end_time': 90},
            {'type': 'chorus', 'start_time': 90, 'end_time': 120},
        ]
        simple_complexity = self.processor._calculate_structural_complexity(simple_sections)

        # Complex structure should have higher complexity
        self.assertGreaterEqual(complexity, simple_complexity)

    def test_summarize_sections(self):
        """Test section summarization."""
        summary = self.processor.summarize_sections(self.test_sections)

        # Should return a string
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

        # Should contain section information
        self.assertIn('intro', summary.lower())
        self.assertIn('verse', summary.lower())
        self.assertIn('chorus', summary.lower())

    def test_calculate_energy_scale(self):
        """Test energy scale calculation."""
        energy_scale = self.processor.calculate_energy_scale(self.test_audio)

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

    def test_merge_short_sections(self):
        """Test merging of short sections."""
        # Create sections with some very short ones
        sections_with_short = [
            {'start_time': 0.0, 'end_time': 15.0, 'type': 'intro', 'confidence': 0.8},
            {'start_time': 15.0, 'end_time': 17.0, 'type': 'verse', 'confidence': 0.6},  # Very short
            {'start_time': 17.0, 'end_time': 40.0, 'type': 'verse', 'confidence': 0.9},
            {'start_time': 40.0, 'end_time': 42.0, 'type': 'bridge', 'confidence': 0.5},  # Very short
            {'start_time': 42.0, 'end_time': 65.0, 'type': 'chorus', 'confidence': 0.95},
        ]

        processed = self.processor.post_process_sections(
            sections_with_short, total_duration=65.0, merge_threshold=5.0  # Merge sections shorter than 5 seconds
        )

        # Should have fewer sections after merging
        self.assertLessEqual(len(processed), len(sections_with_short))

        # All remaining sections should be longer than threshold or at boundaries
        for section in processed:
            duration = section['end_time'] - section['start_time']
            # Allow short sections only at the very beginning or end
            if section['start_time'] > 2.0 and section['end_time'] < 63.0:
                self.assertGreaterEqual(duration, 4.0)  # Some tolerance

    def test_empty_sections(self):
        """Test behavior with empty sections list."""
        processed = self.processor.post_process_sections([], total_duration=100.0)

        # Should handle empty input gracefully
        self.assertIsInstance(processed, list)
        self.assertEqual(len(processed), 0)

    def test_single_section(self):
        """Test processing with single section."""
        single_section = [self.test_sections[0]]

        processed = self.processor.post_process_sections(single_section, total_duration=10.0)

        # Should return the single section (possibly modified)
        self.assertIsInstance(processed, list)
        self.assertGreaterEqual(len(processed), 1)

    def test_form_analysis_edge_cases(self):
        """Test form analysis with edge cases."""
        # Test with minimal sections
        minimal_sections = [{'type': 'verse', 'start_time': 0, 'end_time': 60}]

        form_analysis = self.processor.analyze_form(minimal_sections)
        self.assertIsInstance(form_analysis, dict)
        self.assertEqual(form_analysis['section_count'], 1)

        # Test with many repeated sections
        repeated_sections = [{'type': 'verse', 'start_time': i * 10, 'end_time': (i + 1) * 10} for i in range(10)]

        form_analysis = self.processor.analyze_form(repeated_sections)
        self.assertIsInstance(form_analysis, dict)
        self.assertEqual(form_analysis['section_count'], 10)

    def test_bdB_merging_strict_tolerance(self):
        """Test B-D-B merging with strict 10% tolerance."""
        # 8 bars = (8 * 4 * 60) / 130.5 ≈ 14.7s
        # 10% tolerance = 1.47s
        # So 16.2s instrumental should NOT be absorbed

        test_sections = [
            {
                'type': 'chorus',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 15.0,
                'end_time': 31.2,
                'duration': 16.2,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # 16.2s > 14.7s + 1.47s
            {
                'type': 'chorus',
                'start_time': 31.2,
                'end_time': 46.2,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'chorus',
                'start_time': 46.2,
                'end_time': 61.2,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 61.2,
                'end_time': 75.0,
                'duration': 13.8,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # 13.8s < 14.7s + 1.47s
            {
                'type': 'chorus',
                'start_time': 75.0,
                'end_time': 90.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        processed = self.processor._merge_chorus_instrumental_chorus(test_sections, 130.5)

        # With max chorus length limit, the behavior changes:
        # - First B-D-B: 16.2s instrumental > 16.18s limit, so no merge
        # - Second B-D-B: would create 43.8s chorus > 29.4s limit, so no merge
        # Result: all sections remain separate due to limits
        self.assertEqual(len(processed), 6)

        # All sections should remain separate due to length constraints
        self.assertEqual(processed[0]['type'], 'chorus')
        self.assertEqual(processed[1]['type'], 'instrumental')
        self.assertEqual(processed[2]['type'], 'chorus')
        self.assertEqual(processed[3]['type'], 'chorus')
        self.assertEqual(processed[4]['type'], 'instrumental')
        self.assertEqual(processed[5]['type'], 'chorus')

    def test_max_chorus_length_limit(self):
        """Test maximum chorus length limit prevents super-long choruses."""
        # Create a scenario that would result in a super-long chorus without the limit
        # 16 bars = (16 * 4 * 60) / 130.5 ≈ 29.4s (max allowed)
        # Test case: B(15s) + D(14s) + B(15s) = 44s > 29.4s (should NOT merge)

        test_sections = [
            {
                'type': 'chorus',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 15.0,
                'end_time': 29.0,
                'duration': 14.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Within 8-bar limit
            {
                'type': 'chorus',
                'start_time': 29.0,
                'end_time': 44.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            # Total would be 44s > 29.4s limit
        ]

        processed = self.processor._merge_chorus_instrumental_chorus(test_sections, 130.5)

        # Should NOT merge due to length limit, so should remain 3 sections
        self.assertEqual(len(processed), 3)

        # All sections should remain separate
        self.assertEqual(processed[0]['type'], 'chorus')
        self.assertEqual(processed[1]['type'], 'instrumental')
        self.assertEqual(processed[2]['type'], 'chorus')

        # Durations should remain original
        self.assertAlmostEqual(processed[0]['duration'], 15.0, places=1)
        self.assertAlmostEqual(processed[1]['duration'], 14.0, places=1)
        self.assertAlmostEqual(processed[2]['duration'], 15.0, places=1)

    def test_acceptable_chorus_length_merging(self):
        """Test that acceptable length choruses still merge correctly."""
        # Create a scenario within the 16-bar limit
        # B(10s) + D(8s) + B(10s) = 28s < 29.4s (should merge)

        test_sections = [
            {
                'type': 'chorus',
                'start_time': 0.0,
                'end_time': 10.0,
                'duration': 10.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 10.0,
                'end_time': 18.0,
                'duration': 8.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Within 8-bar limit
            {
                'type': 'chorus',
                'start_time': 18.0,
                'end_time': 28.0,
                'duration': 10.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            # Total would be 28s < 29.4s limit
        ]

        processed = self.processor._merge_chorus_instrumental_chorus(test_sections, 130.5)

        # Should merge into 1 section
        self.assertEqual(len(processed), 1)

        # Should be a merged chorus
        merged = processed[0]
        self.assertEqual(merged['type'], 'chorus')
        self.assertAlmostEqual(merged['duration'], 28.0, places=1)
        self.assertEqual(merged['ascii_label'], 'Sabi')

    def test_yoru_ni_kakeru_super_long_chorus_prevention(self):
        """Test prevention of super-long chorus chains like in YOASOBI's '夜に駆ける' (Yoru ni Kakeru)."""
        # Simulate the problematic B-D-B-D-B pattern from "夜に駆ける"
        # Original issue: 73s-147s (74s total) creating a 40-bar super-long chorus

        yoru_ni_kakeru_pattern = [
            {
                'type': 'chorus',
                'start_time': 73.0,
                'end_time': 88.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 88.0,
                'end_time': 102.0,
                'duration': 14.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },
            {
                'type': 'chorus',
                'start_time': 102.0,
                'end_time': 117.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 117.0,
                'end_time': 132.0,
                'duration': 15.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },
            {
                'type': 'chorus',
                'start_time': 132.0,
                'end_time': 147.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        processed = self.processor._merge_chorus_instrumental_chorus(yoru_ni_kakeru_pattern, 130.5)

        # Calculate max allowed duration (16 bars)
        eight_bar = (8 * 4 * 60) / 130.5
        max_allowed = 16 * eight_bar / 8  # 16 bars ≈ 29.4s

        # Should prevent creation of super-long chorus (74s > 29.4s)
        # Should result in partial merging or no merging to stay within limits
        self.assertGreater(len(processed), 1, "Should not merge all into one super-long chorus")

        # Check that no individual chorus exceeds the 16-bar limit
        for i, section in enumerate(processed):
            if section['type'] == 'chorus':
                self.assertLessEqual(
                    section['duration'],
                    max_allowed + 1.0,  # Small tolerance
                    f"Chorus section {i+1} exceeds 16-bar limit: {section['duration']:.1f}s > {max_allowed:.1f}s",
                )

    def test_yoru_ni_kakeru_second_long_chorus_pattern(self):
        """Test the second problematic pattern from '夜に駆ける': 205s-261s (56s, 30 bars)."""
        # Simulate the second super-long chorus pattern

        second_pattern = [
            {
                'type': 'chorus',
                'start_time': 205.0,
                'end_time': 220.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 220.0,
                'end_time': 234.0,
                'duration': 14.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },
            {
                'type': 'chorus',
                'start_time': 234.0,
                'end_time': 249.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 249.0,
                'end_time': 261.0,
                'duration': 12.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },
        ]

        processed = self.processor._merge_chorus_instrumental_chorus(second_pattern, 130.5)

        # Calculate max allowed duration
        eight_bar = (8 * 4 * 60) / 130.5
        max_allowed = 16 * eight_bar / 8  # 16 bars ≈ 29.4s

        # Total would be 56s without limits, should be prevented
        total_duration = second_pattern[-1]['end_time'] - second_pattern[0]['start_time']
        self.assertEqual(total_duration, 56.0, "Test data verification: total should be 56s")

        # Should not create a single 56s chorus
        single_chorus_found = False
        for section in processed:
            if section['type'] == 'chorus' and section['duration'] > 50.0:
                single_chorus_found = True
                break

        self.assertFalse(single_chorus_found, "Should not create a single 56s super-long chorus")

        # All chorus sections should be within reasonable limits
        for section in processed:
            if section['type'] == 'chorus':
                self.assertLessEqual(
                    section['duration'],
                    max_allowed + 1.0,
                    f"Chorus duration {section['duration']:.1f}s exceeds 16-bar limit",
                )

    def test_consecutive_chorus_chain_breaking(self):
        """Test breaking consecutive chorus chains to restore instrumentals."""
        # Create a B-B-B pattern that should be converted to B-D-B
        consecutive_chorus_pattern = [
            {
                'type': 'chorus',
                'start_time': 73.0,
                'end_time': 88.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'chorus',
                'start_time': 88.0,
                'end_time': 102.0,
                'duration': 14.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Should become instrumental
            {
                'type': 'chorus',
                'start_time': 102.0,
                'end_time': 117.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        # Create mock audio data (simple sine wave)
        duration = 117.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)

        processed = self.processor._break_consecutive_chorus_chains(consecutive_chorus_pattern, y, sr, 130.5)

        # Should have vocal_ratio added to all sections
        for section in processed:
            self.assertIn('vocal_ratio', section)
            self.assertIsInstance(section['vocal_ratio'], (int, float))
            self.assertGreaterEqual(section['vocal_ratio'], 0.0)
            self.assertLessEqual(section['vocal_ratio'], 1.0)

        # Middle section should potentially be converted to instrumental
        # (depends on vocal detection, but structure should be preserved)
        self.assertEqual(len(processed), 3)

    def test_vocal_presence_detection(self):
        """Test vocal presence detection functionality."""
        # Create test audio signal
        duration = 10.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))

        # Simple sine wave (simulating instrumental)
        y_instrumental = 0.5 * np.sin(2 * np.pi * 440 * t)

        # More complex signal (simulating vocal)
        y_vocal = (
            0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1760 * t)
        )

        # Test vocal detection
        vocal_ratio_instrumental = self.processor._detect_vocal_presence(y_instrumental, sr, 0.0, 5.0)
        vocal_ratio_vocal = self.processor._detect_vocal_presence(y_vocal, sr, 0.0, 5.0)

        # Both should return valid ratios
        self.assertIsInstance(vocal_ratio_instrumental, (int, float))
        self.assertIsInstance(vocal_ratio_vocal, (int, float))
        self.assertGreaterEqual(vocal_ratio_instrumental, 0.0)
        self.assertLessEqual(vocal_ratio_instrumental, 1.0)
        self.assertGreaterEqual(vocal_ratio_vocal, 0.0)
        self.assertLessEqual(vocal_ratio_vocal, 1.0)

    def test_yoru_ni_kakeru_instrumental_restoration(self):
        """Test instrumental restoration for '夜に駆ける' specific patterns."""
        # Simulate the problematic patterns from the user's analysis
        # Pattern: B(73-88) + B(88-102) + B(102-117) where middle should be D
        yoru_pattern = [
            {
                'type': 'chorus',
                'start_time': 73.0,
                'end_time': 88.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'chorus',
                'start_time': 88.0,
                'end_time': 102.0,
                'duration': 14.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Should become instrumental
            {
                'type': 'chorus',
                'start_time': 102.0,
                'end_time': 117.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'chorus',
                'start_time': 117.0,
                'end_time': 132.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Should become instrumental
            {
                'type': 'chorus',
                'start_time': 132.0,
                'end_time': 147.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        # Create mock audio with varying characteristics
        duration = 147.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave for testing
        y = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Process with full post-processing pipeline
        processed = self.processor.post_process_sections(yoru_pattern, bpm=130.5, y=y, sr=sr)

        # Should reduce section count from original
        self.assertLessEqual(len(processed), len(yoru_pattern), "Should reduce or maintain section count")

        # Should have vocal_ratio information added
        for section in processed:
            self.assertIn('vocal_ratio', section, "Should have vocal_ratio information")

        # Should not create super-long sections (> 60s)
        for section in processed:
            self.assertLessEqual(
                section['duration'], 60.0, f"Section should not exceed 60s: {section['duration']:.1f}s"
            )

        # Should maintain reasonable structure
        self.assertGreaterEqual(len(processed), 1, "Should have at least 1 section")
        self.assertLessEqual(len(processed), 5, "Should not have too many sections")

    def test_consecutive_pre_chorus_suppression_with_lock(self):
        """Test consecutive pre-chorus suppression with lock mechanism."""
        test_sections = [
            {
                'type': 'verse',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },
            {
                'type': 'pre_chorus',
                'start_time': 15.0,
                'end_time': 30.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'pre_chorus',
                'start_time': 30.0,
                'end_time': 45.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },  # Should be downgraded
            {
                'type': 'chorus',
                'start_time': 45.0,
                'end_time': 60.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        # Test consecutive pre-chorus suppression
        processed = self.processor._suppress_consecutive_pre_chorus(test_sections.copy())

        # Second pre_chorus should be downgraded to verse and locked
        self.assertEqual(processed[2]['type'], 'verse')
        self.assertTrue(processed[2].get('_locked', False))

        # Test that locked sections are not upgraded in pairing
        paired = self.processor._enforce_pre_chorus_chorus_pairing(processed.copy())

        # Locked section should remain as verse
        self.assertEqual(paired[2]['type'], 'verse')

    def test_ascii_label_consistency(self):
        """Test ASCII label consistency after processing."""
        test_sections = [
            {
                'type': 'verse',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'verse',
                'energy_level': 0.5,
                'complexity': 0.5,
            },  # Inconsistent
            {
                'type': 'pre_chorus',
                'start_time': 15.0,
                'end_time': 30.0,
                'duration': 15.0,
                'ascii_label': 'pre_chorus',
                'energy_level': 0.6,
                'complexity': 0.6,
            },  # Inconsistent
            {
                'type': 'chorus',
                'start_time': 30.0,
                'end_time': 45.0,
                'duration': 15.0,
                'ascii_label': 'chorus',
                'energy_level': 0.8,
                'complexity': 0.7,
            },  # Inconsistent
        ]

        processed = self.processor.post_process_sections(test_sections, bpm=130.5)

        # Check that all ASCII labels are consistent with JP_ASCII_LABELS
        for section in processed:
            expected_label = self.processor.JP_ASCII_LABELS.get(section['type'], section['type'])
            actual_label = section.get('ascii_label', section['type'])
            self.assertEqual(
                actual_label,
                expected_label,
                f"Inconsistent ASCII label for {section['type']}: expected {expected_label}, got {actual_label}",
            )

    def test_processing_order_pairing_before_suppression(self):
        """Test that pairing happens before consecutive pre-chorus suppression."""
        # Create a scenario where order matters
        test_sections = [
            {
                'type': 'verse',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },
            {
                'type': 'pre_chorus',
                'start_time': 15.0,
                'end_time': 30.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'pre_chorus',
                'start_time': 30.0,
                'end_time': 45.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'verse',
                'start_time': 45.0,
                'end_time': 60.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },  # Should become chorus
        ]

        processed = self.processor.post_process_sections(test_sections, bpm=130.5)

        # Check that the processing order prevents R→R→B patterns
        consecutive_pre_chorus_count = 0
        for i in range(len(processed) - 1):
            if processed[i]['type'] == 'pre_chorus' and processed[i + 1]['type'] == 'pre_chorus':
                consecutive_pre_chorus_count += 1

        # Should have no consecutive pre-chorus sections
        self.assertEqual(consecutive_pre_chorus_count, 0)

    def test_alternating_ar_pattern_collapse(self):
        """Test A-R alternating pattern collapse to A-R-B structure."""
        test_sections = [
            {
                'type': 'verse',
                'start_time': 0.0,
                'end_time': 15.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },
            {
                'type': 'pre_chorus',
                'start_time': 15.0,
                'end_time': 30.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'verse',
                'start_time': 30.0,
                'end_time': 45.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },  # Should become chorus
        ]

        processed = self.processor._collapse_alternating_ar_patterns(test_sections.copy())

        # Third section (verse) should be converted to chorus
        self.assertEqual(processed[2]['type'], 'chorus')
        self.assertEqual(processed[2]['ascii_label'], 'Sabi')

    def test_jpop_structure_optimization(self):
        """Test complete J-Pop structure optimization (18→10 section reduction)."""
        # Simulate the "夜に駆ける" pattern: ARARRBDB...
        jpop_sections = [
            {
                'type': 'verse',
                'start_time': 0.0,
                'end_time': 14.0,
                'duration': 14.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },
            {
                'type': 'pre_chorus',
                'start_time': 14.0,
                'end_time': 29.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'verse',
                'start_time': 29.0,
                'end_time': 44.0,
                'duration': 15.0,
                'ascii_label': 'A-melo',
                'energy_level': 0.4,
                'complexity': 0.5,
            },
            {
                'type': 'pre_chorus',
                'start_time': 44.0,
                'end_time': 58.0,
                'duration': 14.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'pre_chorus',
                'start_time': 58.0,
                'end_time': 73.0,
                'duration': 15.0,
                'ascii_label': 'B-melo',
                'energy_level': 0.6,
                'complexity': 0.6,
            },
            {
                'type': 'chorus',
                'start_time': 73.0,
                'end_time': 88.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
            {
                'type': 'instrumental',
                'start_time': 88.0,
                'end_time': 102.0,
                'duration': 14.0,
                'ascii_label': 'Kansou',
                'energy_level': 0.5,
                'complexity': 0.5,
            },
            {
                'type': 'chorus',
                'start_time': 102.0,
                'end_time': 117.0,
                'duration': 15.0,
                'ascii_label': 'Sabi',
                'energy_level': 0.8,
                'complexity': 0.7,
            },
        ]

        processed = self.processor.post_process_sections(jpop_sections, bpm=130.5)

        # Should achieve some section reduction (with max chorus length limits)
        # Original: 8 sections, expect around 6 sections after processing
        self.assertLessEqual(len(processed), 7)  # Some reduction expected
        self.assertGreaterEqual(len(processed), 5)  # But not too aggressive due to length limits

        # Should not have consecutive pre-chorus
        for i in range(len(processed) - 1):
            if processed[i]['type'] == 'pre_chorus':
                self.assertNotEqual(processed[i + 1]['type'], 'pre_chorus')

        # Should have proper ASCII labels
        for section in processed:
            expected_label = self.processor.JP_ASCII_LABELS.get(section['type'], section['type'])
            self.assertEqual(section.get('ascii_label'), expected_label)


if __name__ == '__main__':
    unittest.main()
