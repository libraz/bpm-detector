"""Test-driven development for '夜に駆ける' specific fixes."""

import unittest
import numpy as np
from src.bpm_detector.section_processor import SectionProcessor


class TestYoruNiKakeruFixes(unittest.TestCase):
    """Test cases for fixing specific issues in '夜に駆ける' analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SectionProcessor(hop_length=512)
        self.sr = 22050
        self.bpm = 130.5

    def test_consecutive_chorus_limit_max_2(self):
        """Test that consecutive chorus sections are limited to maximum 2."""
        # FAILING TEST: Current implementation allows 3+ consecutive chorus
        four_chorus_pattern = [
            {'type': 'chorus', 'start_time': 73.0, 'end_time': 88.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 88.0, 'end_time': 102.0, 'duration': 14.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.3, 'complexity': 0.4},  # Should become instrumental
            {'type': 'chorus', 'start_time': 102.0, 'end_time': 117.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 117.0, 'end_time': 132.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.3, 'complexity': 0.4},  # Should become instrumental
        ]
        
        # Create mock audio
        duration = 132.0
        t = np.linspace(0, duration, int(self.sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        processed = self.processor.post_process_sections(four_chorus_pattern, bpm=self.bpm, y=y, sr=self.sr)
        
        # Count maximum consecutive chorus sections
        max_consecutive = 0
        current_consecutive = 0
        for section in processed:
            if section['type'] == 'chorus':
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # THIS SHOULD FAIL with current implementation
        self.assertLessEqual(max_consecutive, 2, 
                           f"Should not have more than 2 consecutive chorus sections, found {max_consecutive}")

    def test_instrumental_restoration_from_low_vocal_chorus(self):
        """Test that low vocal ratio chorus sections are converted to instrumental."""
        # FAILING TEST: Current implementation doesn't properly restore instrumentals
        bbb_pattern = [
            {'type': 'chorus', 'start_time': 73.0, 'end_time': 88.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 88.0, 'end_time': 102.0, 'duration': 14.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.3, 'complexity': 0.4},  # Should become instrumental
            {'type': 'chorus', 'start_time': 102.0, 'end_time': 117.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        # Create mock audio with low vocal content in middle section
        duration = 117.0
        t = np.linspace(0, duration, int(self.sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        processed = self.processor.post_process_sections(bbb_pattern, bpm=self.bpm, y=y, sr=self.sr)
        
        # Should have at least one instrumental section
        instrumental_count = sum(1 for s in processed if s['type'] == 'instrumental')
        
        # THIS SHOULD FAIL with current implementation
        self.assertGreater(instrumental_count, 0, 
                         "Should restore at least one instrumental section from B-B-B pattern")

    def test_yoru_ni_kakeru_target_structure(self):
        """Test that we can achieve the target structure: ARAR B D B D B R A B R B D O."""
        # FAILING TEST: Current implementation produces ARBRBBBRABRBB (56% accuracy)
        # Target: ARAR B D B D B R A B R B D O (90%+ accuracy)
        
        current_parser_output = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 14.0, 'duration': 14.0, 
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 14.0, 'end_time': 29.0, 'duration': 15.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 29.0, 'end_time': 44.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'pre_chorus', 'start_time': 44.0, 'end_time': 58.0, 'duration': 14.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.5, 'complexity': 0.5},  # Should be verse
            {'type': 'chorus', 'start_time': 58.0, 'end_time': 73.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.6, 'complexity': 0.6},  # Should be pre_chorus
            {'type': 'chorus', 'start_time': 73.0, 'end_time': 88.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 88.0, 'end_time': 102.0, 'duration': 14.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.3, 'complexity': 0.4},  # Should be instrumental
            {'type': 'chorus', 'start_time': 102.0, 'end_time': 117.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'pre_chorus', 'start_time': 117.0, 'end_time': 132.0, 'duration': 15.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.3, 'complexity': 0.4},  # Should be instrumental
            {'type': 'verse', 'start_time': 132.0, 'end_time': 147.0, 'duration': 15.0, 
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},  # Should be chorus
            {'type': 'chorus', 'start_time': 147.0, 'end_time': 161.0, 'duration': 14.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.6, 'complexity': 0.6},  # Should be pre_chorus
            {'type': 'verse', 'start_time': 161.0, 'end_time': 176.0, 'duration': 15.0, 
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'chorus', 'start_time': 176.0, 'end_time': 191.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'pre_chorus', 'start_time': 191.0, 'end_time': 205.0, 'duration': 14.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 205.0, 'end_time': 220.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            # Missing instrumental section 220-234
            {'type': 'chorus', 'start_time': 234.0, 'end_time': 249.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 249.0, 'end_time': 261.0, 'duration': 12.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.2, 'complexity': 0.3},  # Should be outro
        ]
        
        # Create mock audio
        duration = 261.0
        t = np.linspace(0, duration, int(self.sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Add fade out at the end
        fade_start = int(249.0 * self.sr)
        y[fade_start:] *= np.linspace(1.0, 0.1, len(y) - fade_start)
        
        processed = self.processor.post_process_sections(current_parser_output, bpm=self.bpm, y=y, sr=self.sr)
        
        # Count section types
        section_counts = {}
        for section in processed:
            section_type = section['type']
            section_counts[section_type] = section_counts.get(section_type, 0) + 1
        
        # Should have reasonable distribution
        # Target structure should have: A, R, B, D, O sections
        expected_types = {'verse', 'pre_chorus', 'chorus', 'instrumental'}
        found_types = set(section_counts.keys())
        
        # THIS SHOULD FAIL with current implementation
        self.assertTrue(expected_types.issubset(found_types), 
                       f"Should have all expected section types. Found: {found_types}, Expected: {expected_types}")
        
        # Should have at least 2 instrumental sections
        instrumental_count = section_counts.get('instrumental', 0)
        self.assertGreaterEqual(instrumental_count, 2, 
                              f"Should have at least 2 instrumental sections, found {instrumental_count}")
        
        # Should have outro detection
        outro_count = section_counts.get('outro', 0)
        self.assertGreaterEqual(outro_count, 1, 
                              f"Should detect outro section, found {outro_count}")

    def test_r_consecutive_suppression_with_lock(self):
        """Test that consecutive R sections are properly suppressed and locked."""
        # FAILING TEST: R-R patterns should be converted to R-A with lock
        rr_pattern = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0, 
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'pre_chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0, 
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},  # Should become verse
            {'type': 'chorus', 'start_time': 45.0, 'end_time': 60.0, 'duration': 15.0, 
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(rr_pattern, bpm=self.bpm)
        
        # Should not have consecutive pre_chorus sections
        consecutive_r_found = False
        for i in range(len(processed) - 1):
            if processed[i]['type'] == 'pre_chorus' and processed[i+1]['type'] == 'pre_chorus':
                consecutive_r_found = True
                break
        
        # THIS SHOULD FAIL with current implementation
        self.assertFalse(consecutive_r_found, "Should not have consecutive pre_chorus sections")

    def test_instrumental_alias_normalization(self):
        """Test normalization of instrumental aliases (solo, interlude, etc.) to 'instrumental'."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'solo', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},  # Should become instrumental
            {'type': 'interlude', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'Interlude', 'energy_level': 0.5, 'complexity': 0.6},  # Should become instrumental
            {'type': 'buildup', 'start_time': 45.0, 'end_time': 60.0, 'duration': 15.0,
             'ascii_label': 'Buildup', 'energy_level': 0.8, 'complexity': 0.7},  # Should become instrumental
            {'type': 'breakdown', 'start_time': 60.0, 'end_time': 75.0, 'duration': 15.0,
             'ascii_label': 'Breakdown', 'energy_level': 0.3, 'complexity': 0.4},  # Should become instrumental
            {'type': 'chorus', 'start_time': 75.0, 'end_time': 90.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Check that all instrumental aliases were normalized
        instrumental_count = 0
        for section in processed:
            if section['type'] == 'instrumental':
                instrumental_count += 1
                self.assertEqual(section['ascii_label'], 'Kansou',
                               "Instrumental sections should have 'Kansou' ASCII label")
        
        # Should have converted 4 alias sections to instrumental
        self.assertGreaterEqual(instrumental_count, 3,
                               "Should normalize instrumental aliases to 'instrumental'")
        
        # No sections should have the original alias types
        for section in processed:
            self.assertNotIn(section['type'], ['solo', 'interlude', 'buildup', 'breakdown'],
                           f"Section type '{section['type']}' should be normalized to 'instrumental'")

    def test_ending_verse_to_outro_conversion(self):
        """Test conversion of ending short verse sections to outro."""
        # Create sections with short verse at the end
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'verse', 'start_time': 30.0, 'end_time': 38.0, 'duration': 8.0,  # Short ending verse
             'ascii_label': 'A-melo', 'energy_level': 0.3, 'complexity': 0.4},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Last section should be converted to outro if it's short enough
        if processed:
            last_section = processed[-1]
            # Calculate 8-bar duration at 130.5 BPM
            eight_bar_duration = (8 * 4 * 60.0) / 130.5  # ≈ 14.7s
            
            if last_section['duration'] <= eight_bar_duration:
                self.assertEqual(last_section['type'], 'outro',
                               "Short ending verse should be converted to outro")
                self.assertEqual(last_section['ascii_label'], 'Outro',
                               "Outro section should have 'Outro' ASCII label")

    def test_ending_instrumental_to_outro_conversion(self):
        """Test conversion of ending instrumental sections to outro."""
        # Create sections with instrumental at the end
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'instrumental', 'start_time': 30.0, 'end_time': 42.0, 'duration': 12.0,  # Short ending instrumental
             'ascii_label': 'Kansou', 'energy_level': 0.4, 'complexity': 0.5},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Last section should be converted to outro if it's short enough
        if processed:
            last_section = processed[-1]
            # Calculate 8-bar duration at 130.5 BPM
            eight_bar_duration = (8 * 4 * 60.0) / 130.5  # ≈ 14.7s
            
            if last_section['duration'] <= eight_bar_duration:
                self.assertEqual(last_section['type'], 'outro',
                               "Short ending instrumental should be converted to outro")
                self.assertEqual(last_section['ascii_label'], 'Outro',
                               "Outro section should have 'Outro' ASCII label")

    def test_oversized_pre_chorus_splitting(self):
        """Test that oversized pre_chorus sections (>16 bars) are split into verse + pre_chorus."""
        # Create a very long pre_chorus section (35 bars ≈ 64s at 130.5 BPM)
        long_pre_chorus_duration = (35 * 4 * 60.0) / 130.5  # ≈ 64s
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 15.0 + long_pre_chorus_duration,
             'duration': long_pre_chorus_duration,  # Very long pre_chorus (35 bars)
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 15.0 + long_pre_chorus_duration, 'end_time': 30.0 + long_pre_chorus_duration,
             'duration': 15.0, 'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Should have split the long pre_chorus into verse + pre_chorus
        verse_count = sum(1 for s in processed if s['type'] == 'verse')
        pre_chorus_count = sum(1 for s in processed if s['type'] == 'pre_chorus')
        
        # Should have at least 2 verse sections (original + split from long pre_chorus)
        self.assertGreaterEqual(verse_count, 2,
                               "Long pre_chorus should be split, creating additional verse section")
        
        # Should still have pre_chorus sections
        self.assertGreaterEqual(pre_chorus_count, 1,
                               "Should still have pre_chorus sections after splitting")
        
        # Check that no single pre_chorus is longer than 16 bars
        max_r_duration = (16 * 4 * 60.0) / 130.5  # 16 bars at 130.5 BPM
        for section in processed:
            if section['type'] == 'pre_chorus':
                self.assertLessEqual(section['duration'], max_r_duration + 1.0,  # Allow 1s tolerance
                                   f"Pre_chorus section should not exceed 16 bars, found {section['duration']:.1f}s")

    def test_multi_scale_boundary_detection(self):
        """Test that multi-scale novelty detection produces more boundaries."""
        from src.bpm_detector.boundary_detector import BoundaryDetector
        
        # Create a mock similarity matrix with clear boundaries
        n_frames = 1000
        similarity_matrix = np.eye(n_frames) * 0.8
        
        # Add some structure patterns
        for i in range(0, n_frames, 100):  # Every 100 frames
            end = min(i + 50, n_frames)
            similarity_matrix[i:end, i:end] = 0.9
        
        detector = BoundaryDetector(hop_length=512)
        sr = 22050
        bpm = 130.5
        
        # Test with new multi-scale detection
        boundaries = detector.detect_boundaries(similarity_matrix, sr, min_segment_length=4.0, bpm=bpm)
        
        # Should detect more boundaries with multi-scale approach
        self.assertGreater(len(boundaries), 3,
                          "Multi-scale novelty should detect more boundaries")
        
        # Boundaries should be snapped to 4-beat grid
        bar_frames = int((4 * 60 / bpm) * sr / detector.hop_length)
        for boundary in boundaries[1:-1]:  # Skip start and end
            self.assertEqual(boundary % bar_frames, 0,
                           f"Boundary {boundary} should be snapped to 4-beat grid")

    def test_instrumental_count_target(self):
        """Test that we achieve the target of at least 2 instrumental sections."""
        # Simulate a more realistic pattern with separated instrumental sections
        current_pattern = [
            {'type': 'intro', 'start_time': 0.0, 'end_time': 14.0, 'duration': 14.0,
             'ascii_label': 'Intro', 'energy_level': 0.3, 'complexity': 0.4},
            {'type': 'pre_chorus', 'start_time': 14.0, 'end_time': 49.0, 'duration': 35.0,  # Long R (35 bars)
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 49.0, 'end_time': 64.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'solo', 'start_time': 64.0, 'end_time': 79.0, 'duration': 15.0,  # First instrumental
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
            {'type': 'pre_chorus', 'start_time': 79.0, 'end_time': 136.0, 'duration': 57.0,  # Long R (57 bars)
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 136.0, 'end_time': 151.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'interlude', 'start_time': 151.0, 'end_time': 166.0, 'duration': 15.0,  # Second instrumental
             'ascii_label': 'Interlude', 'energy_level': 0.5, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 166.0, 'end_time': 181.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(current_pattern, bpm=130.5)
        
        # Count instrumental sections (including normalized aliases)
        instrumental_count = sum(1 for s in processed if s['type'] == 'instrumental')
        
        # Should have at least 2 instrumental sections
        self.assertGreaterEqual(instrumental_count, 2,
                               f"Should have at least 2 instrumental sections, found {instrumental_count}")
        
        # Should not have more than 2 consecutive chorus sections
        max_consecutive_chorus = 0
        current_consecutive = 0
        for section in processed:
            if section['type'] == 'chorus':
                current_consecutive += 1
                max_consecutive_chorus = max(max_consecutive_chorus, current_consecutive)
            else:
                current_consecutive = 0
        
        self.assertLessEqual(max_consecutive_chorus, 2,
                           f"Should not have more than 2 consecutive chorus sections, found {max_consecutive_chorus}")


    def test_oversize_verse_pre_chorus_splitting(self):
        """Test that oversized verse and pre_chorus sections (>16 bars) are split into verse + pre_chorus."""
        # Create very long verse and pre_chorus sections
        long_duration = (20 * 4 * 60.0) / 130.5  # 20 bars ≈ 36.8s at 130.5 BPM
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': long_duration, 'duration': long_duration,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': long_duration, 'end_time': 2 * long_duration,
             'duration': long_duration, 'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 2 * long_duration, 'end_time': 2 * long_duration + 15.0,
             'duration': 15.0, 'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Should have more sections due to splitting
        self.assertGreater(len(processed), len(test_sections),
                          "Should have more sections after splitting oversized ones")
        
        # Check that no single verse or pre_chorus is longer than 16 bars
        max_duration = (16 * 4 * 60.0) / 130.5  # 16 bars at 130.5 BPM
        for section in processed:
            if section['type'] in ['verse', 'pre_chorus']:
                self.assertLessEqual(section['duration'], max_duration + 1.0,  # Allow 1s tolerance
                                   f"{section['type']} section should not exceed 16 bars, found {section['duration']:.1f}s")

    def test_short_solo_consolidation(self):
        """Test that short solo sections are consolidated into instrumental."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'solo', 'start_time': 15.0, 'end_time': 20.0, 'duration': 5.0,  # Short solo
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
            {'type': 'solo', 'start_time': 20.0, 'end_time': 25.0, 'duration': 5.0,  # Another short solo
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
            {'type': 'chorus', 'start_time': 25.0, 'end_time': 40.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Should have consolidated short solos into one instrumental
        instrumental_sections = [s for s in processed if s['type'] == 'instrumental']
        self.assertGreaterEqual(len(instrumental_sections), 1,
                               "Should have at least one instrumental section from consolidated solos")
        
        # Should not have any solo sections left
        solo_sections = [s for s in processed if s['type'] == 'solo']
        self.assertEqual(len(solo_sections), 0,
                        "Should not have any solo sections after consolidation")

    def test_short_chorus_handling(self):
        """Test that short chorus sections are handled appropriately."""
        # Create a short chorus (< 8 bars)
        short_chorus_duration = (6 * 4 * 60.0) / 130.5  # 6 bars ≈ 11.0s at 130.5 BPM
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 30.0, 'end_time': 30.0 + short_chorus_duration,
             'duration': short_chorus_duration, 'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'verse', 'start_time': 30.0 + short_chorus_duration, 'end_time': 45.0 + short_chorus_duration,
             'duration': 15.0, 'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Check that short chorus was either absorbed into pre_chorus or converted to instrumental
        short_chorus_found = False
        for section in processed:
            if section['type'] == 'chorus':
                min_chorus_duration = (8 * 4 * 60.0) / 130.5  # 8 bars
                if section['duration'] < min_chorus_duration:
                    short_chorus_found = True
        
        self.assertFalse(short_chorus_found,
                        "Short chorus sections should be absorbed or converted")

    def test_improved_boundary_detection_granularity(self):
        """Test that improved boundary detection allows finer granularity."""
        from src.bpm_detector.boundary_detector import BoundaryDetector
        
        # Create a mock similarity matrix with fine-grained boundaries
        n_frames = 500
        similarity_matrix = np.eye(n_frames) * 0.8
        
        # Add structure patterns every 50 frames (finer than before)
        for i in range(0, n_frames, 50):
            end = min(i + 25, n_frames)
            similarity_matrix[i:end, i:end] = 0.9
        
        detector = BoundaryDetector(hop_length=512)
        sr = 22050
        bpm = 130.5
        
        # Test with new finer detection (min_segment_length=2.0)
        boundaries = detector.detect_boundaries(similarity_matrix, sr, min_segment_length=2.0, bpm=bpm)
        
        # Should detect more boundaries with finer granularity
        self.assertGreater(len(boundaries), 2,
                          "Finer boundary detection should find more boundaries")


    def test_advanced_verse_pre_chorus_correction(self):
        """Test advanced verse/pre_chorus boundary correction based on energy levels."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.6, 'complexity': 0.5},  # High energy verse
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'pre_chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'pre_chorus', 'start_time': 45.0, 'end_time': 60.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},  # Consecutive pre_chorus
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # High energy verse before chorus should become pre_chorus
        first_section = processed[0]
        if first_section['energy_level'] > 0.55:
            self.assertEqual(first_section['type'], 'pre_chorus',
                           "High energy verse before chorus should become pre_chorus")
        
        # Should not have consecutive pre_chorus sections
        consecutive_pre_chorus_found = False
        for i in range(len(processed) - 1):
            if processed[i]['type'] == 'pre_chorus' and processed[i+1]['type'] == 'pre_chorus':
                consecutive_pre_chorus_found = True
                break
        
        self.assertFalse(consecutive_pre_chorus_found,
                        "Should not have consecutive pre_chorus sections after correction")

    def test_strict_chorus_chain_limitation(self):
        """Test strict limitation of consecutive chorus sections to maximum 2."""
        # Create a pattern with 4 consecutive chorus sections
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 45.0, 'end_time': 60.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.7, 'complexity': 0.6},  # Should become instrumental
            {'type': 'chorus', 'start_time': 60.0, 'end_time': 75.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.7, 'complexity': 0.6},  # Should become instrumental
            {'type': 'verse', 'start_time': 75.0, 'end_time': 90.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
        ]
        
        processed = self.processor.post_process_sections(test_sections, bpm=130.5)
        
        # Count maximum consecutive chorus sections
        max_consecutive_chorus = 0
        current_consecutive = 0
        for section in processed:
            if section['type'] == 'chorus':
                current_consecutive += 1
                max_consecutive_chorus = max(max_consecutive_chorus, current_consecutive)
            else:
                current_consecutive = 0
        
        self.assertLessEqual(max_consecutive_chorus, 2,
                           f"Should not have more than 2 consecutive chorus sections, found {max_consecutive_chorus}")
        
        # Should have at least one instrumental section from converted chorus
        instrumental_count = sum(1 for s in processed if s['type'] == 'instrumental')
        self.assertGreaterEqual(instrumental_count, 1,
                               "Should have at least one instrumental section from converted excess chorus")

    def test_realistic_yoru_ni_kakeru_pattern(self):
        """Test with a more realistic pattern based on actual analysis results."""
        # Simulate the problematic pattern: IARRSSARS → target: ARARABDBDBRABRBDO
        realistic_pattern = [
            {'type': 'intro', 'start_time': 0.0, 'end_time': 8.0, 'duration': 8.0,
             'ascii_label': 'Intro', 'energy_level': 0.3, 'complexity': 0.4},
            {'type': 'verse', 'start_time': 8.0, 'end_time': 38.0, 'duration': 30.0,  # Long verse (should split)
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 38.0, 'end_time': 53.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'pre_chorus', 'start_time': 53.0, 'end_time': 68.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'solo', 'start_time': 68.0, 'end_time': 73.0, 'duration': 5.0,  # Short solo
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
            {'type': 'solo', 'start_time': 73.0, 'end_time': 78.0, 'duration': 5.0,  # Short solo
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
            {'type': 'verse', 'start_time': 78.0, 'end_time': 93.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.6, 'complexity': 0.5},  # High energy verse
            {'type': 'pre_chorus', 'start_time': 93.0, 'end_time': 108.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'solo', 'start_time': 108.0, 'end_time': 123.0, 'duration': 15.0,
             'ascii_label': 'Solo', 'energy_level': 0.7, 'complexity': 0.8},
        ]
        
        processed = self.processor.post_process_sections(realistic_pattern, bpm=130.5)
        
        # Should have improved structure
        section_types = [s['type'] for s in processed]
        
        # Should have both verse and pre_chorus sections
        self.assertIn('verse', section_types, "Should have verse sections")
        self.assertIn('pre_chorus', section_types, "Should have pre_chorus sections")
        
        # Should have instrumental sections from solo consolidation
        instrumental_count = sum(1 for s in processed if s['type'] == 'instrumental')
        self.assertGreaterEqual(instrumental_count, 1,
                               "Should have instrumental sections from solo consolidation")
        
        # Should not have excessive consecutive sections of same type
        max_consecutive = {}
        current_type = None
        current_count = 0
        
        for section in processed:
            if section['type'] == current_type:
                current_count += 1
            else:
                if current_type:
                    max_consecutive[current_type] = max(max_consecutive.get(current_type, 0), current_count)
                current_type = section['type']
                current_count = 1
        
        # Final section
        if current_type:
            max_consecutive[current_type] = max(max_consecutive.get(current_type, 0), current_count)
        
        # Check limits
        self.assertLessEqual(max_consecutive.get('chorus', 0), 2,
                           "Should not have more than 2 consecutive chorus sections")
        self.assertLessEqual(max_consecutive.get('pre_chorus', 0), 2,
                           "Should not have more than 2 consecutive pre_chorus sections")


if __name__ == '__main__':
    unittest.main()