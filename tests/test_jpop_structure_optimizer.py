"""Tests for J-Pop structure optimizer module."""

import unittest
import numpy as np
from src.bpm_detector.jpop_structure_optimizer import JPopStructureOptimizer


class TestJPopStructureOptimizer(unittest.TestCase):
    """Test cases for JPopStructureOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = JPopStructureOptimizer()
        self.sr = 22050
        self.bpm = 130.5

    def test_suppress_consecutive_pre_chorus(self):
        """Test consecutive pre-chorus suppression."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'pre_chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},  # Should be downgraded
            {'type': 'chorus', 'start_time': 45.0, 'end_time': 60.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        processed = self.optimizer.suppress_consecutive_pre_chorus(test_sections.copy())
        
        # Second pre_chorus should be downgraded to verse and locked
        self.assertEqual(processed[2]['type'], 'verse')
        self.assertTrue(processed[2].get('_locked', False))

    def test_enforce_pre_chorus_chorus_pairing(self):
        """Test pre-chorus to chorus pairing enforcement."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.5, 'complexity': 0.5},  # Higher energy for upgrade
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},  # Already chorus
        ]
        
        processed = self.optimizer.enforce_pre_chorus_chorus_pairing(test_sections.copy())
        
        # Check that pairing rules are enforced
        has_pre_chorus_chorus_pair = False
        for i in range(len(processed) - 1):
            if processed[i]['type'] == 'pre_chorus' and processed[i+1]['type'] == 'chorus':
                has_pre_chorus_chorus_pair = True
                break
        
        # Should upgrade verse to pre_chorus if it precedes chorus and has sufficient energy
        if processed[0]['energy_level'] > 0.4:
            expected_pairing = processed[0]['type'] == 'pre_chorus' or has_pre_chorus_chorus_pair
            self.assertTrue(expected_pairing, "Should have pre_chorus → chorus pairing or upgrade verse")
        else:
            self.assertTrue(has_pre_chorus_chorus_pair, "Should have pre_chorus → chorus pairing")

    def test_collapse_alternating_ar_patterns(self):
        """Test A-R alternating pattern collapse."""
        test_sections = [
            {'type': 'verse', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},
            {'type': 'pre_chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'B-melo', 'energy_level': 0.6, 'complexity': 0.6},
            {'type': 'verse', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'A-melo', 'energy_level': 0.4, 'complexity': 0.5},  # Should become chorus
        ]
        
        processed = self.optimizer.collapse_alternating_ar_patterns(test_sections.copy())
        
        # Third section (verse) should be converted to chorus
        self.assertEqual(processed[2]['type'], 'chorus')
        self.assertEqual(processed[2]['ascii_label'], 'Sabi')

    def test_break_consecutive_chorus_chains(self):
        """Test breaking consecutive chorus chains."""
        consecutive_chorus_pattern = [
            {'type': 'chorus', 'start_time': 73.0, 'end_time': 88.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 88.0, 'end_time': 102.0, 'duration': 14.0,
             'ascii_label': 'Sabi', 'energy_level': 0.5, 'complexity': 0.5},  # Should become instrumental
            {'type': 'chorus', 'start_time': 102.0, 'end_time': 117.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        # Create mock audio data (simple sine wave)
        duration = 117.0
        t = np.linspace(0, duration, int(self.sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        processed = self.optimizer.break_consecutive_chorus_chains(
            consecutive_chorus_pattern, y, self.sr, self.bpm
        )
        
        # Should have vocal_ratio added to all sections
        for section in processed:
            self.assertIn('vocal_ratio', section)
            self.assertIsInstance(section['vocal_ratio'], (int, float))
            self.assertGreaterEqual(section['vocal_ratio'], 0.0)
            self.assertLessEqual(section['vocal_ratio'], 1.0)
        
        # Structure should be preserved
        self.assertEqual(len(processed), 3)

    def test_process_chorus_chain_two_sections(self):
        """Test processing of two consecutive chorus sections."""
        test_sections = [
            {'type': 'chorus', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7, 'vocal_ratio': 0.8},
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.3, 'complexity': 0.4, 'vocal_ratio': 0.2},  # Low vocal
        ]
        
        self.optimizer._process_chorus_chain(test_sections, [0, 1], self.bpm)
        
        # Second section should potentially be converted to instrumental based on vocal ratio
        # (depends on energy and vocal detection)
        self.assertIn(test_sections[1]['type'], ['chorus', 'instrumental'])

    def test_process_chorus_chain_three_sections(self):
        """Test processing of three consecutive chorus sections."""
        test_sections = [
            {'type': 'chorus', 'start_time': 0.0, 'end_time': 15.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
            {'type': 'chorus', 'start_time': 15.0, 'end_time': 30.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.5, 'complexity': 0.5},  # Should become instrumental
            {'type': 'chorus', 'start_time': 30.0, 'end_time': 45.0, 'duration': 15.0,
             'ascii_label': 'Sabi', 'energy_level': 0.8, 'complexity': 0.7},
        ]
        
        self.optimizer._process_chorus_chain(test_sections, [0, 1, 2], self.bpm)
        
        # Middle section should be converted to instrumental
        self.assertEqual(test_sections[1]['type'], 'instrumental')
        self.assertEqual(test_sections[1]['ascii_label'], 'Kansou')

    def test_vocal_presence_detection(self):
        """Test vocal presence detection functionality."""
        # Create test audio signal
        duration = 10.0
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Simple sine wave (simulating instrumental)
        y_instrumental = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # More complex signal (simulating vocal)
        y_vocal = (0.3 * np.sin(2 * np.pi * 440 * t) +
                  0.2 * np.sin(2 * np.pi * 880 * t) +
                  0.1 * np.sin(2 * np.pi * 1760 * t))
        
        # Test vocal detection
        vocal_ratio_instrumental = self.optimizer._detect_vocal_presence(
            y_instrumental, self.sr, 0.0, 5.0
        )
        vocal_ratio_vocal = self.optimizer._detect_vocal_presence(
            y_vocal, self.sr, 0.0, 5.0
        )
        
        # Both should return valid ratios
        self.assertIsInstance(vocal_ratio_instrumental, (int, float))
        self.assertIsInstance(vocal_ratio_vocal, (int, float))
        self.assertGreaterEqual(vocal_ratio_instrumental, 0.0)
        self.assertLessEqual(vocal_ratio_instrumental, 1.0)
        self.assertGreaterEqual(vocal_ratio_vocal, 0.0)
        self.assertLessEqual(vocal_ratio_vocal, 1.0)


if __name__ == '__main__':
    unittest.main()