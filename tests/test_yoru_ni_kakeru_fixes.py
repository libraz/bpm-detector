import unittest

import numpy as np

from src.bpm_detector.chord_analyzer import ChordProgressionAnalyzer
from src.bpm_detector.jpop_structure_optimizer import JPopStructureOptimizer
from src.bpm_detector.key_detector import KeyDetector


class TestYoruNiKakeruFixes(unittest.TestCase):
    def test_key_detector_unknown_format(self):
        detector = KeyDetector()
        y = np.zeros(22050)
        result = detector.detect_key(y, 22050, _feature_backend='stft')
        self.assertTrue(result['key'].startswith('Unknown('))
        self.assertIn('analysis_notes', result)

    def test_tension_chord_detection(self):
        analyzer = ChordProgressionAnalyzer()
        template = np.array(analyzer.CHORD_TEMPLATES['C7'], dtype=float)
        chord, conf = analyzer._match_chord_template(template)
        self.assertEqual(chord, 'C7')
        self.assertGreater(conf, 0.5)

    def test_label_drop_chorus(self):
        optimizer = JPopStructureOptimizer()
        sections = [
            {
                'type': 'chorus',
                'start_time': 0.0,
                'end_time': 4.0,
                'duration': 4.0,
                'ascii_label': 'Sabi',
            }
        ]
        y = np.zeros(22050 * 4)
        mods = [{'time': 1.0, 'from_key': 'C', 'to_key': 'F Major', 'confidence': 0.9}]
        result = optimizer.label_special_chorus_sections(sections, y, 22050, mods)
        self.assertEqual(result[0]['type'], 'drop_chorus')
        self.assertEqual(result[0]['ascii_label'], '落ちサビ')
        self.assertEqual(result[-1]['last_chorus_key'], 'F Major')


if __name__ == '__main__':
    unittest.main()
