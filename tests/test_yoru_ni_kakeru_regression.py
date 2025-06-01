"""
Regression tests for "Yoru ni Kakeru" analysis issues.
Detects problems found in 4th Run and prevents future regressions.
"""

import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

from src.bpm_detector.chord_analyzer import ChordProgressionAnalyzer
from src.bpm_detector.jpop_structure_optimizer import JPopStructureOptimizer
from src.bpm_detector.key_detector import KeyDetector
from src.bpm_detector.key_profiles import _Constants
from src.bpm_detector.music_analyzer import AudioAnalyzer


class TestYoruNiKakeruRegression(unittest.TestCase):
    """Regression tests for "Yoru ni Kakeru" analysis issues."""

    def setUp(self):
        """Set up test audio data."""
        self.sr = 22050
        self.duration = 10  # 10 seconds

        # Simulate Eb major chord progression (opening of "Yoru ni Kakeru")
        t = np.linspace(0, self.duration, int(self.sr * self.duration))

        # Eb major chord (Eb-G-Bb) frequencies
        eb_freq = librosa.note_to_hz('Eb4')  # 311.13 Hz
        g_freq = librosa.note_to_hz('G4')  # 392.00 Hz
        bb_freq = librosa.note_to_hz('Bb4')  # 466.16 Hz

        # Synthesize chord
        self.eb_major_audio = (
            np.sin(2 * np.pi * eb_freq * t) + np.sin(2 * np.pi * g_freq * t) + np.sin(2 * np.pi * bb_freq * t)
        ) * 0.3

        # Add slight noise for realism
        self.eb_major_audio += np.random.normal(0, 0.05, len(self.eb_major_audio))

    def test_key_detection_not_none(self):
        """Test that key detection never returns None for valid audio."""
        detector = KeyDetector()
        result = detector.detect_key(self.eb_major_audio, self.sr)

        # Key should never be None for valid audio
        self.assertNotEqual(result['key'], 'None')
        self.assertNotEqual(result['key'], None)

        # Should detect a valid key (preferably Eb Major or related)
        self.assertIsInstance(result['key'], str)
        self.assertGreater(len(result['key']), 0)

        # Confidence should be reasonable
        self.assertGreater(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 100.0)

    def test_key_detection_eb_major_preference(self):
        """Test that Eb major audio is detected as Eb major or closely related key."""
        detector = KeyDetector()
        result = detector.detect_key(self.eb_major_audio, self.sr)

        # Should detect Eb major or closely related keys (Bb major, C minor, etc.)
        expected_keys = ['Eb', 'D#', 'Bb', 'A#', 'C', 'Cm', 'Gm', 'G']
        detected_key = result['key']

        # Check if detected key contains any expected key component
        key_found = any(expected in detected_key for expected in expected_keys)
        self.assertTrue(key_found, f"Expected Eb-related key, got {detected_key}")

    def test_music_analyzer_key_integration(self):
        """Test that AudioAnalyzer properly integrates key detection."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, self.eb_major_audio, self.sr)

            analyzer = AudioAnalyzer()
            result = analyzer.analyze_file(tmp_file.name, detect_key=True, comprehensive=False)

            # Basic info should contain key information
            self.assertIn('basic_info', result)
            basic_info = result['basic_info']

            # Key should not be None
            key = basic_info.get('key')
            self.assertIsNotNone(key)
            self.assertNotEqual(key, 'None')
            self.assertNotEqual(key, '')

            # Key confidence should be present and reasonable
            key_conf = basic_info.get('key_confidence', 0.0)
            self.assertGreater(key_conf, 0.0)

    def test_chord_progression_not_oversimplified(self):
        """Test that chord progression analysis doesn't oversimplify to 2 chords."""
        analyzer = ChordProgressionAnalyzer()

        # Create a 4-chord progression: Eb - Bb - Cm - Ab
        duration_per_chord = 2  # 2 seconds per chord
        t_chord = np.linspace(0, duration_per_chord, int(self.sr * duration_per_chord))

        # Eb major
        eb_chord = (
            np.sin(2 * np.pi * librosa.note_to_hz('Eb4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('G4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('Bb4') * t_chord)
        ) * 0.3

        # Bb major
        bb_chord = (
            np.sin(2 * np.pi * librosa.note_to_hz('Bb3') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('D4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('F4') * t_chord)
        ) * 0.3

        # C minor
        cm_chord = (
            np.sin(2 * np.pi * librosa.note_to_hz('C4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('Eb4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('G4') * t_chord)
        ) * 0.3

        # Ab major
        ab_chord = (
            np.sin(2 * np.pi * librosa.note_to_hz('Ab3') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('C4') * t_chord)
            + np.sin(2 * np.pi * librosa.note_to_hz('Eb4') * t_chord)
        ) * 0.3

        # Concatenate progression
        progression_audio = np.concatenate([eb_chord, bb_chord, cm_chord, ab_chord])

        # Analyze chord progression
        analysis = analyzer.analyze(progression_audio, self.sr, key="Eb Major", bpm=130.0)

        # Should detect more than 2 unique chords
        unique_chords = analysis.get('unique_chords', 0)
        self.assertGreaterEqual(unique_chords, 3, "Chord progression should not be oversimplified to 2 chords")

        # Main progression should have reasonable length
        main_progression = analysis.get('main_progression', [])
        self.assertGreaterEqual(len(main_progression), 3, "Main progression should contain at least 3 chords")

    def test_triad_tetrad_balance(self):
        """Test that triad/tetrad detection is balanced (not all 7th chords)."""
        analyzer = ChordProgressionAnalyzer()

        # Test various chord types
        test_cases = [
            # (chord_notes, expected_type_preference)
            (['C4', 'E4', 'G4'], 'triad'),  # C major triad
            (['C4', 'E4', 'G4', 'Bb4'], 'tetrad'),  # C7 with strong 7th
            (['F4', 'A4', 'C5'], 'triad'),  # F major triad
            (['G4', 'B4', 'D5'], 'triad'),  # G major triad
        ]

        triad_count = 0
        tetrad_count = 0

        for chord_notes, expected_type in test_cases:
            # Create chord audio
            t = np.linspace(0, 2, int(self.sr * 2))
            chord_audio = np.zeros_like(t)

            for note in chord_notes:
                freq = librosa.note_to_hz(note)
                chord_audio += np.sin(2 * np.pi * freq * t) * 0.3

            # Create chroma template
            chroma = np.zeros(12)
            for note in chord_notes:
                note_class = librosa.note_to_hz(note)
                chroma_idx = int(np.round(12 * np.log2(note_class / librosa.note_to_hz('C4')))) % 12
                chroma[chroma_idx] = 1.0

            # Detect chord
            chord_name, confidence = analyzer._match_chord_template(chroma)

            # Count triad vs tetrad detection
            if '7' in chord_name or 'sus' in chord_name:
                tetrad_count += 1
            else:
                triad_count += 1

        # Should not detect all chords as 7th chords (addressing 4th Run issue)
        self.assertGreater(triad_count, 0, "Should detect some triads, not all 7th chords")

    def test_drop_chorus_detection_conditions(self):
        """Test that drop chorus detection works with proper conditions."""
        optimizer = JPopStructureOptimizer()

        # Test case 1: Low RMS + Key change (should detect drop chorus)
        sections_1 = [{'type': 'chorus', 'start_time': 0.0, 'end_time': 4.0, 'duration': 4.0, 'ascii_label': 'Sabi'}]

        # Create low-energy audio
        t = np.linspace(0, 4, self.sr * 4)
        low_energy_audio = np.sin(2 * np.pi * 440 * t) * 0.01  # Very low amplitude

        modulations = [{'time': 1.0, 'from_key': 'Eb Major', 'to_key': 'D Major', 'confidence': 0.9}]

        result = optimizer.label_special_chorus_sections(sections_1, low_energy_audio, self.sr, modulations)

        # Should detect as drop chorus
        self.assertEqual(result[0]['type'], 'drop_chorus')
        self.assertEqual(result[0]['ascii_label'], 'OchiSabi')

        # Test case 2: Normal RMS + No key change (should remain chorus)
        sections_2 = [{'type': 'chorus', 'start_time': 0.0, 'end_time': 4.0, 'duration': 4.0, 'ascii_label': 'Sabi'}]
        normal_energy_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # Normal amplitude
        no_modulations = []

        result_normal = optimizer.label_special_chorus_sections(
            sections_2, normal_energy_audio, self.sr, no_modulations
        )

        # Should remain as chorus
        self.assertEqual(result_normal[0]['type'], 'chorus')

    def test_confidence_thresholds_reasonable(self):
        """Test that confidence thresholds are set to reasonable values."""
        # Check that MIN_CONFIDENCE is not too high (causing None results)
        self.assertLessEqual(
            _Constants.MIN_CONFIDENCE, 0.2, "MIN_CONFIDENCE should not be too high to avoid None results"
        )

        # Check that MIN_CONFIDENCE is not too low (causing false positives)
        self.assertGreaterEqual(
            _Constants.MIN_CONFIDENCE, 0.05, "MIN_CONFIDENCE should not be too low to avoid false positives"
        )

        # Check other thresholds are reasonable
        self.assertLessEqual(_Constants.JPOP_CONF_THRESH, 0.5)
        self.assertLessEqual(_Constants.CHORD_WEIGHT_THRESH, 0.8)

    def test_modulation_detection_functionality(self):
        """Test that modulation detection can identify key changes."""
        detector = KeyDetector()

        # Create audio with key change: Eb major -> D major
        duration_per_key = 5
        t = np.linspace(0, duration_per_key, int(self.sr * duration_per_key))

        # Eb major section
        eb_section = (
            np.sin(2 * np.pi * librosa.note_to_hz('Eb4') * t)
            + np.sin(2 * np.pi * librosa.note_to_hz('G4') * t)
            + np.sin(2 * np.pi * librosa.note_to_hz('Bb4') * t)
        ) * 0.3

        # D major section
        d_section = (
            np.sin(2 * np.pi * librosa.note_to_hz('D4') * t)
            + np.sin(2 * np.pi * librosa.note_to_hz('F#4') * t)
            + np.sin(2 * np.pi * librosa.note_to_hz('A4') * t)
        ) * 0.3

        # Combine sections
        modulation_audio = np.concatenate([eb_section, d_section])

        # Test modulation detection
        modulation_data = detector.compute_modulation_timeseries(modulation_audio, self.sr, bpm=130)

        # Should detect some modulations
        modulations = modulation_data.get('modulations', [])
        self.assertGreater(len(modulations), 0, "Should detect modulations in key-changing audio")

        # Should have reasonable number of time points
        times = modulation_data.get('times', [])
        self.assertGreater(len(times), 0, "Should have time series data")

    def test_integration_consistency_checks(self):
        """Test integration consistency to prevent class mismatch issues."""
        # Test that AudioAnalyzer uses the correct KeyDetector class
        analyzer = AudioAnalyzer()

        # Check that the key_detector is the enhanced version
        self.assertTrue(
            hasattr(analyzer.key_detector, 'detect_key'),
            "AudioAnalyzer should use enhanced KeyDetector with detect_key method",
        )

        # Check that it's the correct class from key_detector.py
        from src.bpm_detector.key_detector import KeyDetector as EnhancedKeyDetector

        self.assertIsInstance(
            analyzer.key_detector,
            EnhancedKeyDetector,
            "AudioAnalyzer should use EnhancedKeyDetector, not legacy KeyDetector",
        )

        # Check that the class has the expected enhanced methods
        self.assertTrue(
            hasattr(analyzer.key_detector, 'compute_modulation_timeseries'),
            "KeyDetector should have modulation detection capability",
        )

        # Test that imports are consistent
        from src.bpm_detector import KeyDetector as ImportedKeyDetector

        self.assertEqual(
            ImportedKeyDetector, EnhancedKeyDetector, "Imported KeyDetector should be the enhanced version"
        )

    def test_audio_analyzer_key_detection_integration(self):
        """Test that AudioAnalyzer properly integrates with enhanced key detection."""
        analyzer = AudioAnalyzer()

        # Test direct key detection through AudioAnalyzer
        result = analyzer.key_detector.detect_key(self.eb_major_audio, self.sr)

        # Should return enhanced KeyDetectionResult format
        self.assertIsInstance(result, dict)
        self.assertIn('key', result)
        self.assertIn('mode', result)
        self.assertIn('confidence', result)
        self.assertIn('key_strength', result)
        self.assertIn('analysis_notes', result)

        # Key should not be None (regression prevention)
        self.assertIsNotNone(result['key'])
        self.assertNotEqual(result['key'], 'None')

        # Should have reasonable confidence range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 100)

    def test_no_legacy_keydetector_usage(self):
        """Ensure no legacy KeyDetector classes are being used."""
        # Test that music_analyzer doesn't define its own KeyDetector
        import src.bpm_detector.music_analyzer as ma

        # Should not have KeyDetector class defined in music_analyzer
        # (it should only import from key_detector.py)
        module_classes = [name for name in dir(ma) if isinstance(getattr(ma, name), type)]
        keydetector_classes = [cls for cls in module_classes if 'KeyDetector' in cls]

        # Check if any KeyDetector classes are defined (not imported) in music_analyzer
        defined_classes = []
        for cls_name in keydetector_classes:
            cls_obj = getattr(ma, cls_name)
            # Check if it's defined in this module (not imported)
            if cls_obj.__module__ == 'src.bpm_detector.music_analyzer':
                defined_classes.append(cls_name)

        # Should not define KeyDetector classes in music_analyzer module
        self.assertEqual(
            len(defined_classes), 0, f"music_analyzer should not define KeyDetector classes, found: {defined_classes}"
        )

        # Verify AudioAnalyzer uses the correct KeyDetector
        analyzer = AudioAnalyzer()
        key_detector_module = analyzer.key_detector.__class__.__module__
        self.assertEqual(
            key_detector_module,
            'src.bpm_detector.key_detector',
            f"AudioAnalyzer should use KeyDetector from key_detector.py, not {key_detector_module}",
        )

    def test_keydetector_method_consistency(self):
        """Test that KeyDetector methods work consistently across different usage patterns."""
        detector = KeyDetector()

        # Test both detect_key and detect methods exist and work
        self.assertTrue(hasattr(detector, 'detect_key'))
        self.assertTrue(hasattr(detector, 'detect'))

        # Test that both methods return reasonable results
        result_enhanced = detector.detect_key(self.eb_major_audio, self.sr)
        result_legacy = detector.detect(self.eb_major_audio, self.sr)

        # Enhanced method should return dict
        self.assertIsInstance(result_enhanced, dict)

        # Legacy method should return tuple
        self.assertIsInstance(result_legacy, tuple)
        self.assertEqual(len(result_legacy), 2)

        # Both should detect valid keys (not None)
        self.assertNotEqual(result_enhanced['key'], 'None')
        self.assertNotEqual(result_legacy[0], 'None')

    def test_parallel_analyzer_key_integration(self):
        """Test that SmartParallelAudioAnalyzer also uses correct KeyDetector."""
        from src.bpm_detector.parallel_analyzer import SmartParallelAudioAnalyzer

        # Test that SmartParallelAudioAnalyzer uses the correct KeyDetector
        parallel_analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)  # Disable parallel for testing

        # Check that it uses the enhanced KeyDetector
        from src.bpm_detector.key_detector import KeyDetector as EnhancedKeyDetector

        self.assertIsInstance(
            parallel_analyzer.key_detector,
            EnhancedKeyDetector,
            "SmartParallelAudioAnalyzer should use EnhancedKeyDetector",
        )

        # Check that it has the enhanced methods
        self.assertTrue(
            hasattr(parallel_analyzer.key_detector, 'detect_key'),
            "SmartParallelAudioAnalyzer KeyDetector should have detect_key method",
        )

        # Test that the key detection works correctly
        result = parallel_analyzer.key_detector.detect_key(self.eb_major_audio, self.sr)

        # Should return enhanced KeyDetectionResult format
        self.assertIsInstance(result, dict)
        self.assertIn('key', result)
        self.assertIn('mode', result)
        self.assertIn('confidence', result)

        # Key should not be None (regression prevention)
        self.assertIsNotNone(result['key'])
        self.assertNotEqual(result['key'], 'None')


if __name__ == '__main__':
    unittest.main()
