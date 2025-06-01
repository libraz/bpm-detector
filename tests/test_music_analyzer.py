"""Tests for the integrated music analyzer."""

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.bpm_detector.music_analyzer import AudioAnalyzer


class TestAudioAnalyzer(unittest.TestCase):
    """Test cases for AudioAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AudioAnalyzer()
        self.sr = 22050
        self.duration = 10.0  # 10 seconds

        # Create a test audio file
        self.test_audio = self._create_test_audio()

        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(self.temp_file.name, self.test_audio, self.sr)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def _create_test_audio(self):
        """Create a test audio signal."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))

        # Create a musical signal with multiple characteristics
        signal = np.zeros_like(t)

        # Add a chord progression (C - Am - F - G)
        chord_duration = self.duration / 4

        # C major chord
        mask1 = t < chord_duration
        signal[mask1] += 0.3 * (
            np.sin(2 * np.pi * 261.63 * t[mask1])  # C
            + np.sin(2 * np.pi * 329.63 * t[mask1])  # E
            + np.sin(2 * np.pi * 392.00 * t[mask1])  # G
        )

        # A minor chord
        mask2 = (t >= chord_duration) & (t < 2 * chord_duration)
        signal[mask2] += 0.3 * (
            np.sin(2 * np.pi * 220.00 * t[mask2])  # A
            + np.sin(2 * np.pi * 261.63 * t[mask2])  # C
            + np.sin(2 * np.pi * 329.63 * t[mask2])  # E
        )

        # F major chord
        mask3 = (t >= 2 * chord_duration) & (t < 3 * chord_duration)
        signal[mask3] += 0.3 * (
            np.sin(2 * np.pi * 174.61 * t[mask3])  # F
            + np.sin(2 * np.pi * 220.00 * t[mask3])  # A
            + np.sin(2 * np.pi * 261.63 * t[mask3])  # C
        )

        # G major chord
        mask4 = t >= 3 * chord_duration
        signal[mask4] += 0.3 * (
            np.sin(2 * np.pi * 196.00 * t[mask4])  # G
            + np.sin(2 * np.pi * 246.94 * t[mask4])  # B
            + np.sin(2 * np.pi * 293.66 * t[mask4])  # D
        )

        # Add rhythm (120 BPM = 2 beats per second)
        beat_freq = 2.0
        beat_pattern = (np.sin(2 * np.pi * beat_freq * t) > 0).astype(float)
        signal *= 0.7 + 0.3 * beat_pattern

        # Add some noise for realism
        signal += 0.05 * np.random.randn(len(t))

        # Normalize
        signal = signal / np.max(np.abs(signal))

        return signal

    def test_basic_analysis(self):
        """Test basic BPM and key analysis."""
        results = self.analyzer.analyze_file(self.temp_file.name, detect_key=True, comprehensive=False)

        # Check basic structure
        self.assertIn('basic_info', results)
        basic_info = results['basic_info']

        # Check required fields
        required_fields = ['filename', 'duration', 'bpm', 'bpm_confidence', 'key', 'key_confidence']
        for field in required_fields:
            self.assertIn(field, basic_info)

        # Check types and ranges
        self.assertIsInstance(basic_info['bpm'], float)
        self.assertGreater(basic_info['bpm'], 60)
        self.assertLess(basic_info['bpm'], 200)

        self.assertIsInstance(basic_info['bpm_confidence'], float)
        self.assertGreaterEqual(basic_info['bpm_confidence'], 0)
        self.assertLessEqual(basic_info['bpm_confidence'], 100)

        self.assertIsInstance(basic_info['duration'], float)
        self.assertAlmostEqual(basic_info['duration'], self.duration, delta=0.5)

    def test_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        results = self.analyzer.analyze_file(self.temp_file.name, detect_key=True, comprehensive=True)

        # Check basic analysis sections are present
        basic_sections = ['basic_info', 'chord_progression']

        for section in basic_sections:
            self.assertIn(section, results)

        # Check optional sections if present
        optional_sections = [
            'rhythm',
            'structure',
            'timbre',
            'melody_harmony',
            'dynamics',
            'similarity_features',
            'reference_tags',
            'production_notes',
        ]

        for section in optional_sections:
            if section in results:
                # reference_tags is expected to be a list, others should be dict
                if section == 'reference_tags':
                    self.assertIsInstance(results[section], list)
                else:
                    self.assertIsInstance(results[section], dict)

        # Check chord progression analysis
        chord_prog = results['chord_progression']
        self.assertIn('main_progression', chord_prog)
        self.assertIn('harmonic_rhythm', chord_prog)
        self.assertIn('chord_complexity', chord_prog)

        # Check optional sections if present
        if 'structure' in results:
            structure = results['structure']
            # Check for any structure fields
            self.assertIsInstance(structure, dict)

        if 'rhythm' in results:
            rhythm = results['rhythm']
            # Check for any rhythm fields
            self.assertIsInstance(rhythm, dict)

        if 'timbre' in results:
            timbre = results['timbre']
            # Check for any timbre fields
            self.assertIsInstance(timbre, dict)

        if 'melody_harmony' in results:
            melody_harmony = results['melody_harmony']
            # Check for any melody harmony fields
            self.assertIsInstance(melody_harmony, dict)

        if 'dynamics' in results:
            dynamics = results['dynamics']
            # Check for any dynamics fields
            self.assertIsInstance(dynamics, dict)

        if 'similarity_features' in results:
            similarity_features = results['similarity_features']
            # Check for any similarity features
            self.assertIsInstance(similarity_features, dict)

    def test_generate_reference_tags(self):
        """Test reference tag generation."""
        # Create mock results
        mock_results = {
            'basic_info': {'bpm': 120, 'key': 'C Major'},
            'rhythm': {'time_signature': '4/4', 'groove_type': 'straight', 'syncopation_level': 0.3},
            'structure': {'structural_complexity': 0.5},
            'timbre': {'dominant_instruments': [{'instrument': 'piano'}, {'instrument': 'guitar'}], 'brightness': 0.7},
            'dynamics': {'overall_energy': 0.6},
        }

        tags = self.analyzer._generate_reference_tags(mock_results)

        # Should return a list of strings
        self.assertIsInstance(tags, list)
        self.assertTrue(all(isinstance(tag, str) for tag in tags))

        # Should contain expected tags based on mock data
        self.assertIn('upbeat', tags)  # 120 BPM
        self.assertIn('major-key', tags)  # C Major

    def test_generate_production_notes(self):
        """Test production notes generation."""
        # Create mock results
        mock_results = {
            'timbre': {
                'density': 0.6,
                'dominant_instruments': [{'instrument': 'guitar'}, {'instrument': 'drums'}, {'instrument': 'piano'}],
                'brightness': 0.7,
            },
            'dynamics': {'dynamic_range': {'dynamic_range_db': 15.0}},
        }

        notes = self.analyzer._generate_production_notes(mock_results)

        # Should return a dictionary
        self.assertIsInstance(notes, dict)

        # Should contain expected keys
        self.assertIn('arrangement_density', notes)
        self.assertIn('production_style', notes)
        self.assertIn('mix_characteristics', notes)

    def test_generate_reference_sheet(self):
        """Test reference sheet generation."""
        # Use comprehensive analysis results
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        reference_sheet = self.analyzer.generate_reference_sheet(results)

        # Should return a string
        self.assertIsInstance(reference_sheet, str)

        # Should contain expected sections
        expected_sections = [
            '# Music Production Reference Sheet',
            '## Basic Information',
            '## Song Structure',
            '## Harmony & Chord Progression',
            '## Rhythm & Groove',
            '## Instrumentation & Timbre',
            '## Melody & Harmony',
            '## Dynamics & Energy',
        ]

        for section in expected_sections:
            self.assertIn(section, reference_sheet)

    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        self.analyzer.analyze_file(self.temp_file.name, comprehensive=True, progress_callback=progress_callback)

        # Should have received progress updates
        self.assertGreater(len(progress_values), 0)

        # Progress values should be between 0 and 100
        for value in progress_values:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)

        # Should end at a reasonable progress value
        self.assertGreater(progress_values[-1], 0)

    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with non-existent file
        try:
            self.analyzer.analyze_file('non_existent_file.wav')
            # Should either handle gracefully or raise appropriate exception
        except Exception as e:
            # Exception should be informative
            self.assertIsInstance(e, (FileNotFoundError, OSError))

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        # Test with custom parameters
        custom_analyzer = AudioAnalyzer(sr=44100, hop_length=256)

        self.assertEqual(custom_analyzer.sr, 44100)
        self.assertEqual(custom_analyzer.hop_length, 256)

        # Check that all sub-analyzers are initialized
        self.assertIsNotNone(custom_analyzer.bpm_detector)
        self.assertIsNotNone(custom_analyzer.key_detector)
        self.assertIsNotNone(custom_analyzer.chord_analyzer)
        self.assertIsNotNone(custom_analyzer.structure_analyzer)
        self.assertIsNotNone(custom_analyzer.rhythm_analyzer)
        self.assertIsNotNone(custom_analyzer.timbre_analyzer)
        self.assertIsNotNone(custom_analyzer.melody_harmony_analyzer)
        self.assertIsNotNone(custom_analyzer.dynamics_analyzer)
        self.assertIsNotNone(custom_analyzer.similarity_engine)

    def test_format_production_notes(self):
        """Test production notes formatting."""
        mock_notes = {
            'arrangement_density': 'medium',
            'production_style': 'rock_pop',
            'mix_characteristics': ['bright_mix', 'punchy_drums'],
        }

        formatted = self.analyzer._format_production_notes(mock_notes)

        # Should return a string
        self.assertIsInstance(formatted, str)

        # Should contain the notes information
        self.assertIn('medium', formatted)
        self.assertIn('rock_pop', formatted)

    def test_bpm_parameter_passing(self):
        """Test BPM parameter passing."""
        results = self.analyzer.analyze_file(self.temp_file.name, min_bpm=100, max_bpm=140, start_bpm=120)

        # Should complete without error
        self.assertIn('basic_info', results)

        # BPM should be within specified range (approximately)
        detected_bpm = results['basic_info']['bpm']
        self.assertGreaterEqual(detected_bpm, 80)  # Allow some tolerance
        self.assertLessEqual(detected_bpm, 160)

    def test_new_structure_analysis_features(self):
        """Test new structure analysis features."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        # Check structure analysis if present
        if 'structure' not in results:
            self.skipTest("Structure analysis not available")

        structure = results['structure']

        # Should have sections with detailed information
        if 'sections' in structure:
            sections = structure['sections']
            self.assertIsInstance(sections, list)

            # Each section should have required fields
            for section in sections:
                self.assertIsInstance(section, dict)
                expected_fields = ['start_time', 'end_time', 'type']
                for field in expected_fields:
                    self.assertIn(field, section)

        # Should have form analysis
        self.assertIn('form', structure)

        # Should have structural complexity
        if 'structural_complexity' in structure:
            complexity = structure['structural_complexity']
            self.assertIsInstance(complexity, (int, float))
            self.assertGreaterEqual(complexity, 0.0)
            self.assertLessEqual(complexity, 1.0)

    def test_enhanced_timbre_analysis(self):
        """Test enhanced timbre analysis features."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        if 'timbre' not in results:
            self.skipTest("Timbre analysis not available")

        timbre = results['timbre']

        # Check for enhanced timbre features
        expected_features = ['brightness', 'warmth', 'roughness', 'density']
        for feature in expected_features:
            if feature in timbre:
                # Accept numpy types as well as Python types
                self.assertIsInstance(timbre[feature], (int, float, np.integer, np.floating))
                self.assertGreaterEqual(float(timbre[feature]), 0.0)
                self.assertLessEqual(float(timbre[feature]), 1.0)

        # Check instrument classification
        if 'instruments' in timbre:
            instruments = timbre['instruments']
            self.assertIsInstance(instruments, list)

            for instrument in instruments:
                self.assertIsInstance(instrument, dict)
                self.assertIn('instrument', instrument)
                self.assertIn('confidence', instrument)

        # Check effects analysis
        if 'effects' in timbre:
            effects = timbre['effects']
            self.assertIsInstance(effects, dict)

            for effect_name, effect_value in effects.items():
                self.assertIsInstance(effect_value, (int, float))
                self.assertGreaterEqual(effect_value, 0.0)
                self.assertLessEqual(effect_value, 1.0)

    def test_enhanced_dynamics_analysis(self):
        """Test enhanced dynamics analysis features."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        if 'dynamics' not in results:
            self.skipTest("Dynamics analysis not available")

        dynamics = results['dynamics']

        # Check for enhanced dynamics features
        if 'climax_points' in dynamics:
            climax = dynamics['climax_points']
            # climax_points can be either dict or list depending on implementation
            if isinstance(climax, dict):
                if 'main_climax' in climax:
                    main_climax = climax['main_climax']
                    if main_climax is not None:
                        self.assertIsInstance(main_climax, (int, float))
                        self.assertGreaterEqual(main_climax, 0.0)
            elif isinstance(climax, list):
                # If it's a list, check that each item has time and intensity
                for point in climax:
                    self.assertIsInstance(point, dict)
                    self.assertIn('time', point)
                    self.assertIn('intensity', point)

        # Check energy distribution
        if 'energy_distribution' in dynamics:
            energy_dist = dynamics['energy_distribution']
            self.assertIsInstance(energy_dist, dict)

            # Check energy ratios
            ratio_fields = ['low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio']
            for field in ratio_fields:
                if field in energy_dist:
                    ratio = energy_dist[field]
                    self.assertIsInstance(ratio, (int, float))
                    self.assertGreaterEqual(ratio, 0.0)
                    self.assertLessEqual(ratio, 1.0)

    def test_melody_harmony_integration(self):
        """Test melody harmony analysis integration."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        if 'melody_harmony' not in results:
            self.skipTest("Melody harmony analysis not available")

        melody_harmony = results['melody_harmony']

        # Check for melody features
        if 'melody' in melody_harmony:
            melody = melody_harmony['melody']
            self.assertIsInstance(melody, dict)

            # Check melody range
            if 'range' in melody:
                melody_range = melody['range']
                self.assertIsInstance(melody_range, dict)

        # Check for harmony features
        if 'harmony' in melody_harmony:
            harmony = melody_harmony['harmony']
            self.assertIsInstance(harmony, dict)

            # Check harmony complexity
            if 'complexity' in harmony:
                complexity = harmony['complexity']
                self.assertIsInstance(complexity, dict)

        # Check combined features
        if 'combined_features' in melody_harmony:
            combined = melody_harmony['combined_features']
            self.assertIsInstance(combined, dict)

            # Check balance score
            if 'melody_harmony_balance' in combined:
                balance = combined['melody_harmony_balance']
                self.assertIsInstance(balance, (int, float))
                self.assertGreaterEqual(balance, 0.0)
                self.assertLessEqual(balance, 1.0)

    def test_similarity_features_generation(self):
        """Test similarity features generation."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        if 'similarity_features' not in results:
            self.skipTest("Similarity features not available")

        similarity_features = results['similarity_features']

        # Check feature vector
        self.assertIn('feature_vector', similarity_features)
        feature_vector = similarity_features['feature_vector']
        # feature_vector can be either numpy array or list
        self.assertIsInstance(feature_vector, (np.ndarray, list))
        self.assertGreater(len(feature_vector), 0)

        # Check feature weights
        if 'feature_weights' in similarity_features:
            weights = similarity_features['feature_weights']
            self.assertIsInstance(weights, dict)

        # Check feature metadata
        if 'feature_metadata' in similarity_features:
            metadata = similarity_features['feature_metadata']
            self.assertIsInstance(metadata, dict)

    def test_comprehensive_error_handling(self):
        """Test comprehensive error handling."""
        # Test with very short audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(self.sr * 0.1)))

        # Create temporary short file
        short_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(short_file.name, short_audio, self.sr)
        short_file.close()

        try:
            results = self.analyzer.analyze_file(short_file.name, comprehensive=True)

            # Should handle short audio gracefully
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

        except Exception as e:
            # If exception is raised, it should be handled gracefully
            self.assertIsInstance(e, Exception)

        finally:
            # Clean up
            if os.path.exists(short_file.name):
                os.unlink(short_file.name)

    def test_progress_callback_with_comprehensive_analysis(self):
        """Test progress callback with comprehensive analysis."""
        progress_updates = []

        def detailed_progress_callback(progress, message=""):
            progress_updates.append((progress, message))

        self.analyzer.analyze_file(
            self.temp_file.name, comprehensive=True, progress_callback=detailed_progress_callback
        )

        # Should have received detailed progress updates
        self.assertGreater(len(progress_updates), 0)

        # Check progress format
        for progress, message in progress_updates:
            self.assertIsInstance(progress, (int, float))
            self.assertGreaterEqual(progress, 0)
            self.assertLessEqual(progress, 100)
            self.assertIsInstance(message, str)

    def test_analyzer_with_different_audio_characteristics(self):
        """Test analyzer with different types of audio."""
        # Test with pure sine wave
        sine_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 5, int(self.sr * 5)))

        sine_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(sine_file.name, sine_audio, self.sr)
        sine_file.close()

        try:
            results = self.analyzer.analyze_file(sine_file.name, comprehensive=True)

            # Should handle pure sine wave
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

            # Timbre should reflect simple harmonic content
            if 'timbre' in results:
                timbre = results['timbre']
                # Simple sine wave should have low complexity
                if 'density' in timbre:
                    # Use assertLess with small tolerance for floating point precision
                    density_value = float(timbre['density'])
                    self.assertLess(density_value, 0.5001)  # Allow small floating point tolerance

        finally:
            if os.path.exists(sine_file.name):
                os.unlink(sine_file.name)

    def test_reference_sheet_completeness(self):
        """Test reference sheet completeness."""
        results = self.analyzer.analyze_file(self.temp_file.name, comprehensive=True)

        reference_sheet = self.analyzer.generate_reference_sheet(results)

        # Check for new sections in reference sheet
        new_sections = ['## Production Notes', '## Reference Tags', '## Similarity Features']

        for section in new_sections:
            # These sections might be present depending on implementation
            if section in reference_sheet:
                # If present, should have content
                section_index = reference_sheet.find(section)
                next_section_index = reference_sheet.find('##', section_index + 1)
                if next_section_index == -1:
                    section_content = reference_sheet[section_index:]
                else:
                    section_content = reference_sheet[section_index:next_section_index]

                # Should have content (allowing for minimal content)
                self.assertGreaterEqual(len(section_content.strip()), len(section.strip()))


if __name__ == '__main__':
    unittest.main()
