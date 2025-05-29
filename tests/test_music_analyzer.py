"""Tests for the integrated music analyzer."""

import unittest
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
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
            np.sin(2 * np.pi * 261.63 * t[mask1]) +  # C
            np.sin(2 * np.pi * 329.63 * t[mask1]) +  # E
            np.sin(2 * np.pi * 392.00 * t[mask1])    # G
        )
        
        # A minor chord
        mask2 = (t >= chord_duration) & (t < 2 * chord_duration)
        signal[mask2] += 0.3 * (
            np.sin(2 * np.pi * 220.00 * t[mask2]) +  # A
            np.sin(2 * np.pi * 261.63 * t[mask2]) +  # C
            np.sin(2 * np.pi * 329.63 * t[mask2])    # E
        )
        
        # F major chord
        mask3 = (t >= 2 * chord_duration) & (t < 3 * chord_duration)
        signal[mask3] += 0.3 * (
            np.sin(2 * np.pi * 174.61 * t[mask3]) +  # F
            np.sin(2 * np.pi * 220.00 * t[mask3]) +  # A
            np.sin(2 * np.pi * 261.63 * t[mask3])    # C
        )
        
        # G major chord
        mask4 = t >= 3 * chord_duration
        signal[mask4] += 0.3 * (
            np.sin(2 * np.pi * 196.00 * t[mask4]) +  # G
            np.sin(2 * np.pi * 246.94 * t[mask4]) +  # B
            np.sin(2 * np.pi * 293.66 * t[mask4])    # D
        )
        
        # Add rhythm (120 BPM = 2 beats per second)
        beat_freq = 2.0
        beat_pattern = (np.sin(2 * np.pi * beat_freq * t) > 0).astype(float)
        signal *= (0.7 + 0.3 * beat_pattern)
        
        # Add some noise for realism
        signal += 0.05 * np.random.randn(len(t))
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal
        
    def test_basic_analysis(self):
        """Test basic BPM and key analysis."""
        results = self.analyzer.analyze_file(
            self.temp_file.name,
            detect_key=True,
            comprehensive=False
        )
        
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
        results = self.analyzer.analyze_file(
            self.temp_file.name,
            detect_key=True,
            comprehensive=True
        )
        
        # Check all analysis sections are present
        expected_sections = [
            'basic_info', 'chord_progression', 'structure', 'rhythm',
            'timbre', 'melody_harmony', 'dynamics', 'similarity_features',
            'reference_tags', 'production_notes'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results)
            
        # Check chord progression analysis
        chord_prog = results['chord_progression']
        self.assertIn('main_progression', chord_prog)
        self.assertIn('harmonic_rhythm', chord_prog)
        self.assertIn('chord_complexity', chord_prog)
        
        # Check structure analysis
        structure = results['structure']
        self.assertIn('sections', structure)
        self.assertIn('form', structure)
        self.assertIn('repetition_ratio', structure)
        
        # Check rhythm analysis
        rhythm = results['rhythm']
        self.assertIn('time_signature', rhythm)
        self.assertIn('groove_type', rhythm)
        self.assertIn('syncopation_level', rhythm)
        
        # Check timbre analysis
        timbre = results['timbre']
        self.assertIn('brightness', timbre)
        self.assertIn('dominant_instruments', timbre)
        
        # Check melody harmony analysis
        melody_harmony = results['melody_harmony']
        self.assertIn('melodic_range', melody_harmony)
        self.assertIn('consonance', melody_harmony)
        
        # Check dynamics analysis
        dynamics = results['dynamics']
        self.assertIn('dynamic_range', dynamics)
        self.assertIn('energy_profile', dynamics)
        
        # Check similarity features
        similarity_features = results['similarity_features']
        self.assertIn('feature_vector', similarity_features)
        self.assertIn('feature_weights', similarity_features)
        
    def test_generate_reference_tags(self):
        """Test reference tag generation."""
        # Create mock results
        mock_results = {
            'basic_info': {'bpm': 120, 'key': 'C Major'},
            'rhythm': {'time_signature': '4/4', 'groove_type': 'straight', 'syncopation_level': 0.3},
            'structure': {'structural_complexity': 0.5},
            'timbre': {
                'dominant_instruments': [{'instrument': 'piano'}, {'instrument': 'guitar'}],
                'brightness': 0.7
            },
            'dynamics': {'overall_energy': 0.6}
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
                'dominant_instruments': [
                    {'instrument': 'guitar'},
                    {'instrument': 'drums'},
                    {'instrument': 'piano'}
                ],
                'brightness': 0.7
            },
            'dynamics': {
                'dynamic_range': {'dynamic_range_db': 15.0}
            }
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
        results = self.analyzer.analyze_file(
            self.temp_file.name,
            comprehensive=True
        )
        
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
            '## Dynamics & Energy'
        ]
        
        for section in expected_sections:
            self.assertIn(section, reference_sheet)
            
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_values = []
        
        def progress_callback(value):
            progress_values.append(value)
            
        results = self.analyzer.analyze_file(
            self.temp_file.name,
            comprehensive=True,
            progress_callback=progress_callback
        )
        
        # Should have received progress updates
        self.assertGreater(len(progress_values), 0)
        
        # Progress values should be between 0 and 100
        for value in progress_values:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)
            
        # Should end at 100%
        self.assertEqual(progress_values[-1], 100)
        
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with non-existent file
        try:
            results = self.analyzer.analyze_file('non_existent_file.wav')
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
            'mix_characteristics': ['bright_mix', 'punchy_drums']
        }
        
        formatted = self.analyzer._format_production_notes(mock_notes)
        
        # Should return a string
        self.assertIsInstance(formatted, str)
        
        # Should contain the notes information
        self.assertIn('medium', formatted)
        self.assertIn('rock_pop', formatted)
        
    def test_bpm_parameter_passing(self):
        """Test BPM parameter passing."""
        results = self.analyzer.analyze_file(
            self.temp_file.name,
            min_bpm=100,
            max_bpm=140,
            start_bpm=120
        )
        
        # Should complete without error
        self.assertIn('basic_info', results)
        
        # BPM should be within specified range (approximately)
        detected_bpm = results['basic_info']['bpm']
        self.assertGreaterEqual(detected_bpm, 80)  # Allow some tolerance
        self.assertLessEqual(detected_bpm, 160)


if __name__ == '__main__':
    unittest.main()