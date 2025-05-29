"""Tests for similarity engine."""

import unittest
import numpy as np
from src.bpm_detector.similarity_engine import SimilarityEngine


class TestSimilarityEngine(unittest.TestCase):
    """Test cases for SimilarityEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = SimilarityEngine()
        
        # Create mock analysis results
        self.mock_results = {
            'basic_info': {
                'bpm': 120.0,
                'key': 'C Major',
                'duration': 180.0
            },
            'chord_progression': {
                'harmonic_rhythm': 2.0,
                'chord_complexity': 0.6,
                'substitute_chords_ratio': 0.1,
                'modulations': []
            },
            'rhythm': {
                'time_signature': '4/4',
                'rhythmic_complexity': 0.5,
                'syncopation_level': 0.3,
                'pattern_regularity': 0.7,
                'subdivision_density': 0.4,
                'swing_ratio': 0.5,
                'polyrhythm_detected': False,
                'groove_type': 'straight'
            },
            'timbre': {
                'brightness': 0.7,
                'roughness': 0.3,
                'warmth': 0.8,
                'density': 0.5,
                'dominant_instruments': [
                    {'instrument': 'piano', 'confidence': 0.8, 'prominence': 0.7},
                    {'instrument': 'guitar', 'confidence': 0.6, 'prominence': 0.5}
                ],
                'effects_usage': {
                    'reverb': 0.4,
                    'distortion': 0.2,
                    'chorus': 0.3,
                    'compression': 0.6
                },
                'texture': {
                    'smoothness': 0.6,
                    'richness': 0.7,
                    'clarity': 0.8,
                    'fullness': 0.5
                }
            },
            'structure': {
                'repetition_ratio': 0.6,
                'structural_complexity': 0.4,
                'section_count': 5,
                'unique_sections': 3,
                'sections': [
                    {'type': 'intro', 'duration': 8},
                    {'type': 'verse', 'duration': 16}
                ]
            },
            'melody_harmony': {
                'melodic_range': {
                    'range_octaves': 2.0,
                    'pitch_std': 6.0
                },
                'melodic_direction': {
                    'ascending_ratio': 0.4,
                    'descending_ratio': 0.3,
                    'contour_complexity': 0.5,
                    'average_step_size': 2.0
                },
                'interval_distribution': {
                    'unison': 0.1,
                    'major_second': 0.3,
                    'major_third': 0.2,
                    'perfect_fourth': 0.15,
                    'perfect_fifth': 0.1
                },
                'pitch_stability': {
                    'pitch_stability': 0.8,
                    'vibrato_rate': 5.0,
                    'vibrato_extent': 0.1
                },
                'melody_present': True,
                'melody_coverage': 0.7,
                'harmony_complexity': {
                    'harmonic_complexity': 0.6
                },
                'consonance': {
                    'consonance_level': 0.8
                }
            },
            'dynamics': {
                'dynamic_range': {
                    'dynamic_range_db': 20.0,
                    'peak_to_average_ratio': 3.0,
                    'crest_factor': 4.0,
                    'dynamic_variance': 50.0
                },
                'loudness': {
                    'average_loudness_db': -15.0,
                    'perceived_loudness': 0.7
                },
                'overall_energy': 0.6,
                'energy_variance': 0.3,
                'energy_distribution': {
                    'low_freq_ratio': 0.3,
                    'mid_freq_ratio': 0.5,
                    'high_freq_ratio': 0.2,
                    'spectral_balance': 0.6
                },
                'climax_points': [
                    {'time': 60.0, 'intensity': 0.9},
                    {'time': 120.0, 'intensity': 0.8}
                ]
            }
        }
        
    def test_extract_feature_vector(self):
        """Test feature vector extraction."""
        feature_vector = self.engine.extract_feature_vector(self.mock_results)
        
        # Should return a numpy array
        self.assertIsInstance(feature_vector, np.ndarray)
        
        # Should have reasonable length (all features combined)
        self.assertGreater(len(feature_vector), 50)
        self.assertLess(len(feature_vector), 200)
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(feature_vector)))
        
        # Most values should be in [0, 1] range (normalized)
        in_range_count = np.sum((feature_vector >= 0) & (feature_vector <= 1))
        self.assertGreater(in_range_count, len(feature_vector) * 0.8)
        
    def test_key_to_numeric(self):
        """Test key to numeric conversion."""
        # Test major keys
        c_major = self.engine._key_to_numeric('C Major')
        self.assertEqual(c_major, 0.0)
        
        g_major = self.engine._key_to_numeric('G Major')
        self.assertGreater(g_major, 0.0)
        self.assertLess(g_major, 1.0)
        
        # Test minor keys
        a_minor = self.engine._key_to_numeric('A Minor')
        self.assertGreater(a_minor, 0.0)  # Minor keys should be greater than 0
        
        # Test invalid input
        invalid = self.engine._key_to_numeric('Invalid Key')
        self.assertEqual(invalid, 0.5)  # Default value for unknown keys
        
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        # Create two similar feature vectors
        vector1 = np.array([0.5, 0.6, 0.7, 0.8])
        vector2 = np.array([0.5, 0.6, 0.7, 0.8])
        
        # Test cosine similarity
        cosine_sim = self.engine.calculate_similarity(vector1, vector2, 'cosine')
        self.assertAlmostEqual(cosine_sim, 1.0, places=5)
        
        # Test euclidean similarity
        euclidean_sim = self.engine.calculate_similarity(vector1, vector2, 'euclidean')
        self.assertAlmostEqual(euclidean_sim, 1.0, places=5)
        
        # Test with different vectors
        vector3 = np.array([0.1, 0.2, 0.3, 0.4])
        cosine_sim_diff = self.engine.calculate_similarity(vector1, vector3, 'cosine')
        self.assertLess(cosine_sim_diff, 1.0)
        self.assertGreater(cosine_sim_diff, 0.0)
        
    def test_calculate_weighted_similarity(self):
        """Test weighted similarity calculation."""
        # Create feature vectors with known structure
        vector1 = np.random.rand(77)  # Expected feature vector length
        vector2 = np.random.rand(77)
        
        weighted_sim = self.engine._calculate_weighted_similarity(vector1, vector2)
        
        # Should return a value between 0 and 1
        self.assertGreaterEqual(weighted_sim, 0.0)
        self.assertLessEqual(weighted_sim, 1.0)
        
    def test_find_similar_tracks(self):
        """Test similar track finding."""
        # Create target vector
        target_vector = np.random.rand(50)
        
        # Create database of tracks
        database_vectors = [
            ('track1', np.random.rand(50)),
            ('track2', np.random.rand(50)),
            ('track3', target_vector + 0.1 * np.random.rand(50)),  # Similar to target
            ('track4', np.random.rand(50))
        ]
        
        similar_tracks = self.engine.find_similar_tracks(
            target_vector, database_vectors, top_k=3
        )
        
        # Should return list of tuples
        self.assertIsInstance(similar_tracks, list)
        self.assertLessEqual(len(similar_tracks), 3)
        
        # Each result should be (track_id, similarity_score)
        for track_id, similarity in similar_tracks:
            self.assertIsInstance(track_id, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
            
        # Results should be sorted by similarity (highest first)
        similarities = [sim for _, sim in similar_tracks]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        
    def test_generate_similarity_matrix(self):
        """Test similarity matrix generation."""
        # Create feature vectors
        vectors = [
            np.random.rand(20),
            np.random.rand(20),
            np.random.rand(20)
        ]
        
        similarity_matrix = self.engine.generate_similarity_matrix(vectors)
        
        # Should be square matrix
        self.assertEqual(similarity_matrix.shape[0], similarity_matrix.shape[1])
        self.assertEqual(similarity_matrix.shape[0], len(vectors))
        
        # Diagonal should be 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        np.testing.assert_allclose(diagonal, 1.0, rtol=1e-10)
        
        # Should be symmetric
        np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, rtol=1e-10)
        
        # Values should be in [0, 1] range
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))
        
    def test_fit_and_normalize_features(self):
        """Test feature scaling."""
        # Create feature vectors
        vectors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([0.5, 1.0, 1.5])
        ]
        
        # Fit scaler
        self.engine.fit_scaler(vectors)
        self.assertTrue(self.engine.is_fitted)
        
        # Normalize a vector
        test_vector = np.array([1.5, 3.0, 4.5])
        normalized = self.engine.normalize_features(test_vector)
        
        # Should return normalized vector
        self.assertEqual(len(normalized), len(test_vector))
        self.assertTrue(np.all(np.isfinite(normalized)))
        
    def test_reduce_dimensionality(self):
        """Test dimensionality reduction."""
        # Create high-dimensional feature vectors
        vectors = [np.random.rand(100) for _ in range(10)]
        
        reduced_vectors, pca_model = self.engine.reduce_dimensionality(vectors, n_components=5)
        
        # Should reduce dimensionality
        self.assertEqual(reduced_vectors.shape[0], len(vectors))
        self.assertEqual(reduced_vectors.shape[1], 5)
        
        # PCA model should be returned
        self.assertIsNotNone(pca_model)
        
    def test_export_import_feature_vector(self):
        """Test feature vector export/import."""
        feature_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metadata = {'track_name': 'test_track', 'artist': 'test_artist'}
        
        # Export
        exported = self.engine.export_feature_vector(feature_vector, metadata)
        
        # Check export format
        self.assertIn('feature_vector', exported)
        self.assertIn('metadata', exported)
        self.assertIn('feature_weights', exported)
        self.assertIn('vector_length', exported)
        self.assertIn('version', exported)
        
        # Import
        imported_vector, imported_metadata = self.engine.import_feature_vector(exported)
        
        # Check imported data
        np.testing.assert_array_almost_equal(imported_vector, feature_vector, decimal=6)
        self.assertEqual(imported_metadata, metadata)
        
    def test_extract_harmonic_features(self):
        """Test harmonic feature extraction."""
        harmonic_features = self.engine._extract_harmonic_features(self.mock_results)
        
        # Should return a list of floats
        self.assertIsInstance(harmonic_features, list)
        self.assertTrue(all(isinstance(f, float) for f in harmonic_features))
        
        # Should have expected number of features
        self.assertGreater(len(harmonic_features), 5)
        
    def test_extract_rhythmic_features(self):
        """Test rhythmic feature extraction."""
        rhythmic_features = self.engine._extract_rhythmic_features(self.mock_results)
        
        # Should return a list of floats
        self.assertIsInstance(rhythmic_features, list)
        self.assertTrue(all(isinstance(f, float) for f in rhythmic_features))
        
        # Should have expected number of features
        self.assertGreater(len(rhythmic_features), 5)
        
    def test_time_signature_to_numeric(self):
        """Test time signature conversion."""
        # Test common time signatures
        four_four = self.engine._time_signature_to_numeric('4/4')
        self.assertEqual(four_four, 0.0)
        
        three_four = self.engine._time_signature_to_numeric('3/4')
        self.assertGreater(three_four, 0.0)
        
        # Test unknown time signature
        unknown = self.engine._time_signature_to_numeric('13/16')
        self.assertEqual(unknown, 0.0)
        
    def test_empty_results(self):
        """Test behavior with empty analysis results."""
        empty_results = {}
        
        # Should handle empty results gracefully
        feature_vector = self.engine.extract_feature_vector(empty_results)
        
        # Should return a vector with default values
        self.assertIsInstance(feature_vector, np.ndarray)
        self.assertGreater(len(feature_vector), 0)
        
    def test_feature_weights(self):
        """Test feature weight configuration."""
        # Test default weights
        default_weights = self.engine.feature_weights
        self.assertIn('harmonic', default_weights)
        self.assertIn('rhythmic', default_weights)
        self.assertIn('timbral', default_weights)
        
        # Test custom weights
        custom_weights = {'harmonic': 0.5, 'rhythmic': 0.3, 'timbral': 0.2}
        custom_engine = SimilarityEngine(custom_weights)
        self.assertEqual(custom_engine.feature_weights['harmonic'], 0.5)


if __name__ == '__main__':
    unittest.main()