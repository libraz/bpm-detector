"""Similarity calculation and feature vector generation engine."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SimilarityEngine:
    """Generates feature vectors and calculates similarity between tracks."""

    # Feature weights for different aspects of music
    DEFAULT_FEATURE_WEIGHTS = {
        'harmonic': 0.25,  # Chord progressions, key, harmony
        'rhythmic': 0.20,  # BPM, time signature, rhythm patterns
        'timbral': 0.20,  # Instruments, brightness, texture
        'structural': 0.15,  # Song structure, form
        'melodic': 0.10,  # Melody characteristics
        'dynamic': 0.10,  # Energy, dynamics, loudness
    }

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """Initialize similarity engine.

        Args:
            feature_weights: Custom weights for different feature categories
        """
        self.feature_weights = feature_weights or self.DEFAULT_FEATURE_WEIGHTS
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_feature_vector(self, analysis_results: Dict[str, Any]) -> np.ndarray:
        """Extract comprehensive feature vector from analysis results.

        Args:
            analysis_results: Complete analysis results from all analyzers

        Returns:
            Feature vector array
        """
        features = []

        # Basic info features
        basic_info = analysis_results.get('basic_info', {})
        features.extend(
            [
                basic_info.get('bpm', 120.0) / 200.0,  # Normalize BPM
                self._key_to_numeric(basic_info.get('key', 'C Major')),
                basic_info.get('duration', 180.0) / 300.0,  # Normalize duration
            ]
        )

        # Harmonic features
        harmonic_features = self._extract_harmonic_features(analysis_results)
        features.extend(harmonic_features)

        # Rhythmic features
        rhythmic_features = self._extract_rhythmic_features(analysis_results)
        features.extend(rhythmic_features)

        # Timbral features
        timbral_features = self._extract_timbral_features(analysis_results)
        features.extend(timbral_features)

        # Structural features
        structural_features = self._extract_structural_features(analysis_results)
        features.extend(structural_features)

        # Melodic features
        melodic_features = self._extract_melodic_features(analysis_results)
        features.extend(melodic_features)

        # Dynamic features
        dynamic_features = self._extract_dynamic_features(analysis_results)
        features.extend(dynamic_features)

        return np.array(features, dtype=np.float32)

    def _key_to_numeric(self, key: str) -> float:
        """Convert key to numeric representation.

        Args:
            key: Key string (e.g., 'C Major', 'A Minor')

        Returns:
            Numeric representation (0-1)
        """
        if not key or len(key.split()) != 2:
            return 0.0

        note, mode = key.split()

        # Convert note to number (0-11)
        note_map = {
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
            'A': 9,
            'A#': 10,
            'B': 11,
        }

        note_num = note_map.get(note, 0)
        mode_num = 0 if mode.lower() == 'major' else 1

        # Combine note and mode into single value
        return (note_num + mode_num * 12) / 24.0

    def _extract_harmonic_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract harmonic features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of harmonic features
        """
        features = []

        # Chord progression features
        chord_analysis = analysis_results.get('chord_progression', {})
        features.extend(
            [
                chord_analysis.get('harmonic_rhythm', 0.0) / 5.0,  # Normalize
                chord_analysis.get('chord_complexity', 0.0),
                chord_analysis.get('substitute_chords_ratio', 0.0),
                len(chord_analysis.get('modulations', []))
                / 5.0,  # Normalize modulation count
            ]
        )

        # Melody harmony features
        melody_harmony = analysis_results.get('melody_harmony', {})
        harmony_complexity = melody_harmony.get('harmony_complexity', {})
        consonance = melody_harmony.get('consonance', {})

        features.extend(
            [
                harmony_complexity.get('harmonic_complexity', 0.0),
                harmony_complexity.get('harmonic_entropy', 0.0),
                consonance.get('consonance_level', 0.0),
                consonance.get('harmonic_tension', 0.0),
            ]
        )

        return features

    def _extract_rhythmic_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract rhythmic features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of rhythmic features
        """
        features = []

        rhythm_analysis = analysis_results.get('rhythm', {})

        # Time signature encoding
        time_sig = rhythm_analysis.get('time_signature', '4/4')
        time_sig_numeric = self._time_signature_to_numeric(time_sig)

        features.extend(
            [
                time_sig_numeric,
                rhythm_analysis.get('rhythmic_complexity', 0.0),
                rhythm_analysis.get('syncopation_level', 0.0),
                rhythm_analysis.get('pattern_regularity', 0.0),
                rhythm_analysis.get('subdivision_density', 0.0),
                rhythm_analysis.get('swing_ratio', 0.5),
                1.0 if rhythm_analysis.get('polyrhythm_detected', False) else 0.0,
            ]
        )

        # Groove type encoding
        groove_type = rhythm_analysis.get('groove_type', 'straight')
        groove_numeric = {'straight': 0.0, 'shuffle': 0.5, 'swing': 1.0}.get(
            groove_type, 0.0
        )
        features.append(groove_numeric)

        return features

    def _time_signature_to_numeric(self, time_sig: str) -> float:
        """Convert time signature to numeric representation.

        Args:
            time_sig: Time signature string (e.g., '4/4')

        Returns:
            Numeric representation
        """
        time_sig_map = {
            '4/4': 0.0,
            '3/4': 0.2,
            '2/4': 0.4,
            '6/8': 0.6,
            '9/8': 0.7,
            '12/8': 0.8,
            '5/4': 0.9,
            '7/8': 1.0,
        }
        return time_sig_map.get(time_sig, 0.0)

    def _extract_timbral_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract timbral features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of timbral features
        """
        features = []

        timbre_analysis = analysis_results.get('timbre', {})

        # Basic timbral characteristics
        features.extend(
            [
                timbre_analysis.get('brightness', 0.0),
                timbre_analysis.get('roughness', 0.0),
                timbre_analysis.get('warmth', 0.0),
                timbre_analysis.get('density', 0.0),
            ]
        )

        # Instrument presence (top 5 instruments)
        instruments = timbre_analysis.get('dominant_instruments', [])
        instrument_features = [0.0] * 10  # Fixed size for consistency

        for i, instrument in enumerate(instruments[:5]):
            if i < 5:
                instrument_features[i * 2] = instrument.get('confidence', 0.0)
                instrument_features[i * 2 + 1] = instrument.get('prominence', 0.0)

        features.extend(instrument_features)

        # Effects usage
        effects = timbre_analysis.get('effects_usage', {})
        features.extend(
            [
                effects.get('reverb', 0.0),
                effects.get('distortion', 0.0),
                effects.get('chorus', 0.0),
                effects.get('compression', 0.0),
            ]
        )

        # Texture characteristics
        texture = timbre_analysis.get('texture', {})
        features.extend(
            [
                texture.get('smoothness', 0.0),
                texture.get('richness', 0.0),
                texture.get('clarity', 0.0),
                texture.get('fullness', 0.0),
            ]
        )

        return features

    def _extract_structural_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract structural features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of structural features
        """
        features = []

        structure_analysis = analysis_results.get('structure', {})

        features.extend(
            [
                structure_analysis.get('repetition_ratio', 0.0),
                structure_analysis.get('structural_complexity', 0.0),
                structure_analysis.get('section_count', 0) / 10.0,  # Normalize
                structure_analysis.get('unique_sections', 0) / 6.0,  # Normalize
            ]
        )

        # Section type distribution
        sections = structure_analysis.get('sections', [])
        section_types = ['intro', 'verse', 'chorus', 'bridge', 'instrumental', 'outro']
        section_counts = {stype: 0 for stype in section_types}

        for section in sections:
            stype = section.get('type', 'unknown')
            if stype in section_counts:
                section_counts[stype] += 1

        total_sections = len(sections) if sections else 1
        section_ratios = [
            section_counts[stype] / total_sections for stype in section_types
        ]
        features.extend(section_ratios)

        return features

    def _extract_melodic_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract melodic features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of melodic features
        """
        features = []

        melody_harmony = analysis_results.get('melody_harmony', {})

        # Melodic range
        melodic_range = melody_harmony.get('melodic_range', {})
        features.extend(
            [
                melodic_range.get('range_octaves', 0.0) / 4.0,  # Normalize
                melodic_range.get('pitch_std', 0.0) / 12.0,  # Normalize
            ]
        )

        # Melodic direction
        melodic_direction = melody_harmony.get('melodic_direction', {})
        features.extend(
            [
                melodic_direction.get('ascending_ratio', 0.0),
                melodic_direction.get('descending_ratio', 0.0),
                melodic_direction.get('contour_complexity', 0.0),
                melodic_direction.get('average_step_size', 0.0) / 5.0,  # Normalize
            ]
        )

        # Interval distribution (top 5 intervals)
        interval_dist = melody_harmony.get('interval_distribution', {})
        interval_features = [
            interval_dist.get('unison', 0.0),
            interval_dist.get('major_second', 0.0),
            interval_dist.get('major_third', 0.0),
            interval_dist.get('perfect_fourth', 0.0),
            interval_dist.get('perfect_fifth', 0.0),
        ]
        features.extend(interval_features)

        # Pitch stability
        pitch_stability = melody_harmony.get('pitch_stability', {})
        features.extend(
            [
                pitch_stability.get('pitch_stability', 0.0),
                pitch_stability.get('vibrato_rate', 0.0) / 10.0,  # Normalize
                pitch_stability.get('vibrato_extent', 0.0),
            ]
        )

        # Melody presence
        features.extend(
            [
                1.0 if melody_harmony.get('melody_present', False) else 0.0,
                melody_harmony.get('melody_coverage', 0.0),
            ]
        )

        return features

    def _extract_dynamic_features(
        self, analysis_results: Dict[str, Any]
    ) -> List[float]:
        """Extract dynamic features.

        Args:
            analysis_results: Analysis results

        Returns:
            List of dynamic features
        """
        features = []

        dynamics_analysis = analysis_results.get('dynamics', {})

        # Dynamic range
        dynamic_range = dynamics_analysis.get('dynamic_range', {})
        features.extend(
            [
                dynamic_range.get('dynamic_range_db', 0.0) / 60.0,  # Normalize
                dynamic_range.get('peak_to_average_ratio', 0.0) / 10.0,  # Normalize
                dynamic_range.get('crest_factor', 0.0) / 10.0,  # Normalize
                dynamic_range.get('dynamic_variance', 0.0) / 100.0,  # Normalize
            ]
        )

        # Loudness
        loudness = dynamics_analysis.get('loudness', {})
        features.extend(
            [
                (loudness.get('average_loudness_db', -30.0) + 60.0) / 60.0,  # Normalize
                loudness.get('perceived_loudness', 0.0),
            ]
        )

        # Energy characteristics
        features.extend(
            [
                dynamics_analysis.get('overall_energy', 0.0),
                dynamics_analysis.get('energy_variance', 0.0),
            ]
        )

        # Energy distribution
        energy_dist = dynamics_analysis.get('energy_distribution', {})
        features.extend(
            [
                energy_dist.get('low_freq_ratio', 0.0),
                energy_dist.get('mid_freq_ratio', 0.0),
                energy_dist.get('high_freq_ratio', 0.0),
                energy_dist.get('spectral_balance', 0.0),
            ]
        )

        # Climax points
        climax_points = dynamics_analysis.get('climax_points', [])
        features.extend(
            [
                len(climax_points) / 5.0,  # Normalize climax count
                (
                    np.mean([cp.get('intensity', 0.0) for cp in climax_points])
                    if climax_points
                    else 0.0
                ),
            ]
        )

        return features

    def fit_scaler(self, feature_vectors: List[np.ndarray]) -> None:
        """Fit the feature scaler on a collection of feature vectors.

        Args:
            feature_vectors: List of feature vectors for fitting
        """
        if not feature_vectors:
            return

        # Stack all feature vectors
        all_features = np.vstack(feature_vectors)

        # Fit the scaler
        self.scaler.fit(all_features)
        self.is_fitted = True

    def normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """Normalize feature vector using fitted scaler.

        Args:
            feature_vector: Raw feature vector

        Returns:
            Normalized feature vector
        """
        if not self.is_fitted:
            # If scaler not fitted, use simple min-max normalization
            return np.clip(feature_vector, 0, 1)

        # Reshape for sklearn
        features_2d = feature_vector.reshape(1, -1)
        normalized = self.scaler.transform(features_2d)

        return normalized.flatten()

    def calculate_similarity(
        self, vector1: np.ndarray, vector2: np.ndarray, method: str = 'cosine'
    ) -> float:
        """Calculate similarity between two feature vectors.

        Args:
            vector1: First feature vector
            vector2: Second feature vector
            method: Similarity method ('cosine', 'euclidean', 'weighted')

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if len(vector1) != len(vector2):
            raise ValueError("Feature vectors must have the same length")

        if method == 'cosine':
            # Cosine similarity
            similarity = cosine_similarity([vector1], [vector2])[0, 0]
            return float(similarity)

        elif method == 'euclidean':
            # Euclidean distance converted to similarity
            distance = euclidean_distances([vector1], [vector2])[0, 0]
            # Convert distance to similarity (0-1 scale)
            max_distance = np.sqrt(len(vector1))  # Maximum possible distance
            similarity = 1.0 - (distance / max_distance)
            return float(max(0.0, similarity))

        elif method == 'weighted':
            # Weighted similarity using feature category weights
            return self._calculate_weighted_similarity(vector1, vector2)

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _calculate_weighted_similarity(
        self, vector1: np.ndarray, vector2: np.ndarray
    ) -> float:
        """Calculate weighted similarity based on feature categories.

        Args:
            vector1: First feature vector
            vector2: Second feature vector

        Returns:
            Weighted similarity score
        """
        # Define feature ranges for each category
        # These indices correspond to the order in extract_feature_vector
        feature_ranges = {
            'basic': (0, 3),  # BPM, key, duration
            'harmonic': (3, 11),  # Harmonic features
            'rhythmic': (11, 19),  # Rhythmic features
            'timbral': (19, 41),  # Timbral features
            'structural': (41, 51),  # Structural features
            'melodic': (51, 65),  # Melodic features
            'dynamic': (65, 77),  # Dynamic features
        }

        # Map feature categories to weights
        category_weights = {
            'basic': 0.05,
            'harmonic': self.feature_weights['harmonic'],
            'rhythmic': self.feature_weights['rhythmic'],
            'timbral': self.feature_weights['timbral'],
            'structural': self.feature_weights['structural'],
            'melodic': self.feature_weights['melodic'],
            'dynamic': self.feature_weights['dynamic'],
        }

        weighted_similarity = 0.0

        for category, (start, end) in feature_ranges.items():
            if end <= len(vector1) and end <= len(vector2):
                # Extract category features
                cat_vec1 = vector1[start:end]
                cat_vec2 = vector2[start:end]

                # Calculate cosine similarity for this category
                if np.linalg.norm(cat_vec1) > 0 and np.linalg.norm(cat_vec2) > 0:
                    cat_similarity = cosine_similarity([cat_vec1], [cat_vec2])[0, 0]
                else:
                    cat_similarity = 1.0 if np.allclose(cat_vec1, cat_vec2) else 0.0

                # Add weighted contribution
                weight = category_weights.get(category, 0.0)
                weighted_similarity += cat_similarity * weight

        return float(weighted_similarity)

    def find_similar_tracks(
        self,
        target_vector: np.ndarray,
        database_vectors: List[Tuple[str, np.ndarray]],
        top_k: int = 10,
        method: str = 'weighted',
    ) -> List[Tuple[str, float]]:
        """Find most similar tracks from a database.

        Args:
            target_vector: Target feature vector
            database_vectors: List of (track_id, feature_vector) tuples
            top_k: Number of similar tracks to return
            method: Similarity calculation method

        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for track_id, db_vector in database_vectors:
            try:
                similarity = self.calculate_similarity(target_vector, db_vector, method)
                similarities.append((track_id, similarity))
            except Exception as e:
                print(f"Error calculating similarity for {track_id}: {e}")
                continue

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def generate_similarity_matrix(
        self, feature_vectors: List[np.ndarray], method: str = 'cosine'
    ) -> np.ndarray:
        """Generate similarity matrix for a collection of feature vectors.

        Args:
            feature_vectors: List of feature vectors
            method: Similarity calculation method

        Returns:
            Similarity matrix
        """
        n = len(feature_vectors)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.calculate_similarity(
                        feature_vectors[i], feature_vectors[j], method
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric

        return similarity_matrix

    def reduce_dimensionality(
        self, feature_vectors: List[np.ndarray], n_components: int = 50
    ) -> Tuple[np.ndarray, PCA]:
        """Reduce dimensionality of feature vectors using PCA.

        Args:
            feature_vectors: List of feature vectors
            n_components: Number of components to keep

        Returns:
            (reduced_vectors, fitted_pca_model)
        """
        if not feature_vectors:
            return np.array([]), None

        # Stack feature vectors
        features_matrix = np.vstack(feature_vectors)

        # Apply PCA
        pca = PCA(n_components=min(n_components, features_matrix.shape[1]))
        reduced_features = pca.fit_transform(features_matrix)

        return reduced_features, pca

    def export_feature_vector(
        self, feature_vector: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export feature vector with metadata for storage.

        Args:
            feature_vector: Feature vector
            metadata: Additional metadata

        Returns:
            Exportable dictionary
        """
        return {
            'feature_vector': feature_vector.tolist(),
            'feature_weights': self.feature_weights,
            'metadata': metadata,
            'vector_length': len(feature_vector),
            'version': '1.0',
        }

    def import_feature_vector(
        self, exported_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Import feature vector from exported data.

        Args:
            exported_data: Exported feature vector data

        Returns:
            (feature_vector, metadata)
        """
        feature_vector = np.array(exported_data['feature_vector'], dtype=np.float32)
        metadata = exported_data.get('metadata', {})

        return feature_vector, metadata
