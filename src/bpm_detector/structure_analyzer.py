"""Musical structure analysis module."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class StructureAnalyzer:
    """Analyzes musical structure and form."""
    
    # Common section types and their characteristics
    SECTION_TYPES = {
        'intro': {'energy_range': (0.1, 0.4), 'complexity_range': (0.1, 0.5)},
        'verse': {'energy_range': (0.3, 0.7), 'complexity_range': (0.4, 0.8)},
        'chorus': {'energy_range': (0.6, 1.0), 'complexity_range': (0.5, 0.9)},
        'bridge': {'energy_range': (0.4, 0.8), 'complexity_range': (0.6, 1.0)},
        'outro': {'energy_range': (0.1, 0.5), 'complexity_range': (0.2, 0.6)},
        'instrumental': {'energy_range': (0.3, 0.9), 'complexity_range': (0.3, 0.8)}
    }
    
    def __init__(self, hop_length: int = 512, frame_size: int = 4096):
        """Initialize structure analyzer.
        
        Args:
            hop_length: Hop length for analysis
            frame_size: Frame size for feature extraction
        """
        self.hop_length = hop_length
        self.frame_size = frame_size
        
    def extract_structural_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract features for structural analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of feature matrices
        """
        features = {}
        
        # MFCC features for timbral similarity
        features['mfcc'] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Chroma features for harmonic content
        features['chroma'] = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Spectral centroid for brightness
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # RMS energy for dynamics
        features['rms'] = librosa.feature.rms(
            y=y, hop_length=self.hop_length
        )
        
        # Zero crossing rate for texture
        features['zcr'] = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )
        
        # Tempo features
        onset_envelope = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        features['onset_strength'] = onset_envelope.reshape(1, -1)
        
        return features
    
    def compute_self_similarity_matrix(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute self-similarity matrix from features.
        
        Args:
            features: Dictionary of feature matrices
            
        Returns:
            Self-similarity matrix
        """
        # Combine all features
        combined_features = []
        
        for feature_name, feature_matrix in features.items():
            # Normalize features
            normalized = librosa.util.normalize(feature_matrix, axis=0)
            combined_features.append(normalized)
        
        # Concatenate all features
        all_features = np.vstack(combined_features)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(all_features.T)
        
        # Clip values to [0, 1] range to handle numerical issues
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        return similarity_matrix
    
    def detect_boundaries(self, similarity_matrix: np.ndarray,
                         sr: int, min_segment_length: float = 15.0) -> List[int]:
        """Detect structural boundaries using novelty detection.
        
        Args:
            similarity_matrix: Self-similarity matrix
            sr: Sample rate
            min_segment_length: Minimum segment length in seconds
            
        Returns:
            List of boundary frame indices
        """
        # Compute novelty function
        novelty = self._compute_novelty(similarity_matrix)
        
        # Find peaks in novelty function with stricter criteria
        min_frames = int(min_segment_length * sr / self.hop_length)
        height_threshold = np.mean(novelty) + 1.5 * np.std(novelty)  # Stricter threshold
        peaks, _ = find_peaks(novelty, distance=min_frames, height=height_threshold)
        
        # Add start and end boundaries
        boundaries = [0] + peaks.tolist() + [similarity_matrix.shape[0] - 1]
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _compute_novelty(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute novelty function from similarity matrix.
        
        Args:
            similarity_matrix: Self-similarity matrix
            
        Returns:
            Novelty function
        """
        n_frames = similarity_matrix.shape[0]
        novelty = np.zeros(n_frames)
        
        # Kernel for novelty detection
        kernel_size = min(16, n_frames // 4)
        
        for i in range(kernel_size, n_frames - kernel_size):
            # Compare similarity before and after current frame
            before = similarity_matrix[i-kernel_size:i, i-kernel_size:i]
            after = similarity_matrix[i:i+kernel_size, i:i+kernel_size]
            
            # Calculate novelty as difference in self-similarity
            novelty[i] = np.mean(after) - np.mean(before)
        
        # Smooth novelty function
        novelty = librosa.util.normalize(novelty)
        
        return novelty
    
    def classify_sections(self, y: np.ndarray, sr: int, 
                         boundaries: List[int]) -> List[Dict[str, Any]]:
        """Classify sections based on their characteristics.
        
        Args:
            y: Audio signal
            sr: Sample rate
            boundaries: List of boundary frame indices
            
        Returns:
            List of classified sections
        """
        sections = []
        
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            
            # Convert to time
            start_time = start_frame * self.hop_length / sr
            end_time = end_frame * self.hop_length / sr
            duration = end_time - start_time
            
            # Extract segment audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]
            
            if len(segment) == 0:
                continue
            
            # Analyze segment characteristics
            characteristics = self._analyze_segment_characteristics(segment, sr)
            
            # Classify section type
            section_type = self._classify_section_type(characteristics, start_time, end_time)
            
            sections.append({
                'type': section_type,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'characteristics': characteristics['labels'],
                'energy_level': characteristics['energy'],
                'complexity': characteristics['complexity']
            })
        
        return sections
    
    def _analyze_segment_characteristics(self, segment: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze characteristics of a segment.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            Dictionary of segment characteristics
        """
        if len(segment) == 0:
            return {
                'energy': 0.0,
                'complexity': 0.0,
                'brightness': 0.0,
                'labels': []
            }
        
        # Energy analysis
        rms_energy = np.sqrt(np.mean(segment**2))
        energy_level = min(1.0, rms_energy * 10)  # Normalize roughly to 0-1
        
        # Spectral complexity
        stft = librosa.stft(segment, hop_length=self.hop_length)
        spectral_complexity = np.std(np.abs(stft)) / (np.mean(np.abs(stft)) + 1e-8)
        spectral_complexity = min(1.0, spectral_complexity)
        
        # Brightness (spectral centroid)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        brightness = np.mean(spectral_centroid) / (sr / 2)  # Normalize to 0-1
        
        # Determine characteristics labels
        labels = []
        
        if energy_level < 0.3:
            labels.append('low_energy')
        elif energy_level > 0.7:
            labels.append('high_energy')
        else:
            labels.append('mid_energy')
        
        if spectral_complexity < 0.3:
            labels.append('simple')
        elif spectral_complexity > 0.7:
            labels.append('complex')
        else:
            labels.append('moderate_complexity')
        
        if brightness < 0.3:
            labels.append('dark')
        elif brightness > 0.7:
            labels.append('bright')
        
        # Check for vocal presence (simplified heuristic)
        if self._detect_vocal_presence(segment, sr):
            labels.append('vocal_present')
        else:
            labels.append('instrumental')
        
        return {
            'energy': energy_level,
            'complexity': spectral_complexity,
            'brightness': brightness,
            'labels': labels
        }
    
    def _detect_vocal_presence(self, segment: np.ndarray, sr: int) -> bool:
        """Simple vocal presence detection.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            True if vocals are likely present
        """
        # Use harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(segment)
        
        # Check energy in vocal frequency range (roughly 80-1000 Hz)
        stft = librosa.stft(harmonic)
        freqs = librosa.fft_frequencies(sr=sr)
        
        vocal_range_mask = (freqs >= 80) & (freqs <= 1000)
        vocal_energy = np.mean(np.abs(stft[vocal_range_mask, :]))
        total_energy = np.mean(np.abs(stft))
        
        # Simple threshold-based detection
        vocal_ratio = vocal_energy / (total_energy + 1e-8)
        
        return bool(vocal_ratio > 0.3)
    
    def _classify_section_type(self, characteristics: Dict[str, Any],
                              start_time: float, end_time: float) -> str:
        """Classify section type based on characteristics and position.
        
        Args:
            characteristics: Segment characteristics
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Section type string
        """
        energy = characteristics['energy']
        complexity = characteristics['complexity']
        labels = characteristics['labels']
        
        # Position-based heuristics (more aggressive)
        if start_time < 30:  # First 30 seconds likely intro/verse
            if energy < 0.5:
                return 'intro'
            else:
                return 'verse'
        
        if end_time > 200:  # Last part of long songs
            if energy < 0.4:
                return 'outro'
        
        # Create more variation in classification
        section_index = int(start_time / 30)  # Rough section index
        
        # Alternate between verse and chorus for variety
        if section_index % 3 == 0:  # Every 3rd section
            if energy > 0.4:
                return 'chorus'
            else:
                return 'verse'
        elif section_index % 3 == 1:
            if 'instrumental' in labels or complexity > 0.6:
                return 'bridge'
            else:
                return 'verse'
        else:  # section_index % 3 == 2
            if energy > 0.5:
                return 'chorus'
            else:
                return 'verse'
    
    def analyze_form(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall musical form.
        
        Args:
            sections: List of classified sections
            
        Returns:
            Form analysis results
        """
        if not sections:
            return {
                'form': '',
                'repetition_ratio': 0.0,
                'structural_complexity': 0.0,
                'section_count': 0
            }
        
        # Extract section types
        section_types = [section['type'] for section in sections]
        
        # Create form string
        form = ''.join([self._section_to_letter(stype) for stype in section_types])
        
        # Calculate repetition ratio
        unique_sections = len(set(section_types))
        total_sections = len(section_types)
        repetition_ratio = 1.0 - (unique_sections / total_sections) if total_sections > 0 else 0.0
        
        # Calculate structural complexity
        structural_complexity = self._calculate_structural_complexity(sections)
        
        return {
            'form': form,
            'repetition_ratio': repetition_ratio,
            'structural_complexity': structural_complexity,
            'section_count': total_sections,
            'unique_sections': unique_sections,
            'section_types': section_types
        }
    
    def _section_to_letter(self, section_type: str) -> str:
        """Convert section type to letter for form notation.
        
        Args:
            section_type: Section type string
            
        Returns:
            Single letter representing the section
        """
        mapping = {
            'intro': 'I',
            'verse': 'A',
            'chorus': 'B',
            'bridge': 'C',
            'instrumental': 'D',
            'outro': 'O'
        }
        return mapping.get(section_type, 'X')
    
    def _calculate_structural_complexity(self, sections: List[Dict[str, Any]]) -> float:
        """Calculate structural complexity score.
        
        Args:
            sections: List of sections
            
        Returns:
            Complexity score (0-1)
        """
        if not sections:
            return 0.0
        
        # Factors contributing to complexity:
        # 1. Number of different section types
        section_types = [s['type'] for s in sections]
        unique_types = len(set(section_types))
        type_complexity = min(1.0, unique_types / 6.0)  # Normalize by max expected types
        
        # 2. Variation in section durations
        durations = [s['duration'] for s in sections]
        duration_std = np.std(durations) / (np.mean(durations) + 1e-8)
        duration_complexity = min(1.0, duration_std)
        
        # 3. Non-standard form patterns
        form_complexity = 0.0
        if len(sections) > 8:  # Long form
            form_complexity += 0.3
        if unique_types > 4:  # Many section types
            form_complexity += 0.3
        
        # Combine factors
        overall_complexity = (type_complexity + duration_complexity + form_complexity) / 3.0
        
        return min(1.0, overall_complexity)
    
    def detect_repetitions(self, similarity_matrix: np.ndarray, 
                          sr: int) -> List[Dict[str, Any]]:
        """Detect repeated sections in the music.
        
        Args:
            similarity_matrix: Self-similarity matrix
            sr: Sample rate
            
        Returns:
            List of detected repetitions
        """
        repetitions = []
        n_frames = similarity_matrix.shape[0]
        
        # Look for diagonal patterns in similarity matrix (optimized)
        min_length = int(4 * sr / self.hop_length)  # Reduced to 4 seconds for speed
        stride = max(1, min_length // 8)  # Use stride to skip frames
        
        for i in range(0, n_frames - min_length, stride):
            for j in range(i + min_length, n_frames - min_length, stride):
                # Limit search range for performance
                if j - i > n_frames // 2:
                    break
                    
                # Check diagonal similarity
                max_len = min(n_frames - i, n_frames - j, min_length)
                
                if max_len < min_length:
                    continue
                
                # Sample fewer points for speed
                sample_step = max(1, max_len // 10)  # Sample 10 points max
                sample_indices = range(0, max_len, sample_step)
                
                # Calculate similarity along diagonal
                diagonal_sim = np.mean([
                    similarity_matrix[i + k, j + k]
                    for k in sample_indices
                    if i + k < n_frames and j + k < n_frames
                ])
                
                if diagonal_sim > 0.8:  # High similarity threshold
                    start_time_1 = i * self.hop_length / sr
                    start_time_2 = j * self.hop_length / sr
                    duration = min_length * self.hop_length / sr
                    
                    repetitions.append({
                        'first_occurrence': start_time_1,
                        'second_occurrence': start_time_2,
                        'duration': duration,
                        'similarity': diagonal_sim
                    })
        
        # Remove overlapping repetitions (keep highest similarity)
        repetitions = self._remove_overlapping_repetitions(repetitions)
        
        return repetitions
    
    def _remove_overlapping_repetitions(self, repetitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping repetition detections.
        
        Args:
            repetitions: List of repetition detections
            
        Returns:
            Filtered list without overlaps
        """
        if not repetitions:
            return repetitions
        
        # Sort by similarity (highest first)
        repetitions.sort(key=lambda x: x['similarity'], reverse=True)
        
        filtered = []
        
        for rep in repetitions:
            # Check if this repetition overlaps with any already accepted
            overlaps = False
            
            for accepted in filtered:
                if self._repetitions_overlap(rep, accepted):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(rep)
        
        return filtered
    
    def _repetitions_overlap(self, rep1: Dict[str, Any], rep2: Dict[str, Any]) -> bool:
        """Check if two repetitions overlap.
        
        Args:
            rep1: First repetition
            rep2: Second repetition
            
        Returns:
            True if repetitions overlap
        """
        # Check overlap in either occurrence
        for occ1 in ['first_occurrence', 'second_occurrence']:
            for occ2 in ['first_occurrence', 'second_occurrence']:
                start1, end1 = rep1[occ1], rep1[occ1] + rep1['duration']
                start2, end2 = rep2[occ2], rep2[occ2] + rep2['duration']
                
                if not (end1 <= start2 or end2 <= start1):  # Overlap condition
                    return True
        
        return False
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete structural analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Complete structural analysis results
        """
        # Extract features
        features = self.extract_structural_features(y, sr)
        
        # Compute self-similarity matrix
        similarity_matrix = self.compute_self_similarity_matrix(features)
        
        # Detect boundaries
        boundaries = self.detect_boundaries(similarity_matrix, sr)
        
        # Classify sections
        sections = self.classify_sections(y, sr, boundaries)
        
        # Analyze form
        form_analysis = self.analyze_form(sections)
        
        # Detect repetitions
        repetitions = self.detect_repetitions(similarity_matrix, sr)
        
        return {
            'sections': sections,
            'form': form_analysis['form'],
            'repetition_ratio': form_analysis['repetition_ratio'],
            'structural_complexity': form_analysis['structural_complexity'],
            'section_count': form_analysis['section_count'],
            'unique_sections': form_analysis['unique_sections'],
            'repetitions': repetitions,
            'boundaries': [b * self.hop_length / sr for b in boundaries]
        }