"""Boundary detection module for musical structure analysis."""

import numpy as np
import librosa
from typing import List, Dict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.decomposition import PCA


class BoundaryDetector:
    """Detects structural boundaries in musical audio."""
    
    def __init__(self, hop_length: int = 512):
        """Initialize boundary detector.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length
    
    def extract_structural_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract essential features for reliable structural analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of feature matrices
        """
        features = {}
        
        # MFCC features for timbral similarity (most important)
        features['mfcc'] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Chroma features for harmonic content (essential for structure)
        features['chroma'] = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # RMS energy for dynamics (important for section changes)
        features['rms'] = librosa.feature.rms(
            y=y, hop_length=self.hop_length
        )
        
        # Spectral centroid for brightness changes
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        return features
    
    def compute_self_similarity_matrix(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute simple and reliable self-similarity matrix using cosine similarity.
        
        Args:
            features: Dictionary of feature matrices
            
        Returns:
            Self-similarity matrix
        """
        # Combine key features only (MFCC, Chroma, RMS for simplicity)
        key_features = []
        
        # Use only the most important features
        for feature_name in ['mfcc', 'chroma', 'rms']:
            if feature_name in features:
                # Normalize features
                normalized = librosa.util.normalize(features[feature_name], axis=0)
                key_features.append(normalized)
        
        # If key features not available, use all features
        if not key_features:
            for feature_matrix in features.values():
                normalized = librosa.util.normalize(feature_matrix, axis=0)
                key_features.append(normalized)
        
        # Concatenate features
        all_features = np.vstack(key_features)
        
        # Transpose for similarity computation (frames x features)
        features_transposed = all_features.T
        
        # Compute cosine similarity matrix (simpler and more reliable)
        similarity_matrix = cosine_similarity(features_transposed)
        
        # Ensure values are in [0, 1] range
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        return similarity_matrix
    
    def detect_boundaries(self, similarity_matrix: np.ndarray,
                         sr: int, min_segment_length: float = 6.0, bpm: float = 130.0) -> List[int]:
        """Detect structural boundaries using multi-scale novelty detection.
        
        Args:
            similarity_matrix: Self-similarity matrix
            sr: Sample rate
            min_segment_length: Minimum segment length in seconds
            bpm: BPM for beat alignment
            
        Returns:
            List of boundary frame indices
        """
        n_frames = similarity_matrix.shape[0]
        
        # ==== ① multi-scale novelty =====
        nov_4  = self._compute_simple_novelty(similarity_matrix, ksize=8)   # ≒2bars
        nov_8  = self._compute_simple_novelty(similarity_matrix, ksize=16)  # ≒4bars
        novelty = librosa.util.normalize(nov_4 + 0.7 * nov_8)
        
        # allow 2-bar resolution but keep sanity
        min_frames = int(min_segment_length * sr / self.hop_length)
        min_frames = max(2, min(min_frames, n_frames // 6))  # Ensure reasonable bounds
        
        # Use lower threshold for better detection
        novelty_sorted = np.sort(novelty)
        # Start with 70th percentile
        height_threshold = novelty_sorted[int(len(novelty_sorted) * 0.7)]
        
        # Find peaks with constraints
        peaks, _ = find_peaks(novelty, distance=min_frames, height=height_threshold)
        
        # If too few boundaries, progressively relax constraints
        if len(peaks) < 3:
            height_threshold = novelty_sorted[int(len(novelty_sorted) * 0.6)]
            peaks, _ = find_peaks(novelty, distance=min_frames//2, height=height_threshold)
        
        if len(peaks) < 3:
            height_threshold = novelty_sorted[int(len(novelty_sorted) * 0.5)]
            peaks, _ = find_peaks(novelty, distance=min_frames//3, height=height_threshold)
        
        # snap to 4-beat grid first
        bar_frames = int((4 * 60 / bpm) * sr / self.hop_length)
        peaks = [round(p / bar_frames) * bar_frames for p in peaks]

        boundaries = [0] + peaks + [n_frames - 1]
        boundaries = sorted(list(set(boundaries)))
        
        # 8 bars (=2 bar *4) 以上離れているかでフィルタ
        min_sep = int((8 * 60 / bpm) * sr / self.hop_length)
        
        # Ensure minimum separation
        filtered_boundaries = [boundaries[0]]
        
        for i in range(1, len(boundaries)):
            if boundaries[i] - filtered_boundaries[-1] >= min_sep:
                filtered_boundaries.append(boundaries[i])
        
        return filtered_boundaries
    
    def _compute_beat_synchronized_novelty(self, similarity_matrix: np.ndarray,
                                         sr: int, bpm: float) -> np.ndarray:
        """Compute beat-synchronized novelty function using Foote kernel.
        
        Args:
            similarity_matrix: Self-similarity matrix
            sr: Sample rate
            bpm: BPM for beat synchronization
            
        Returns:
            Beat-synchronized novelty function
        """
        n_frames = similarity_matrix.shape[0]
        novelty = np.zeros(n_frames)
        
        # Calculate 1/8 note resolution for beat synchronization
        eighth_note_frames = int((60.0 / (bpm * 2)) * sr / self.hop_length)
        kernel_size = max(8, min(eighth_note_frames * 4, n_frames // 8))  # 4 eighth notes = half beat
        
        # Create Foote kernel for boundary detection
        foote_kernel = self._create_foote_kernel(kernel_size)
        
        for i in range(kernel_size, n_frames - kernel_size):
            # Extract local similarity matrix
            local_sim = similarity_matrix[i-kernel_size:i+kernel_size, i-kernel_size:i+kernel_size]
            
            # Apply Foote kernel convolution
            novelty[i] = np.sum(local_sim * foote_kernel)
        
        # Log normalization for stable peak thresholding across tracks
        novelty = np.log1p(np.maximum(novelty, 0))
        
        # Smooth and normalize novelty function
        novelty = librosa.util.normalize(novelty)
        
        return novelty
    
    def _create_foote_kernel(self, size: int) -> np.ndarray:
        """Create Foote kernel for boundary detection.
        
        Args:
            size: Kernel size
            
        Returns:
            Foote kernel matrix
        """
        kernel = np.zeros((2 * size, 2 * size))
        
        # Create checkerboard pattern for boundary detection
        kernel[:size, :size] = 1.0  # Top-left quadrant
        kernel[size:, size:] = 1.0  # Bottom-right quadrant
        kernel[:size, size:] = -1.0  # Top-right quadrant
        kernel[size:, :size] = -1.0  # Bottom-left quadrant
        
        # Normalize kernel
        kernel = kernel / np.sum(np.abs(kernel))
        
        return kernel
    
    def _align_to_adaptive_grid(self, novelty: np.ndarray, primary_grid: int, secondary_grid: int) -> np.ndarray:
        """Align novelty peaks to adaptive grid (2-bar or 4-bar) using Hough transform approach.
        
        Args:
            novelty: Original novelty function
            primary_grid: Primary grid size in frames (e.g., 2-bar or 4-bar)
            secondary_grid: Secondary grid size in frames (e.g., 4-bar or 8-bar)
            
        Returns:
            Grid-aligned novelty function
        """
        aligned_novelty = np.zeros_like(novelty)
        
        # First pass: align to primary grid (finer granularity)
        for i in range(0, len(novelty), primary_grid // 2):  # Check every half-grid
            if i + primary_grid < len(novelty):
                # Find the maximum novelty within ±half-grid of primary grid
                search_start = max(0, i - primary_grid // 4)
                search_end = min(len(novelty), i + primary_grid // 4)
                
                if search_end > search_start:
                    local_max_idx = search_start + np.argmax(novelty[search_start:search_end])
                    
                    # Snap to nearest primary grid boundary
                    grid_position = round(local_max_idx / primary_grid) * primary_grid
                    if 0 <= grid_position < len(aligned_novelty):
                        aligned_novelty[grid_position] = max(aligned_novelty[grid_position], novelty[local_max_idx])
        
        # Second pass: enhance with secondary grid for major boundaries
        for i in range(0, len(novelty), secondary_grid // 2):  # Check every half-secondary-grid
            if i + secondary_grid < len(novelty):
                # Find the maximum novelty within ±quarter-grid of secondary grid
                search_start = max(0, i - secondary_grid // 8)
                search_end = min(len(novelty), i + secondary_grid // 8)
                
                if search_end > search_start:
                    local_max_idx = search_start + np.argmax(novelty[search_start:search_end])
                    
                    # Snap to nearest secondary grid boundary
                    grid_position = round(local_max_idx / secondary_grid) * secondary_grid
                    if 0 <= grid_position < len(aligned_novelty):
                        # Boost secondary grid positions for major boundaries
                        aligned_novelty[grid_position] = max(aligned_novelty[grid_position], novelty[local_max_idx] * 1.2)
        
        return aligned_novelty
    
    def _align_to_8bar_grid(self, novelty: np.ndarray, bar_8_frames: int) -> np.ndarray:
        """Align novelty peaks to 8-bar grid using Hough transform approach (legacy method).
        
        Args:
            novelty: Original novelty function
            bar_8_frames: Number of frames in 8 bars
            
        Returns:
            Grid-aligned novelty function
        """
        aligned_novelty = np.zeros_like(novelty)
        
        # Find potential 8-bar boundaries
        for i in range(0, len(novelty), bar_8_frames // 4):  # Check every 2 bars
            if i + bar_8_frames < len(novelty):
                # Find the maximum novelty within ±1 bar of 8-bar grid
                search_start = max(0, i - bar_8_frames // 8)
                search_end = min(len(novelty), i + bar_8_frames // 8)
                
                if search_end > search_start:
                    local_max_idx = search_start + np.argmax(novelty[search_start:search_end])
                    
                    # Snap to nearest 8-bar boundary
                    grid_position = round(local_max_idx / bar_8_frames) * bar_8_frames
                    if 0 <= grid_position < len(aligned_novelty):
                        aligned_novelty[grid_position] = novelty[local_max_idx]
        
        return aligned_novelty
    
    def _compute_simple_novelty(self, similarity_matrix: np.ndarray, ksize: int = 12) -> np.ndarray:
        """Compute simple and reliable novelty function from similarity matrix.
        
        Args:
            similarity_matrix: Self-similarity matrix
            ksize: Kernel size for novelty computation
            
        Returns:
            Novelty function
        """
        n_frames = similarity_matrix.shape[0]
        novelty = np.zeros(n_frames)
        
        kernel_size = ksize
        
        for i in range(kernel_size, n_frames - kernel_size):
            # Compare similarity before and after current frame
            before = similarity_matrix[i-kernel_size:i, i-kernel_size:i]
            after = similarity_matrix[i:i+kernel_size, i:i+kernel_size]
            
            # Calculate novelty as difference in mean similarity
            before_mean = np.mean(before)
            after_mean = np.mean(after)
            novelty[i] = abs(before_mean - after_mean)
        
        # Apply Gaussian smoothing to reduce noise
        novelty = gaussian_filter1d(novelty, sigma=2.0)
        
        # Normalize to [0, 1] range
        if np.max(novelty) > 0:
            novelty = novelty / np.max(novelty)
        
        return novelty
    
    def _compute_novelty(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute novelty function from similarity matrix (legacy method).
        
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
            
            # Calculate novelty as absolute difference in self-similarity (improved)
            novelty[i] = np.mean(np.abs(after - before))
        
        # Smooth novelty function
        novelty = librosa.util.normalize(novelty)
        
        return novelty
    
    def snap_to_beat(self, boundaries: List[int], sr: int, bpm: float) -> List[int]:
        """Snap boundaries to 4-beat (1 bar) grid for musical alignment.
        
        Args:
            boundaries: List of boundary frame indices
            sr: Sample rate
            bpm: BPM for beat grid calculation
            
        Returns:
            List of bar-aligned boundary frame indices
        """
        # Calculate frames per bar (4 beats)
        bar_hop = int((4 * 60 / bpm) * sr / self.hop_length)  # 4 beats = 1 bar
        
        snapped = [0]
        for b in boundaries[1:-1]:
            # Snap to nearest bar boundary
            snapped_boundary = round(b / bar_hop) * bar_hop
            snapped.append(snapped_boundary)
        snapped.append(boundaries[-1])
        
        # Remove duplicates and ensure minimum bar separation
        filtered = [snapped[0]]
        for i in range(1, len(snapped)):
            if snapped[i] - filtered[-1] >= bar_hop:  # At least 1 bar apart
                filtered.append(snapped[i])
        
        return filtered
    
    def detect_repetitions(self, similarity_matrix: np.ndarray, 
                          sr: int) -> List[Dict[str, float]]:
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
    
    def _remove_overlapping_repetitions(self, repetitions: List[Dict[str, float]]) -> List[Dict[str, float]]:
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
    
    def _repetitions_overlap(self, rep1: Dict[str, float], rep2: Dict[str, float]) -> bool:
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