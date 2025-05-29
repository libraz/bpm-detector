"""Dynamics and energy analysis module."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy


class DynamicsAnalyzer:
    """Analyzes dynamics, energy, and loudness characteristics."""
    
    def __init__(self, hop_length: int = 512, frame_size: int = 2048):
        """Initialize dynamics analyzer.
        
        Args:
            hop_length: Hop length for analysis
            frame_size: Frame size for analysis
        """
        self.hop_length = hop_length
        self.frame_size = frame_size
        
    def extract_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract energy-related features.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of energy features
        """
        features = {}
        
        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=y, hop_length=self.hop_length, frame_length=self.frame_size
        )[0]
        
        # Spectral energy
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.frame_size)
        features['spectral_energy'] = np.sum(np.abs(stft)**2, axis=0)
        
        # Energy in different frequency bands
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_size)
        
        # Low frequency energy (20-250 Hz)
        low_freq_mask = (freqs >= 20) & (freqs <= 250)
        features['low_freq_energy'] = np.sum(np.abs(stft[low_freq_mask, :])**2, axis=0)
        
        # Mid frequency energy (250-4000 Hz)
        mid_freq_mask = (freqs >= 250) & (freqs <= 4000)
        features['mid_freq_energy'] = np.sum(np.abs(stft[mid_freq_mask, :])**2, axis=0)
        
        # High frequency energy (4000+ Hz)
        high_freq_mask = freqs >= 4000
        features['high_freq_energy'] = np.sum(np.abs(stft[high_freq_mask, :])**2, axis=0)
        
        # Onset strength
        features['onset_strength'] = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Zero crossing rate (related to energy distribution)
        features['zcr'] = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )[0]
        
        return features
    
    def calculate_dynamic_range(self, rms: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic range characteristics.
        
        Args:
            rms: RMS energy values
            
        Returns:
            Dictionary of dynamic range measures
        """
        if len(rms) == 0:
            return {
                'dynamic_range_db': 0.0,
                'peak_to_average_ratio': 0.0,
                'crest_factor': 0.0,
                'dynamic_variance': 0.0
            }
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        
        # Dynamic range (difference between max and min)
        dynamic_range_db = np.max(rms_db) - np.min(rms_db)
        
        # Peak to average ratio
        peak_level = np.max(rms)
        average_level = np.mean(rms)
        peak_to_average_ratio = peak_level / (average_level + 1e-8)
        
        # Crest factor (peak to RMS ratio)
        rms_of_rms = np.sqrt(np.mean(rms**2))
        crest_factor = peak_level / (rms_of_rms + 1e-8)
        
        # Dynamic variance
        dynamic_variance = np.var(rms_db)
        
        return {
            'dynamic_range_db': float(dynamic_range_db),
            'peak_to_average_ratio': float(peak_to_average_ratio),
            'crest_factor': float(crest_factor),
            'dynamic_variance': float(dynamic_variance)
        }
    
    def analyze_loudness(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze loudness characteristics.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of loudness measures
        """
        # Calculate RMS for loudness estimation
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        if len(rms) == 0:
            return {
                'average_loudness_db': -60.0,
                'peak_loudness_db': -60.0,
                'loudness_range_db': 0.0,
                'perceived_loudness': 0.0
            }
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        
        # Average loudness
        average_loudness_db = np.mean(rms_db)
        
        # Peak loudness
        peak_loudness_db = np.max(rms_db)
        
        # Loudness range (similar to dynamic range but for loudness)
        loudness_range_db = np.max(rms_db) - np.min(rms_db)
        
        # Perceived loudness (A-weighted approximation)
        perceived_loudness = self._calculate_perceived_loudness(y, sr)
        
        return {
            'average_loudness_db': float(average_loudness_db),
            'peak_loudness_db': float(peak_loudness_db),
            'loudness_range_db': float(loudness_range_db),
            'perceived_loudness': float(perceived_loudness)
        }
    
    def _calculate_perceived_loudness(self, y: np.ndarray, sr: int) -> float:
        """Calculate perceived loudness using A-weighting approximation.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Perceived loudness score (0-1)
        """
        # Simple A-weighting approximation
        # This is a simplified version - full A-weighting requires more complex filtering
        
        # Calculate power spectral density
        stft = librosa.stft(y, hop_length=self.hop_length)
        power_spectrum = np.mean(np.abs(stft)**2, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
        
        # Simple A-weighting approximation (emphasizes mid frequencies)
        a_weights = np.ones_like(freqs)
        
        # Reduce low frequencies
        low_freq_mask = freqs < 1000
        a_weights[low_freq_mask] *= (freqs[low_freq_mask] / 1000.0)**2
        
        # Reduce very high frequencies
        high_freq_mask = freqs > 8000
        a_weights[high_freq_mask] *= 0.5
        
        # Apply weighting
        weighted_power = power_spectrum * a_weights[:len(power_spectrum)]
        
        # Calculate perceived loudness
        perceived_loudness = np.sum(weighted_power)
        
        # Normalize to 0-1 scale (rough approximation)
        perceived_loudness = min(1.0, perceived_loudness / np.sum(power_spectrum))
        
        return perceived_loudness
    
    def generate_energy_profile(self, energy_features: Dict[str, np.ndarray], 
                               sr: int, window_size: float = 10.0) -> np.ndarray:
        """Generate energy profile over time.
        
        Args:
            energy_features: Dictionary of energy features
            sr: Sample rate
            window_size: Window size in seconds for averaging
            
        Returns:
            Energy profile array
        """
        rms = energy_features.get('rms', np.array([]))
        
        if len(rms) == 0:
            return np.array([])
        
        # Calculate window size in frames
        window_frames = int(window_size * sr / self.hop_length)
        
        if window_frames >= len(rms):
            return np.array([np.mean(rms)])
        
        # Generate profile by averaging over windows
        profile = []
        for i in range(0, len(rms), window_frames):
            window_end = min(i + window_frames, len(rms))
            window_energy = np.mean(rms[i:window_end])
            profile.append(window_energy)
        
        # Normalize to 0-1 scale
        profile = np.array(profile)
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
        
        return profile
    
    def detect_climax_points(self, energy_features: Dict[str, np.ndarray], 
                           sr: int) -> List[Dict[str, float]]:
        """Detect climax points in the audio.
        
        Args:
            energy_features: Dictionary of energy features
            sr: Sample rate
            
        Returns:
            List of climax points with time and intensity
        """
        rms = energy_features.get('rms', np.array([]))
        onset_strength = energy_features.get('onset_strength', np.array([]))
        
        if len(rms) == 0:
            return []
        
        # Smooth the energy signal
        if len(rms) > 10:
            smoothed_rms = savgol_filter(rms, min(11, len(rms)//2*2+1), 3)
        else:
            smoothed_rms = rms
        
        # Find peaks in energy
        peak_threshold = np.mean(smoothed_rms) + np.std(smoothed_rms)
        energy_peaks, _ = find_peaks(
            smoothed_rms, 
            height=peak_threshold,
            distance=int(5 * sr / self.hop_length)  # Minimum 5 seconds apart
        )
        
        climax_points = []
        
        for peak_idx in energy_peaks:
            # Calculate time
            time_seconds = peak_idx * self.hop_length / sr
            
            # Calculate intensity (normalized energy at peak)
            intensity = smoothed_rms[peak_idx] / (np.max(smoothed_rms) + 1e-8)
            
            # Add onset strength if available
            if len(onset_strength) > peak_idx:
                onset_intensity = onset_strength[peak_idx]
                # Combine energy and onset strength
                combined_intensity = (intensity + onset_intensity / np.max(onset_strength + 1e-8)) / 2.0
            else:
                combined_intensity = intensity
            
            climax_points.append({
                'time': float(time_seconds),
                'intensity': float(combined_intensity)
            })
        
        # Sort by intensity (highest first)
        climax_points.sort(key=lambda x: x['intensity'], reverse=True)
        
        return climax_points
    
    def analyze_tension_curve(self, energy_features: Dict[str, np.ndarray], 
                            sr: int) -> np.ndarray:
        """Analyze musical tension over time.
        
        Args:
            energy_features: Dictionary of energy features
            sr: Sample rate
            
        Returns:
            Tension curve array (0-1)
        """
        rms = energy_features.get('rms', np.array([]))
        spectral_energy = energy_features.get('spectral_energy', np.array([]))
        high_freq_energy = energy_features.get('high_freq_energy', np.array([]))
        
        if len(rms) == 0:
            return np.array([])
        
        # Normalize all features
        rms_norm = rms / (np.max(rms) + 1e-8)
        
        if len(spectral_energy) > 0:
            spectral_norm = spectral_energy / (np.max(spectral_energy) + 1e-8)
        else:
            spectral_norm = rms_norm
        
        if len(high_freq_energy) > 0:
            high_freq_norm = high_freq_energy / (np.max(high_freq_energy) + 1e-8)
        else:
            high_freq_norm = rms_norm
        
        # Combine features for tension calculation
        # Higher energy + higher spectral content + higher frequencies = more tension
        min_length = min(len(rms_norm), len(spectral_norm), len(high_freq_norm))
        
        tension_curve = (
            rms_norm[:min_length] * 0.4 +
            spectral_norm[:min_length] * 0.3 +
            high_freq_norm[:min_length] * 0.3
        )
        
        # Smooth the tension curve
        if len(tension_curve) > 10:
            tension_curve = savgol_filter(tension_curve, min(11, len(tension_curve)//2*2+1), 3)
        
        # Ensure values are in 0-1 range
        tension_curve = np.clip(tension_curve, 0, 1)
        
        return tension_curve
    
    def analyze_energy_distribution(self, energy_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze energy distribution across frequency bands.
        
        Args:
            energy_features: Dictionary of energy features
            
        Returns:
            Dictionary of energy distribution measures
        """
        low_freq = energy_features.get('low_freq_energy', np.array([]))
        mid_freq = energy_features.get('mid_freq_energy', np.array([]))
        high_freq = energy_features.get('high_freq_energy', np.array([]))
        
        if len(low_freq) == 0 or len(mid_freq) == 0 or len(high_freq) == 0:
            return {
                'low_freq_ratio': 0.0,
                'mid_freq_ratio': 0.0,
                'high_freq_ratio': 0.0,
                'spectral_balance': 0.0,
                'energy_entropy': 0.0
            }
        
        # Calculate total energy
        total_energy = low_freq + mid_freq + high_freq
        
        # Calculate ratios
        low_freq_ratio = np.mean(low_freq / (total_energy + 1e-8))
        mid_freq_ratio = np.mean(mid_freq / (total_energy + 1e-8))
        high_freq_ratio = np.mean(high_freq / (total_energy + 1e-8))
        
        # Calculate spectral balance (how evenly distributed energy is)
        energy_ratios = np.array([low_freq_ratio, mid_freq_ratio, high_freq_ratio])
        energy_ratios = energy_ratios / (np.sum(energy_ratios) + 1e-8)
        
        # Entropy of energy distribution
        energy_entropy = entropy(energy_ratios + 1e-8)
        max_entropy = np.log(3)  # Maximum entropy for 3 bands
        normalized_entropy = energy_entropy / max_entropy if max_entropy > 0 else 0
        
        # Spectral balance (higher entropy = more balanced)
        spectral_balance = normalized_entropy
        
        return {
            'low_freq_ratio': float(low_freq_ratio),
            'mid_freq_ratio': float(mid_freq_ratio),
            'high_freq_ratio': float(high_freq_ratio),
            'spectral_balance': float(spectral_balance),
            'energy_entropy': float(normalized_entropy)
        }
    
    def detect_dynamic_events(self, energy_features: Dict[str, np.ndarray], 
                            sr: int) -> List[Dict[str, Any]]:
        """Detect significant dynamic events (drops, builds, etc.).
        
        Args:
            energy_features: Dictionary of energy features
            sr: Sample rate
            
        Returns:
            List of dynamic events
        """
        rms = energy_features.get('rms', np.array([]))
        
        if len(rms) < 20:  # Need sufficient data
            return []
        
        # Smooth the signal
        smoothed_rms = savgol_filter(rms, min(11, len(rms)//2*2+1), 3)
        
        # Calculate derivative to find rapid changes
        rms_diff = np.diff(smoothed_rms)
        
        events = []
        
        # Detect drops (sudden decreases)
        drop_threshold = -np.std(rms_diff) * 2
        drop_indices = np.where(rms_diff < drop_threshold)[0]
        
        for idx in drop_indices:
            time_seconds = idx * self.hop_length / sr
            magnitude = abs(rms_diff[idx]) / (np.std(rms_diff) + 1e-8)
            
            events.append({
                'type': 'drop',
                'time': float(time_seconds),
                'magnitude': float(magnitude)
            })
        
        # Detect builds (sudden increases)
        build_threshold = np.std(rms_diff) * 2
        build_indices = np.where(rms_diff > build_threshold)[0]
        
        for idx in build_indices:
            time_seconds = idx * self.hop_length / sr
            magnitude = rms_diff[idx] / (np.std(rms_diff) + 1e-8)
            
            events.append({
                'type': 'build',
                'time': float(time_seconds),
                'magnitude': float(magnitude)
            })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        return events
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete dynamics analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Complete dynamics analysis results
        """
        # Extract energy features
        energy_features = self.extract_energy_features(y, sr)
        
        # Calculate dynamic range
        dynamic_range = self.calculate_dynamic_range(energy_features['rms'])
        
        # Analyze loudness
        loudness = self.analyze_loudness(y, sr)
        
        # Generate energy profile
        energy_profile = self.generate_energy_profile(energy_features, sr)
        
        # Detect climax points
        climax_points = self.detect_climax_points(energy_features, sr)
        
        # Analyze tension curve
        tension_curve = self.analyze_tension_curve(energy_features, sr)
        
        # Analyze energy distribution
        energy_distribution = self.analyze_energy_distribution(energy_features)
        
        # Detect dynamic events
        dynamic_events = self.detect_dynamic_events(energy_features, sr)
        
        return {
            'dynamic_range': dynamic_range,
            'loudness': loudness,
            'energy_profile': energy_profile.tolist() if len(energy_profile) > 0 else [],
            'climax_points': climax_points,
            'tension_curve': tension_curve.tolist() if len(tension_curve) > 0 else [],
            'energy_distribution': energy_distribution,
            'dynamic_events': dynamic_events,
            'overall_energy': float(np.mean(energy_features['rms'])) if len(energy_features['rms']) > 0 else 0.0,
            'energy_variance': float(np.var(energy_features['rms'])) if len(energy_features['rms']) > 0 else 0.0
        }