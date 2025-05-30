"""Audio effects detection module."""

import numpy as np
import librosa
from typing import Dict


class EffectsDetector:
    """Detects audio effects usage in signals."""
    
    def __init__(self, hop_length: int = 512):
        """Initialize effects detector.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length
    
    def analyze_effects_usage(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze usage of audio effects.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of effect usage scores
        """
        effects = {}
        
        # Detect individual effects
        effects['reverb'] = self._detect_reverb(y, sr)
        effects['distortion'] = self._detect_distortion(y, sr)
        effects['chorus'] = self._detect_chorus(y, sr)
        effects['compression'] = self._detect_compression(y)
        
        return effects
    
    def _detect_reverb(self, y: np.ndarray, sr: int) -> float:
        """Detect reverb presence.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Reverb amount (0-1)
        """
        # Calculate envelope decay characteristics
        envelope = np.abs(librosa.stft(y))
        envelope_mean = np.mean(envelope, axis=0)
        
        # Look for exponential decay patterns
        if len(envelope_mean) < 10:
            return 0.2  # Default low reverb amount
        
        # Calculate autocorrelation to find decay patterns
        autocorr = np.correlate(envelope_mean, envelope_mean, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for long-term correlations (indicating reverb tail)
        long_term_corr = np.mean(autocorr[len(autocorr)//4:len(autocorr)//2])
        short_term_corr = np.mean(autocorr[:len(autocorr)//8])
        
        if short_term_corr == 0:
            return 0.2  # Default low reverb amount
        
        reverb_ratio = long_term_corr / short_term_corr
        
        # Check for additional reverb indicators
        # Calculate energy decay rate
        energy_decay = np.diff(envelope_mean)
        decay_variance = np.var(energy_decay)
        
        # More variance in decay suggests reverb
        decay_factor = min(0.3, decay_variance * 10.0)
        
        # Combine ratio and decay analysis
        base_reverb = min(0.7, max(0.0, reverb_ratio * 1.5))
        total_reverb = base_reverb + decay_factor
        
        return min(0.95, total_reverb)
    
    def _detect_distortion(self, y: np.ndarray, sr: int) -> float:
        """Detect distortion presence.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Distortion amount (0-1)
        """
        # Calculate harmonic content
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Look for harmonic distortion (odd harmonics)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
        
        # Calculate total harmonic distortion approximation
        fundamental_energy = np.mean(magnitude[:len(magnitude)//4, :])
        harmonic_energy = np.mean(magnitude[len(magnitude)//4:, :])
        
        if fundamental_energy == 0:
            return 0.0
        
        distortion_ratio = harmonic_energy / fundamental_energy
        
        return min(1.0, distortion_ratio)
    
    def _detect_chorus(self, y: np.ndarray, sr: int) -> float:
        """Detect chorus/modulation effects.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Chorus amount (0-1)
        """
        # Calculate spectral centroid variation
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        if spectral_centroid.size == 0:
            return 0.0
        
        # Look for periodic variations in spectral centroid
        centroid_variation = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8)
        
        # Normalize to 0-1 scale
        chorus_amount = min(1.0, centroid_variation * 5.0)
        
        return chorus_amount
    
    def _detect_compression(self, y: np.ndarray) -> float:
        """Detect compression presence.
        
        Args:
            y: Audio signal
            
        Returns:
            Compression amount (0-1)
        """
        # Calculate dynamic range
        rms = librosa.feature.rms(y=y)
        
        if rms.size == 0:
            return 0.0
        
        # Calculate coefficient of variation
        rms_cv = np.std(rms) / (np.mean(rms) + 1e-8)
        
        # Lower variation indicates more compression
        compression_amount = 1.0 - min(1.0, rms_cv * 2.0)
        
        return float(max(0.0, compression_amount))