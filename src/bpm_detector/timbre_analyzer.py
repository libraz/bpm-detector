"""Timbre and instrument analysis module."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TimbreAnalyzer:
    """Analyzes timbral features and instrument presence."""
    
    # Instrument frequency ranges (Hz)
    INSTRUMENT_RANGES = {
        'bass': (20, 250),
        'kick_drum': (20, 100),
        'snare_drum': (150, 4000),
        'hi_hat': (5000, 20000),
        'piano': (27.5, 4186),
        'guitar': (82, 1319),
        'vocals': (80, 1100),
        'strings': (196, 2093),
        'brass': (87, 1175),
        'woodwinds': (147, 2093)
    }
    
    # Spectral feature ranges for classification
    FEATURE_RANGES = {
        'brightness': {'low': 0.3, 'high': 0.7},
        'roughness': {'low': 0.2, 'high': 0.8},
        'warmth': {'low': 0.3, 'high': 0.7}
    }
    
    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize timbre analyzer.
        
        Args:
            hop_length: Hop length for analysis
            n_fft: FFT size
        """
        self.hop_length = hop_length
        self.n_fft = n_fft
        
    def extract_timbral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive timbral features.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of timbral features
        """
        features = {}
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=y, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )
        
        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=y, hop_length=self.hop_length
        )
        
        # Tonnetz (harmonic network)
        features['tonnetz'] = librosa.feature.tonnetz(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Mel-frequency features
        features['melspectrogram'] = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        return features
    
    def analyze_brightness(self, spectral_centroid: np.ndarray, sr: int) -> float:
        """Analyze spectral brightness.
        
        Args:
            spectral_centroid: Spectral centroid values
            sr: Sample rate
            
        Returns:
            Brightness score (0-1)
        """
        if len(spectral_centroid) == 0:
            return 0.0
        
        # Normalize by Nyquist frequency
        normalized_centroid = np.mean(spectral_centroid) / (sr / 2)
        
        # Convert to 0-1 scale
        brightness = min(1.0, max(0.0, normalized_centroid))
        
        return brightness
    
    def analyze_roughness(self, spectral_contrast: np.ndarray) -> float:
        """Analyze spectral roughness.
        
        Args:
            spectral_contrast: Spectral contrast values
            
        Returns:
            Roughness score (0-1)
        """
        if spectral_contrast.size == 0:
            return 0.0
        
        # Calculate variance in spectral contrast
        contrast_variance = np.var(spectral_contrast)
        
        # Normalize to 0-1 scale
        roughness = min(1.0, contrast_variance / 10.0)
        
        return roughness
    
    def analyze_warmth(self, mfcc: np.ndarray) -> float:
        """Analyze timbral warmth.
        
        Args:
            mfcc: MFCC features
            
        Returns:
            Warmth score (0-1)
        """
        if mfcc.size == 0:
            return 0.0
        
        # Use lower MFCC coefficients for warmth
        # Higher values in lower coefficients indicate more low-frequency content
        low_freq_energy = np.mean(mfcc[1:4, :])  # Skip first coefficient (energy)
        
        # Normalize and convert to warmth score
        warmth = 1.0 / (1.0 + np.exp(-low_freq_energy))  # Sigmoid function
        
        return warmth
    
    def analyze_density(self, features: Dict[str, np.ndarray]) -> float:
        """Analyze acoustic density/texture.
        
        Args:
            features: Dictionary of timbral features
            
        Returns:
            Density score (0-1)
        """
        spectral_flatness = features.get('spectral_flatness', np.array([]))
        rms = features.get('rms', np.array([]))
        
        if len(spectral_flatness) == 0 or len(rms) == 0:
            return 0.0
        
        # High spectral flatness = more noise-like = higher density
        flatness_score = np.mean(spectral_flatness)
        
        # High RMS = more energy = potentially higher density
        rms_score = np.mean(rms)
        
        # Combine scores
        density = (flatness_score + min(1.0, rms_score * 5)) / 2.0
        
        return min(1.0, density)
    
    def classify_instruments(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Classify instruments present in the audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            List of detected instruments with confidence scores
        """
        instruments = []
        
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(y)
        
        # Analyze frequency content for different instruments
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        for instrument, (low_freq, high_freq) in self.INSTRUMENT_RANGES.items():
            confidence = self._calculate_instrument_confidence(
                magnitude, freqs, low_freq, high_freq, instrument, harmonic, percussive
            )
            
            if confidence > 0.3:  # Threshold for detection
                prominence = self._calculate_instrument_prominence(
                    magnitude, freqs, low_freq, high_freq
                )
                
                instruments.append({
                    'instrument': instrument,
                    'confidence': confidence,
                    'prominence': prominence
                })
        
        # Sort by confidence
        instruments.sort(key=lambda x: x['confidence'], reverse=True)
        
        return instruments
    
    def _calculate_instrument_confidence(self, magnitude: np.ndarray, freqs: np.ndarray,
                                       low_freq: float, high_freq: float, 
                                       instrument: str, harmonic: np.ndarray,
                                       percussive: np.ndarray) -> float:
        """Calculate confidence for instrument presence.
        
        Args:
            magnitude: STFT magnitude
            freqs: Frequency bins
            low_freq: Low frequency bound
            high_freq: High frequency bound
            instrument: Instrument name
            harmonic: Harmonic component
            percussive: Percussive component
            
        Returns:
            Confidence score (0-1)
        """
        # Find frequency range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        if not np.any(freq_mask):
            return 0.0
        
        # Calculate energy in frequency range
        range_energy = np.mean(magnitude[freq_mask, :])
        total_energy = np.mean(magnitude)
        
        if total_energy == 0:
            return 0.0
        
        energy_ratio = range_energy / total_energy
        
        # Apply instrument-specific heuristics
        if instrument in ['kick_drum', 'snare_drum', 'hi_hat']:
            # Percussive instruments - check percussive component
            perc_energy = np.mean(np.abs(librosa.stft(percussive)))
            total_perc_energy = np.mean(np.abs(librosa.stft(percussive + harmonic)))
            
            if total_perc_energy > 0:
                perc_ratio = perc_energy / total_perc_energy
                confidence = (energy_ratio + perc_ratio) / 2.0
            else:
                confidence = energy_ratio
        else:
            # Harmonic instruments - check harmonic component
            harm_energy = np.mean(np.abs(librosa.stft(harmonic)))
            total_harm_energy = np.mean(np.abs(librosa.stft(harmonic + percussive)))
            
            if total_harm_energy > 0:
                harm_ratio = harm_energy / total_harm_energy
                confidence = (energy_ratio + harm_ratio) / 2.0
            else:
                confidence = energy_ratio
        
        return min(1.0, confidence * 2.0)  # Amplify for better discrimination
    
    def _calculate_instrument_prominence(self, magnitude: np.ndarray, freqs: np.ndarray,
                                       low_freq: float, high_freq: float) -> float:
        """Calculate instrument prominence in the mix.
        
        Args:
            magnitude: STFT magnitude
            freqs: Frequency bins
            low_freq: Low frequency bound
            high_freq: High frequency bound
            
        Returns:
            Prominence score (0-1)
        """
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        if not np.any(freq_mask):
            return 0.0
        
        range_energy = np.mean(magnitude[freq_mask, :])
        total_energy = np.mean(magnitude)
        
        if total_energy == 0:
            return 0.0
        
        prominence = range_energy / total_energy
        
        return min(1.0, prominence * 3.0)  # Amplify for better range
    
    def analyze_effects_usage(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze usage of audio effects.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of effect usage scores
        """
        effects = {}
        
        # Reverb detection (based on decay characteristics)
        effects['reverb'] = self._detect_reverb(y, sr)
        
        # Distortion detection (based on harmonic content)
        effects['distortion'] = self._detect_distortion(y, sr)
        
        # Chorus/Modulation detection (based on spectral fluctuation)
        effects['chorus'] = self._detect_chorus(y, sr)
        
        # Compression detection (based on dynamic range)
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
            return 0.0
        
        # Calculate autocorrelation to find decay patterns
        autocorr = np.correlate(envelope_mean, envelope_mean, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for long-term correlations (indicating reverb tail)
        long_term_corr = np.mean(autocorr[len(autocorr)//4:len(autocorr)//2])
        short_term_corr = np.mean(autocorr[:len(autocorr)//8])
        
        if short_term_corr == 0:
            return 0.0
        
        reverb_ratio = long_term_corr / short_term_corr
        
        return min(1.0, reverb_ratio)
    
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
        
        return max(0.0, compression_amount)
    
    def analyze_texture(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze overall acoustic texture.
        
        Args:
            features: Dictionary of timbral features
            
        Returns:
            Dictionary of texture characteristics
        """
        texture = {}
        
        # Smoothness (based on spectral flatness)
        spectral_flatness = features.get('spectral_flatness', np.array([]))
        if len(spectral_flatness) > 0:
            texture['smoothness'] = 1.0 - np.mean(spectral_flatness)
        else:
            texture['smoothness'] = 0.0
        
        # Richness (based on spectral bandwidth)
        spectral_bandwidth = features.get('spectral_bandwidth', np.array([]))
        if len(spectral_bandwidth) > 0:
            texture['richness'] = min(1.0, np.mean(spectral_bandwidth) / 5000.0)
        else:
            texture['richness'] = 0.0
        
        # Clarity (based on spectral contrast)
        spectral_contrast = features.get('spectral_contrast', np.array([]))
        if spectral_contrast.size > 0:
            texture['clarity'] = min(1.0, np.mean(spectral_contrast) / 30.0)
        else:
            texture['clarity'] = 0.0
        
        # Fullness (based on RMS energy distribution)
        rms = features.get('rms', np.array([]))
        if len(rms) > 0:
            texture['fullness'] = min(1.0, np.mean(rms) * 10.0)
        else:
            texture['fullness'] = 0.0
        
        return texture
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete timbre analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Complete timbre analysis results
        """
        # Extract timbral features
        features = self.extract_timbral_features(y, sr)
        
        # Analyze basic timbral characteristics
        brightness = self.analyze_brightness(features['spectral_centroid'], sr)
        roughness = self.analyze_roughness(features['spectral_contrast'])
        warmth = self.analyze_warmth(features['mfcc'])
        density = self.analyze_density(features)
        
        # Classify instruments
        instruments = self.classify_instruments(y, sr)
        
        # Analyze effects usage
        effects = self.analyze_effects_usage(y, sr)
        
        # Analyze texture
        texture = self.analyze_texture(features)
        
        return {
            'brightness': brightness,
            'roughness': roughness,
            'warmth': warmth,
            'density': density,
            'dominant_instruments': instruments[:5],  # Top 5 instruments
            'effects_usage': effects,
            'texture': texture,
            'spectral_features': {
                'centroid_mean': float(np.mean(features['spectral_centroid'])),
                'bandwidth_mean': float(np.mean(features['spectral_bandwidth'])),
                'rolloff_mean': float(np.mean(features['spectral_rolloff'])),
                'flatness_mean': float(np.mean(features['spectral_flatness'])),
                'zcr_mean': float(np.mean(features['zcr']))
            }
        }