"""Instrument classification module."""

from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np


class InstrumentClassifier:
    """Classifies instruments present in audio signals."""

    # Instrument frequency ranges (Hz) - expanded and refined
    INSTRUMENT_RANGES = {
        'vocals': (80, 1100),
        'piano': (27.5, 4186),
        'guitar': (82, 1319),
        'electric_guitar': (82, 2093),
        'bass': (20, 250),
        'synthesizer': (20, 8000),
        'strings': (196, 2093),
        'brass': (87, 1175),
        'woodwinds': (147, 2093),
        'drums': (20, 8000),
        'kick_drum': (20, 100),
        'snare_drum': (150, 4000),
        'hi_hat': (5000, 20000),
        'cymbals': (3000, 20000),
        'organ': (16, 4186),
        'flute': (262, 2093),
        'violin': (196, 3136),
        'cello': (65, 1047),
        'saxophone': (138, 880),
    }

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize instrument classifier.

        Args:
            hop_length: Hop length for analysis
            n_fft: FFT size
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

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

        # Compute STFT for frequency analysis
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        for instrument, freq_range in self.INSTRUMENT_RANGES.items():
            low_freq, high_freq = freq_range
            confidence = self._calculate_instrument_confidence(
                magnitude,
                freqs,
                low_freq=low_freq,
                high_freq=high_freq,
                instrument=instrument,
                harmonic=harmonic,
                percussive=percussive,
            )

            # Use different thresholds for different instrument types
            if instrument in ['vocals', 'piano', 'guitar', 'bass']:
                threshold = 0.15
            elif instrument in ['drums', 'kick_drum', 'snare_drum', 'hi_hat']:
                threshold = 0.12
            else:
                threshold = 0.18

            if confidence > threshold:
                prominence = self._calculate_instrument_prominence(
                    magnitude, freqs, low_freq=low_freq, high_freq=high_freq
                )

                instruments.append({'instrument': instrument, 'confidence': confidence, 'prominence': prominence})

        # Filter redundant instruments
        instruments = self._filter_redundant_instruments(instruments)

        # Sort by confidence
        instruments.sort(key=lambda x: float(x['confidence']), reverse=True)  # type: ignore[arg-type]

        return instruments

    def _calculate_instrument_confidence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        instrument: Optional[str] = None,
        harmonic: Optional[np.ndarray] = None,
        percussive: Optional[np.ndarray] = None,
        spectral_shape: Optional[Any] = None,
    ) -> float:
        """Calculate confidence for instrument presence.

        Args:
            magnitude: STFT magnitude
            freqs: Frequency bins
            freq_range: Frequency range tuple (low, high)
            low_freq: Low frequency bound (alternative to freq_range)
            high_freq: High frequency bound (alternative to freq_range)
            instrument: Instrument name
            harmonic: Harmonic component
            percussive: Percussive component
            spectral_shape: Spectral shape hint

        Returns:
            Confidence score (0-1)
        """
        # Handle both parameter styles for backward compatibility
        if freq_range is not None:
            low_freq, high_freq = freq_range
        elif low_freq is None or high_freq is None:
            return 0.0

        # Find frequency range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

        if not np.any(freq_mask):
            return 0.0

        # Calculate energy in frequency range
        if magnitude.ndim == 2:
            range_energy = np.mean(magnitude[freq_mask, :])
            total_energy = np.mean(magnitude)
        else:
            range_energy = np.mean(magnitude[freq_mask])
            total_energy = np.mean(magnitude)

        if total_energy == 0:
            return 0.0

        energy_ratio = range_energy / total_energy

        # Apply instrument-specific heuristics if components are available
        if harmonic is not None and percussive is not None and instrument is not None:
            if instrument in ['kick_drum', 'snare_drum', 'hi_hat', 'drums', 'cymbals']:
                # Percussive instruments - check percussive component
                try:
                    perc_energy = np.mean(np.abs(librosa.stft(percussive)))
                    total_perc_energy = np.mean(np.abs(librosa.stft(percussive + harmonic)))

                    if total_perc_energy > 0:
                        perc_ratio = perc_energy / total_perc_energy
                        confidence = (energy_ratio * 1.5 + perc_ratio) / 2.0
                    else:
                        confidence = energy_ratio * 1.5
                except Exception:
                    confidence = energy_ratio * 1.5
            elif instrument == 'bass':
                # Bass instruments - boost low frequency detection
                confidence = energy_ratio * 2.0
            elif instrument in ['synthesizer', 'organ']:
                # Electronic instruments - moderate boost
                confidence = energy_ratio * 1.3
            else:
                # Harmonic instruments - check harmonic component
                try:
                    harm_energy = np.mean(np.abs(librosa.stft(harmonic)))
                    total_harm_energy = np.mean(np.abs(librosa.stft(harmonic + percussive)))

                    if total_harm_energy > 0:
                        harm_ratio = harm_energy / total_harm_energy
                        confidence = (energy_ratio + harm_ratio) / 2.0
                    else:
                        confidence = energy_ratio
                except Exception:
                    confidence = energy_ratio
        else:
            confidence = energy_ratio

        return float(min(1.0, confidence * 2.5))

    def _calculate_instrument_prominence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> float:
        """Calculate instrument prominence in the mix.

        Args:
            magnitude: STFT magnitude
            freqs: Frequency bins
            low_freq: Low frequency bound
            high_freq: High frequency bound

        Returns:
            Prominence score (0-1)
        """
        # Handle both parameter styles for backward compatibility
        if freq_range is not None:
            low_freq, high_freq = freq_range
        elif low_freq is None or high_freq is None:
            return 0.0

        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

        if not np.any(freq_mask):
            return 0.0

        if magnitude.ndim == 2:
            range_energy = np.mean(magnitude[freq_mask, :])
            total_energy = np.mean(magnitude)
        else:
            range_energy = np.mean(magnitude[freq_mask])
            total_energy = np.mean(magnitude)

        if total_energy == 0:
            return 0.0

        prominence = range_energy / total_energy

        return float(min(1.0, prominence * 3.0))  # Amplify for better range

    def _filter_redundant_instruments(self, instruments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out redundant instrument detections.

        Args:
            instruments: List of detected instruments

        Returns:
            Filtered list of instruments
        """
        # Define instrument groups that are mutually exclusive
        instrument_groups = [
            ['guitar', 'electric_guitar'],
            ['violin', 'strings'],
            ['cello', 'strings'],
            ['piano'],  # Piano should be unique
        ]

        # Allow multiple drum instruments to coexist
        # Don't group drums together - let kick_drum, snare_drum, hi_hat all appear

        filtered = []
        used_groups = set()
        seen_instruments = set()

        for instrument in instruments:
            inst_name = instrument['instrument']

            # Skip if we've already seen this exact instrument
            if inst_name in seen_instruments:
                continue

            # Check if this instrument belongs to a group
            group_found = False
            for i, group in enumerate(instrument_groups):
                if inst_name in group:
                    if i not in used_groups:
                        # First instrument from this group - keep it
                        filtered.append(instrument)
                        used_groups.add(i)
                        seen_instruments.add(inst_name)
                    group_found = True
                    break

            if not group_found:
                # Not in any group - keep it
                filtered.append(instrument)
                seen_instruments.add(inst_name)

        return filtered
