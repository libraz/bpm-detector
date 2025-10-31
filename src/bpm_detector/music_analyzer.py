"""BPM and Key detection algorithms."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, cast

import librosa
import numpy as np

# Import new analyzers
from .chord_analyzer import ChordProgressionAnalyzer
from .dynamics_analyzer import DynamicsAnalyzer
from .key_detector import KeyDetector as NewKeyDetector
from .melody_harmony_analyzer import MelodyHarmonyAnalyzer
from .rhythm_analyzer import RhythmAnalyzer
from .similarity_engine import SimilarityEngine
from .structure_analyzer import StructureAnalyzer
from .timbre_analyzer import TimbreAnalyzer

# --- Configuration constants ---
SR_DEFAULT = 22_050
HOP_DEFAULT = 128
BIN_WIDTH = 0.5
RATIOS = [0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0, 3.0, 4.0]
TOL = 0.05
THRESH_HIGHER = 0.15  # 15% of total votes

# Note: Key detection constants moved to key_detector.py and key_profiles.py


class BPMDetector:
    """BPM detection using harmonic clustering."""

    def __init__(self, sr: int = SR_DEFAULT, hop_length: int = HOP_DEFAULT):
        self.sr = sr
        self.hop_length = hop_length

    def harmonic_cluster(self, bpms: np.ndarray, votes: np.ndarray) -> Dict[float, List[Tuple[float, int]]]:
        """Group BPM candidates into harmonic clusters."""
        clusters: Dict[float, List[Tuple[float, int]]] = {}
        for bpm, hit in sorted(zip(bpms, votes), key=lambda x: -x[1]):
            for base in list(clusters):
                r = bpm / base
                if any(abs(r - k) < TOL or abs(r - 1 / k) < TOL for k in RATIOS):
                    clusters[base].append((bpm, hit))
                    break
            else:
                clusters[bpm] = [(bpm, hit)]
        return clusters

    def smart_choice(self, clusters: Dict[float, List[Tuple[float, int]]], total_votes: int) -> Tuple[float, float]:
        """Choose the best BPM from clusters using smart selection."""
        # Handle empty clusters
        if not clusters:
            return 120.0, 0.0  # Default BPM with zero confidence

        # base cluster = largest votes
        base, base_vals = max(clusters.items(), key=lambda kv: sum(v for _, v in kv[1]))
        base_votes = sum(v for _, v in base_vals)

        higher = [(rep, sum(v for _, v in vals)) for rep, vals in clusters.items() if rep > base]
        higher.sort(key=lambda x: -x[1])

        if higher and higher[0][1] / total_votes >= THRESH_HIGHER:
            # choose the representative BPM of the strongest higher cluster
            rep_bpm = higher[0][0]
            conf = 100 * higher[0][1] / total_votes
        else:
            if not base_vals:  # Additional safety check
                return 120.0, 0.0
            rep_bpm = max(base_vals, key=lambda x: x[1])[0]
            conf = 100 * base_votes / total_votes
        return rep_bpm, conf

    def detect(
        self, y: np.ndarray, sr: int, min_bpm: float = 40.0, max_bpm: float = 300.0, start_bpm: float = 150.0
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Detect BPM from audio signal with optimized processing."""
        # Choose API without FutureWarning
        if (
            hasattr(librosa, "feature")
            and hasattr(librosa.feature, "rhythm")
            and hasattr(librosa.feature.rhythm, "tempo")
        ):
            tempo_func = librosa.feature.rhythm.tempo
        else:
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*librosa.beat.tempo.*")
            tempo_func = librosa.beat.tempo

        # Optimize for speed: use smaller hop_length for faster processing
        optimized_hop = min(self.hop_length, 512)

        # Limit audio length for very long files to improve speed
        max_duration = 180  # 3 minutes max for BPM detection
        if len(y) > max_duration * sr:
            # Use middle section for better representation
            start_idx = len(y) // 4
            end_idx = start_idx + (max_duration * sr)
            y_trimmed = y[start_idx:end_idx]
        else:
            y_trimmed = y

        cands = tempo_func(
            y=y_trimmed, sr=sr, aggregate=None, hop_length=optimized_hop, max_tempo=max_bpm, start_bpm=start_bpm
        )

        bins = np.arange(min_bpm, max_bpm + BIN_WIDTH, BIN_WIDTH)
        hist, edges = np.histogram(cands, bins=bins)
        top_idx = hist.argsort()[::-1][:10]
        top_bpms = edges[top_idx]
        top_hits = hist[top_idx]

        clusters = self.harmonic_cluster(top_bpms, top_hits)
        rep_bpm, conf = self.smart_choice(clusters, hist.sum())

        return rep_bpm, conf, top_bpms, top_hits


# Legacy KeyDetector class removed - using new KeyDetector from key_detector.py
# Export KeyDetector for backward compatibility
KeyDetector = NewKeyDetector


class AudioAnalyzer:
    """Main analyzer combining all music analysis features."""

    def __init__(self, sr: int = SR_DEFAULT, hop_length: int = HOP_DEFAULT):
        self.sr = sr
        self.hop_length = hop_length

        # Original detectors
        self.bpm_detector = BPMDetector(sr, hop_length)
        self.key_detector = NewKeyDetector(hop_length)

        # New analyzers
        self.chord_analyzer = ChordProgressionAnalyzer(hop_length)
        self.structure_analyzer = StructureAnalyzer(hop_length)
        self.rhythm_analyzer = RhythmAnalyzer(hop_length)
        self.timbre_analyzer = TimbreAnalyzer(hop_length)
        self.melody_harmony_analyzer = MelodyHarmonyAnalyzer(hop_length)
        self.dynamics_analyzer = DynamicsAnalyzer(hop_length)
        self.similarity_engine = SimilarityEngine()

        # Feature cache for efficiency
        self._feature_cache: Dict[str, Any] = {}
        self._cache_enabled = True

    def clear_cache(self):
        """Clear feature cache to free memory."""
        self._feature_cache.clear()

    def _get_cached_features(self, cache_key: str, compute_func, *args, **kwargs):
        """Get cached features or compute and cache them."""
        if not self._cache_enabled or cache_key not in self._feature_cache:
            features = compute_func(*args, **kwargs)
            if self._cache_enabled:
                self._feature_cache[cache_key] = features
            return features
        return self._feature_cache[cache_key]

    def analyze_file(
        self,
        path: str,
        detect_key: bool = True,
        comprehensive: bool = True,
        min_bpm: float = 40.0,
        max_bpm: float = 300.0,
        start_bpm: float = 150.0,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Analyze audio file comprehensively.

        Args:
            path: Path to audio file
            detect_key: Whether to detect musical key
            comprehensive: Whether to perform comprehensive analysis
            min_bpm: Minimum BPM to consider
            max_bpm: Maximum BPM to consider
            start_bpm: Starting BPM for detection
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing complete analysis results
        """
        # Load audio with optimized settings
        y, sr_loaded = librosa.load(
            path,
            sr=self.sr,
            mono=True,
            dtype=np.float32,  # Use float32 for better memory efficiency
            res_type='kaiser_fast',  # Faster resampling
        )
        sr = int(sr_loaded)  # Ensure sr is int for type checking

        total_steps = 9 if comprehensive else 3
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(int(100 * current_step / total_steps))

        # Basic info
        duration = len(y) / sr

        # BPM detection
        bpm, bpm_conf, top_bpms, top_hits = self.bpm_detector.detect(y, sr, min_bpm, max_bpm, start_bpm)
        update_progress()

        # Key detection (using improved method from melody_harmony_analyzer)
        key = None
        key_conf = 0.0
        key_detection_result: Optional[Dict[str, Any]] = None
        if detect_key:
            # Use the improved key detection from new KeyDetector
            key_detection_result = cast(Dict[str, Any], self.key_detector.detect_key(y, sr))
            # Check for valid key detection (not Unknown format)
            if not key_detection_result['key'].startswith('Unknown'):
                key = f"{key_detection_result['key']} {key_detection_result['mode']}"
                key_conf = key_detection_result['confidence']  # Already 0-100 scale
            else:
                # Fallback to melody_harmony_analyzer
                fallback_result = self.melody_harmony_analyzer.detect_key(y, sr)
                # Check fallback result as well
                if not fallback_result['key'].startswith('Unknown'):
                    key = f"{fallback_result['key']} {fallback_result['mode']}"
                    key_conf = fallback_result['confidence']
                    # Update key_detection_result for consistency
                    key_detection_result = fallback_result
                else:
                    # Both methods failed, use None
                    key = None
                    key_conf = 0.0
        update_progress()

        # Basic results
        basic_info = {
            "filename": path,
            "duration": duration,
            "bpm": bpm,
            "bpm_confidence": bpm_conf,
            "bpm_candidates": list(zip(top_bpms, top_hits)),
            "key": key,
            "key_confidence": key_conf,
        }

        # Add detailed key detection results if available
        if key_detection_result:
            basic_info["key_detection_details"] = key_detection_result

        results = {"basic_info": basic_info}

        if not comprehensive:
            update_progress()
            return results

        # Comprehensive analysis
        try:
            # Chord progression analysis
            chord_analysis = self.chord_analyzer.analyze(y, sr, key or "C Major")
            results["chord_progression"] = chord_analysis
            update_progress()

            # Structure analysis
            structure_analysis = self.structure_analyzer.analyze(y, sr)
            results["structure"] = structure_analysis
            update_progress()

            # Rhythm analysis
            rhythm_analysis = self.rhythm_analyzer.analyze(y, sr)
            results["rhythm"] = rhythm_analysis
            update_progress()

            # Timbre analysis
            timbre_analysis = self.timbre_analyzer.analyze(y, sr)
            results["timbre"] = timbre_analysis
            update_progress()

            # Melody and harmony analysis
            melody_harmony_analysis = self.melody_harmony_analyzer.analyze(y, sr)
            results["melody_harmony"] = melody_harmony_analysis
            update_progress()

            # Dynamics analysis
            dynamics_analysis = self.dynamics_analyzer.analyze(y, sr)
            results["dynamics"] = dynamics_analysis
            update_progress()

            # Generate feature vector and similarity data
            feature_vector = self.similarity_engine.extract_feature_vector(results)
            results["similarity_features"] = {
                "feature_vector": feature_vector.tolist(),
                "feature_weights": self.similarity_engine.feature_weights,
            }

            # Generate reference tags
            reference_tags = self._generate_reference_tags(results)
            results["reference_tags"] = reference_tags  # type: ignore

            # Generate production notes
            production_notes = self._generate_production_notes(results)
            results["production_notes"] = production_notes

            update_progress()

        except Exception as e:
            print(f"Warning: Error in comprehensive analysis: {e}")
            # Continue with basic results if comprehensive analysis fails
        finally:
            # Clear cache and free memory after analysis
            if hasattr(self, '_feature_cache'):
                self.clear_cache()
            # Force garbage collection for large audio files
            import gc

            gc.collect()

        return results

    def _generate_reference_tags(self, results: Dict[str, Any]) -> List[str]:
        """Generate reference tags for the track.

        Args:
            results: Analysis results

        Returns:
            List of reference tags
        """
        tags = []

        # Basic info tags
        basic_info = results.get("basic_info", {})
        bpm = basic_info.get("bpm", 120)
        key = basic_info.get("key", "")

        # Tempo tags
        if bpm < 80:
            tags.append("slow-tempo")
        elif bpm < 120:
            tags.append("mid-tempo")
        elif bpm < 140:
            tags.append("upbeat")
        else:
            tags.append("fast-tempo")

        # Key tags
        if key:
            if "Major" in key:
                tags.append("major-key")
            elif "Minor" in key:
                tags.append("minor-key")

        # Rhythm tags
        rhythm = results.get("rhythm", {})
        time_sig = rhythm.get("time_signature", "4/4")
        if time_sig != "4/4":
            tags.append(f"time-sig-{time_sig.replace('/', '-')}")

        groove = rhythm.get("groove_type", "straight")
        if groove != "straight":
            tags.append(groove)

        syncopation = rhythm.get("syncopation_level", 0.0)
        if syncopation > 0.5:
            tags.append("syncopated")

        # Structure tags
        structure = results.get("structure", {})
        complexity = structure.get("structural_complexity", 0.0)
        if complexity > 0.7:
            tags.append("complex-structure")
        elif complexity < 0.3:
            tags.append("simple-structure")

        # Timbre tags
        timbre = results.get("timbre", {})
        instruments = timbre.get("dominant_instruments", [])

        for instrument in instruments[:3]:  # Top 3 instruments
            inst_name = instrument.get("instrument", "")
            if inst_name:
                tags.append(f"{inst_name}-driven")

        brightness = timbre.get("brightness", 0.0)
        if brightness > 0.7:
            tags.append("bright")
        elif brightness < 0.3:
            tags.append("dark")

        # Energy tags
        dynamics = results.get("dynamics", {})
        energy = dynamics.get("overall_energy", 0.0)
        if energy > 0.7:
            tags.append("high-energy")
        elif energy < 0.3:
            tags.append("low-energy")

        return tags

    def _generate_production_notes(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate production notes for the track.

        Args:
            results: Analysis results

        Returns:
            Dictionary of production notes
        """
        notes = {}

        # Arrangement density
        timbre = results.get("timbre", {})
        density = timbre.get("density", 0.0)

        if density > 0.7:
            notes["arrangement_density"] = "dense"
        elif density > 0.4:
            notes["arrangement_density"] = "medium"
        else:
            notes["arrangement_density"] = "sparse"

        # Production style inference
        instruments = timbre.get("dominant_instruments", [])
        inst_names = [inst.get("instrument", "") for inst in instruments]

        if "guitar" in inst_names and "drums" in inst_names:
            if "piano" in inst_names:
                notes["production_style"] = "rock_pop"
            else:
                notes["production_style"] = "rock"
        elif "piano" in inst_names:
            notes["production_style"] = "piano_driven"
        else:
            notes["production_style"] = "modern_pop"

        # Mix characteristics
        mix_chars = []

        dynamics = results.get("dynamics", {})
        dynamic_range = dynamics.get("dynamic_range", {}).get("dynamic_range_db", 0)

        if dynamic_range > 20:
            mix_chars.append("wide_dynamic_range")
        elif dynamic_range < 10:
            mix_chars.append("compressed")

        brightness = timbre.get("brightness", 0.0)
        if brightness > 0.7:
            mix_chars.append("bright_mix")
        elif brightness < 0.3:
            mix_chars.append("warm_mix")

        notes["mix_characteristics"] = mix_chars  # type: ignore

        return notes

    def generate_reference_sheet(self, results: Dict[str, Any]) -> str:
        """Generate a formatted reference sheet for music production.

        Args:
            results: Complete analysis results

        Returns:
            Formatted reference sheet as markdown string
        """
        basic_info = results.get("basic_info", {})
        chord_prog = results.get("chord_progression", {})
        structure = results.get("structure", {})
        rhythm = results.get("rhythm", {})
        timbre = results.get("timbre", {})
        melody_harmony = results.get("melody_harmony", {})
        dynamics = results.get("dynamics", {})

        sheet = f"""# Music Production Reference Sheet

## Basic Information
- **Tempo**: {basic_info.get('bpm', 120):.1f} BPM
- **Key**: {basic_info.get('key', 'Unknown')}
- **Time Signature**: {rhythm.get('time_signature', '4/4')}
- **Duration**: {basic_info.get('duration', 0):.0f} seconds

## Song Structure
- **Section Count**: {structure.get('section_count', 0)}
- **Structural Complexity**: {structure.get('structural_complexity', 0.0):.1f}
- **Repetition Ratio**: {structure.get('repetition_ratio', 0.0):.1%}

## Harmony & Chord Progression
- **Main Chord Progression**: {' - '.join(chord_prog.get('main_progression', []))}
- **Chord Complexity**: {chord_prog.get('chord_complexity', 0.0):.1%}
- **Harmonic Rhythm**: {chord_prog.get('harmonic_rhythm', 0.0):.1f} changes/sec

## Rhythm & Groove
- **Groove Type**: {rhythm.get('groove_type', 'straight')}
- **Syncopation Level**: {rhythm.get('syncopation_level', 0.0):.1%}
- **Rhythmic Complexity**: {rhythm.get('rhythmic_complexity', 0.0):.1%}

## Instrumentation & Timbre
- **Main Instruments**: {', '.join([inst.get('instrument', '') for inst in timbre.get('dominant_instruments', [])[:3]])}
- **Timbral Features**: Brightness {timbre.get('brightness', 0.0):.1%}, Warmth {timbre.get('warmth', 0.0):.1%}
- **Acoustic Density**: {timbre.get('density', 0.0):.1%}

## Melody & Harmony
- **Melodic Range**: {melody_harmony.get('melodic_range', {}).get('range_octaves', 0.0):.1f} octaves
- **Harmonic Complexity**: {melody_harmony.get('harmony_complexity', {}).get('harmonic_complexity', 0.0):.1%}
- **Consonance Level**: {melody_harmony.get('consonance', {}).get('consonance_level', 0.0):.1%}

## Dynamics & Energy
- **Dynamic Range**: {dynamics.get('dynamic_range', {}).get('dynamic_range_db', 0.0):.1f}dB
- **Average Loudness**: {dynamics.get('loudness', {}).get('average_loudness_db', -30.0):.1f}dB
- **Overall Energy**: {dynamics.get('overall_energy', 0.0):.1%}

## Production Notes
{self._format_production_notes(results.get('production_notes', {}))}

## Reference Tags
{', '.join(results.get('reference_tags', []))}
"""

        return sheet

    def _format_production_notes(self, notes: Dict[str, Any]) -> str:
        """Format production notes for display.

        Args:
            notes: Production notes dictionary

        Returns:
            Formatted notes string
        """
        formatted = []

        density = notes.get("arrangement_density", "medium")
        formatted.append(f"- Arrangement Density: {density}")

        style = notes.get("production_style", "modern_pop")
        formatted.append(f"- Production Style: {style}")

        mix_chars = notes.get("mix_characteristics", [])
        if mix_chars:
            formatted.append(f"- Mix Characteristics: {', '.join(mix_chars)}")

        return '\n'.join(formatted)
