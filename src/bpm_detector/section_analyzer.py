"""Section analysis and detection module."""

from typing import Any, Dict, List, Optional

import librosa
import numpy as np


class SectionAnalyzer:
    """Analyzes and detects musical section characteristics."""

    # ASCII label definitions (J-Pop terminology)
    JP_ASCII_LABELS = {
        "intro": "Intro",
        "verse": "A-melo",
        "pre_chorus": "B-melo",
        "chorus": "Sabi",
        "bridge": "C-melo",
        "instrumental": "Kansou",
        "break": "Break",
        "interlude": "Interlude",
        "solo": "Solo",
        "spoken": "Serifu",
        "outro": "Outro",
    }

    def __init__(self, hop_length: int = 512):
        """Initialize section analyzer.

        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length

    def refine_section_labels_with_spectral_analysis(
        self, y: np.ndarray, sr: int, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine section labels using spectral flux analysis.

        Args:
            y: Audio signal
            sr: Sample rate
            sections: List of sections to refine

        Returns:
            Refined sections list
        """
        refined_sections = []

        for section in sections:
            start_sample = int(section['start_time'] * sr)
            end_sample = int(section['end_time'] * sr)
            segment = y[start_sample:end_sample]

            if len(segment) == 0:
                refined_sections.append(section)
                continue

            # Calculate spectral flux (measure of spectral change)
            stft = librosa.stft(segment, hop_length=self.hop_length)
            spectral_flux = np.sum(np.diff(np.abs(stft), axis=1) ** 2, axis=0)
            avg_flux = np.mean(spectral_flux) if len(spectral_flux) > 0 else 0

            # Refine section type based on spectral characteristics
            refined_type = section['type']

            # Get characteristics with defaults - handle both dict and list formats
            characteristics = section.get('characteristics', {})
            if isinstance(characteristics, list):
                # Convert list to dict with default values
                characteristics = {
                    'energy': 0.5,
                    'spectral_complexity': 0.5,
                    'harmonic_content': 0.5,
                    'rhythmic_density': 0.5,
                }
            energy_level = characteristics.get('energy', 0.5)
            complexity = characteristics.get('spectral_complexity', 0.5)

            # If classified as outro but has high spectral activity, reclassify
            if section['type'] == 'outro' and avg_flux > 0.1:
                if energy_level > 0.5:
                    refined_type = 'chorus'
                elif complexity > 0.6:
                    refined_type = 'bridge'
                else:
                    refined_type = 'verse'

            # If classified as bridge but has low complexity, might be verse
            elif section['type'] == 'bridge' and complexity < 0.4:
                refined_type = 'verse'

            # Enhanced instrumental section classification
            elif section['type'] == 'instrumental':
                spectral_features = {
                    'spectral_centroid': np.array([2000.0]),
                    'spectral_rolloff': np.array([4000.0]),
                    'mfcc': np.random.randn(13, 1),
                }
                refined_type = self.classify_instrumental_subtype(section, spectral_features)

            # Create refined section
            refined_section = section.copy()
            refined_section['type'] = refined_type
            refined_section['ascii_label'] = self.JP_ASCII_LABELS.get(refined_type, refined_type)
            refined_section['spectral_flux'] = float(avg_flux)

            refined_sections.append(refined_section)

        return refined_sections

    def classify_instrumental_subtype(
        self, section: Dict[str, Any], spectral_features: Optional[Dict[str, Any]] = None
    ) -> str:
        """Classify instrumental sections into more specific subtypes.

        Args:
            section: Section information
            spectral_features: Spectral features dictionary (optional)

        Returns:
            Refined instrumental section type
        """
        # Get characteristics from section, with defaults - handle both dict and list formats
        characteristics = section.get('characteristics', {})
        if isinstance(characteristics, list):
            # Convert list to dict with default values
            characteristics = {
                'energy': 0.5,
                'spectral_complexity': 0.5,
                'harmonic_content': 0.5,
                'rhythmic_density': 0.5,
            }
        energy = characteristics.get('energy', 0.5)
        complexity = characteristics.get('spectral_complexity', 0.5)
        rhythmic_density = characteristics.get('rhythmic_density', 0.5)

        duration = section.get('duration', section.get('end_time', 0) - section.get('start_time', 0))

        # Classify based on position, energy, complexity, and duration

        # Break: Low energy, low complexity, short duration
        if energy < 0.4 and complexity < 0.4 and duration < 15:
            return 'breakdown'

        # Solo: High energy, high complexity, medium duration
        elif energy > 0.6 and complexity > 0.7 and 10 < duration < 30:
            return 'solo'

        # Interlude: Medium energy, medium-high complexity, longer duration
        elif 0.4 <= energy <= 0.7 and complexity > 0.5 and duration > 15:
            return 'interlude'

        # Buildup: High energy, increasing complexity
        elif energy > 0.7 and rhythmic_density > 0.8:
            return 'buildup'

        # Default to solo for high-energy sections
        else:
            return 'solo'

    def enhance_outro_detection(self, sections: List[Dict[str, Any]], y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Enhanced outro detection with fade analysis and harmonic resolution.

        Args:
            sections: List of sections to analyze
            y: Audio signal
            sr: Sample rate

        Returns:
            Sections with improved outro detection
        """
        if not sections:
            return sections

        enhanced_sections = sections.copy()
        total_duration = sections[-1]['end_time']

        # Analyze last 15% of the track for outro candidates
        outro_threshold_time = total_duration * 0.85

        for i, section in enumerate(enhanced_sections):
            if section['start_time'] >= outro_threshold_time:
                # Check for fade/silence ending
                is_fade_ending = self.detect_fade_ending(section, y, sr)

                # Check for harmonic resolution (tonic return)
                has_harmonic_resolution = self.detect_harmonic_resolution(section, y, sr)

                # Reclassify as outro if conditions are met
                if is_fade_ending or (has_harmonic_resolution and section['energy_level'] < 0.5):
                    enhanced_sections[i]['type'] = 'outro'
                    enhanced_sections[i]['ascii_label'] = self.JP_ASCII_LABELS.get('outro', 'outro')
                    enhanced_sections[i]['outro_confidence'] = 0.8 if is_fade_ending else 0.6

        return enhanced_sections

    def detect_fade_ending(self, section: Dict[str, Any], y: np.ndarray, sr: int) -> bool:
        """Detect fade/silence ending pattern.

        Args:
            section: Section to analyze
            y: Audio signal
            sr: Sample rate

        Returns:
            True if fade ending is detected
        """
        start_sample = int(section['start_time'] * sr)
        end_sample = int(section['end_time'] * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr:  # Less than 1 second
            return False

        # Analyze energy in 10-second moving windows
        duration = section.get('duration', section['end_time'] - section['start_time'])
        window_duration = min(10.0, duration / 2)
        window_samples = int(window_duration * sr)

        if len(segment) < window_samples * 2:
            return False

        # Calculate RMS energy for first and last windows
        first_window = segment[:window_samples]
        last_window = segment[-window_samples:]

        first_rms = np.sqrt(np.mean(first_window**2))
        last_rms = np.sqrt(np.mean(last_window**2))

        # Convert to dB
        if first_rms > 0 and last_rms > 0:
            db_drop = 20 * np.log10(last_rms / first_rms)

            # Check for significant fade (6-12dB drop)
            if db_drop < -6.0:
                return True

        # Check for voiced ratio (vocal presence)
        voiced_frames = 0
        total_frames = 0

        # Simple voiced detection using zero crossing rate
        frame_length = 2048
        for i in range(0, len(segment) - frame_length, frame_length // 2):
            frame = segment[i : i + frame_length]
            zcr = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)

            # Low ZCR typically indicates voiced content
            if zcr < 0.1:
                voiced_frames += 1
            total_frames += 1

        voiced_ratio = voiced_frames / max(total_frames, 1)

        # Low voiced ratio indicates instrumental/fade ending
        return voiced_ratio < 0.15

    def detect_harmonic_resolution(self, section: Dict[str, Any], y: np.ndarray, sr: int) -> bool:
        """Detect harmonic resolution to tonic (key return).

        Args:
            section: Section to analyze
            y: Audio signal
            sr: Sample rate

        Returns:
            True if harmonic resolution is detected
        """
        start_sample = int(section['start_time'] * sr)
        end_sample = int(section['end_time'] * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr:  # Less than 1 second
            return False

        # Extract chroma features for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, hop_length=512)

        # Analyze final bars for tonic presence
        final_portion = 0.3  # Last 30% of section
        final_start = int(chroma.shape[1] * (1 - final_portion))
        final_chroma = chroma[:, final_start:]

        if final_chroma.shape[1] == 0:
            return False

        # Calculate average chroma in final portion
        avg_final_chroma = np.mean(final_chroma, axis=1)

        # Find the most prominent pitch class (potential tonic)
        tonic_candidate = np.argmax(avg_final_chroma)
        tonic_strength = avg_final_chroma[tonic_candidate]

        # Check if tonic is significantly stronger than other notes
        other_strengths = np.concatenate([avg_final_chroma[:tonic_candidate], avg_final_chroma[tonic_candidate + 1 :]])

        if len(other_strengths) > 0:
            max_other: float = float(np.max(other_strengths))
            tonic_dominance = tonic_strength / (max_other + 1e-8)

            # Strong tonic presence indicates resolution
            return bool(tonic_dominance > 1.5 and tonic_strength > 0.5)

        return False

    def detect_chorus_hooks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and enforce chorus sections based on hook patterns.

        Args:
            sections: List of sections to process

        Returns:
            Processed sections with chorus hooks detected
        """
        if not sections:
            return sections

        processed = sections.copy()

        for i, section in enumerate(processed):
            # Hook pattern detection: high energy + brightness + 6-10 bar duration
            energy_level = section.get('energy_level', 0.0)
            brightness = section.get('brightness', 0.0)  # May not be available
            duration = section.get('duration', section['end_time'] - section['start_time'])

            # Strong hook pattern criteria
            is_hook = (
                energy_level > 0.65 and 6 <= duration <= 10 and section['type'] in ['verse', 'bridge', 'pre_chorus']
            )  # Convert these to chorus

            # Additional brightness check if available
            if brightness > 0.0:  # If brightness data is available
                is_hook = is_hook and brightness > 0.6

            if is_hook:
                processed[i]['type'] = 'chorus'
                processed[i]['ascii_label'] = self.JP_ASCII_LABELS.get('chorus', 'chorus')

        return processed

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
                'structure_complexity': 0.0,
                'section_count': 0,
                'total_duration': 0.0,
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

        # Calculate total duration
        if sections:
            last_section = sections[-1]
            if 'end_time' in last_section:
                total_duration = last_section['end_time']
            elif 'duration' in last_section and 'start_time' in last_section:
                total_duration = last_section['start_time'] + last_section['duration']
            else:
                # Calculate from all durations
                total_duration = sum(s.get('duration', 10.0) for s in sections)
        else:
            total_duration = 0.0

        return {
            'form': form,
            'repetition_ratio': repetition_ratio,
            'structure_complexity': structural_complexity,
            'structural_complexity': structural_complexity,  # Add both for compatibility
            'section_count': total_sections,
            'total_duration': total_duration,
            'unique_sections': unique_sections,
            'section_types': section_types,
        }

    def _section_to_letter(self, section_type: str) -> str:
        """Convert section type to letter for form notation.

        Args:
            section_type: Section type string

        Returns:
            Single letter representing the section
        """
        mapping = {
            'intro': 'I',  # Intro
            'verse': 'A',  # A-melo (Verse)
            'pre_chorus': 'R',  # B-melo (Pre-Chorus)
            'chorus': 'B',  # Sabi (Chorus)
            'bridge': 'C',  # C-melo (Bridge)
            'instrumental': 'D',  # Kansou (Instrumental)
            'break': 'K',  # Break
            'interlude': 'L',  # Interlude
            'solo': 'S',  # Solo
            'spoken': 'P',  # Serifu (Spoken word/dialogue)
            'outro': 'O',  # Outro
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
        durations = []
        for s in sections:
            if 'duration' in s:
                durations.append(s['duration'])
            elif 'end_time' in s and 'start_time' in s:
                durations.append(s['end_time'] - s['start_time'])
            else:
                durations.append(10.0)  # Default duration if neither available
        duration_std = float(np.std(durations)) / (float(np.mean(durations)) + 1e-8)
        duration_complexity = min(1.0, duration_std)

        # 3. Non-standard form patterns
        form_complexity = 0.0
        if len(sections) > 8:  # Long form
            form_complexity += 0.3
        if unique_types > 4:  # Many section types
            form_complexity += 0.3

        # Combine factors
        overall_complexity = (float(type_complexity) + float(duration_complexity) + form_complexity) / 3.0

        return min(1.0, overall_complexity)

    def summarize_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Generate a summary text of sections for display.

        Args:
            sections: List of sections

        Returns:
            Summary text string
        """
        lines = []
        lines.append(f"Section List (Estimated {len(sections)} sections)")
        tmpl = "  {idx:>2}. {typ:<8}({ascii}) {start:>6.1f}s - {end:>6.1f}s ({dur:>5.1f}s)"
        for i, s in enumerate(sections, 1):
            duration = s.get('duration', s['end_time'] - s['start_time'])
            lines.append(
                tmpl.format(
                    idx=i,
                    typ=s['type'].capitalize(),
                    ascii=s.get('ascii_label', s['type']),
                    start=s['start_time'],
                    end=s['end_time'],
                    dur=duration,
                )
            )
        return "\n".join(lines)

    def calculate_energy_scale(self, y: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive energy scale based on track characteristics.

        Args:
            y: Audio signal

        Returns:
            Dictionary with energy statistics
        """
        # Calculate RMS energy for the entire track
        rms_values = []
        window_size = 2048
        hop_size = 1024

        for i in range(0, len(y) - window_size, hop_size):
            window = y[i : i + window_size]
            rms = np.sqrt(np.mean(window**2))
            if rms > 0:  # Avoid zero values
                rms_values.append(rms)

        if rms_values:
            # Use 10th and 90th percentiles for robust scaling
            p10 = np.percentile(rms_values, 10)
            p90 = np.percentile(rms_values, 90)
            mean_energy = np.mean(rms_values)

            # Set energy scale based on dynamic range
            return {
                'min_energy': float(p10),
                'max_energy': float(p90),
                'mean_energy': float(mean_energy),
                'energy_range': float(p90 - p10),
            }
        else:
            return {'min_energy': 0.0, 'max_energy': 0.05, 'mean_energy': 0.025, 'energy_range': 0.05}
