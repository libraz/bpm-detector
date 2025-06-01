"""J-Pop specific structure optimization module."""

import numpy as np
from typing import List, Dict, Any


class JPopStructureOptimizer:
    """Optimizes musical sections for J-Pop structure patterns."""
    
    # ASCII label definitions (J-Pop terminology)
    JP_ASCII_LABELS = {
        "intro":        "Intro",
        "verse":        "A-melo",
        "pre_chorus":   "B-melo",
        "chorus":       "Sabi",
        "bridge":       "C-melo",
        "instrumental": "Kansou",
        "break":        "Break",
        "interlude":    "Interlude",
        "solo":         "Solo",
        "spoken":       "Serifu",
        "outro":        "Outro",
        "drop_chorus":  "OchiSabi",
    }
    
    def __init__(self):
        """Initialize J-Pop structure optimizer."""
        pass
    
    def suppress_consecutive_pre_chorus(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suppress consecutive pre-chorus sections to prevent over-segmentation.
        
        Args:
            sections: List of sections to process
            
        Returns:
            Processed sections with consecutive pre-chorus suppressed
        """
        if len(sections) <= 1:
            return sections
        
        processed = sections.copy()
        
        # Scan for consecutive pre-chorus sections (extended to handle R,R,R → A,R,B)
        i = 0
        while i < len(processed):
            # Look for patterns of 3+ consecutive pre-chorus
            consecutive_count = 0
            start_idx = i
            
            while (i < len(processed) and processed[i]['type'] == 'pre_chorus'):
                consecutive_count += 1
                i += 1
            
            # R chain normalization
            if consecutive_count >= 3:
                # Convert pattern R,R,R... → A,R,B
                for j in range(start_idx, start_idx + consecutive_count):
                    if j == start_idx:                # First → Verse
                        processed[j]['type'] = 'verse'
                        processed[j]['ascii_label'] = self.JP_ASCII_LABELS.get('verse', 'verse')
                    elif j == start_idx + consecutive_count - 1:  # Last → Chorus
                        processed[j]['type'] = 'chorus'
                        processed[j]['ascii_label'] = self.JP_ASCII_LABELS.get('chorus', 'chorus')
                    # Middle ones stay as pre_chorus
            
            # Handle 2 consecutive pre-chorus (Enhanced: Always downgrade 2nd to Verse for J-Pop structure)
            elif consecutive_count == 2:
                # R R → R A  (Downgrade 2nd to Verse even if followed by Chorus for more J-Pop-like structure)
                processed[start_idx + 1]['type'] = 'verse'
                processed[start_idx + 1]['ascii_label'] = self.JP_ASCII_LABELS['verse']
                # Add lock flag to prevent re-upgrade during pairing
                processed[start_idx + 1]['_locked'] = True
            
            if consecutive_count == 0:
                i += 1
        
        return processed
    
    def enforce_pre_chorus_chorus_pairing(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enforce Pre-Chorus → Chorus pairing rules for J-Pop structure.
        
        Args:
            sections: List of sections to process
            
        Returns:
            Processed sections with enforced R→B pairing
        """
        if len(sections) <= 1:
            return sections
        
        processed = sections.copy()
        
        # First pass: Fix orphaned pre-chorus sections with chain limit (expanded to ±3 sections)
        R_MAX_CHAIN = 2  # Maximum consecutive pre-chorus sections allowed
        
        for i in range(len(processed)):
            if processed[i]['type'] == 'pre_chorus':
                # Count consecutive pre-chorus chain backwards
                r_chain_count = 0
                for j in range(i, -1, -1):  # Count backwards
                    if processed[j]['type'] == 'pre_chorus':
                        r_chain_count += 1
                    else:
                        break
                
                # If chain is too long, force next section to be chorus
                if r_chain_count >= R_MAX_CHAIN:
                    # Look for next section to convert to chorus
                    if i + 1 < len(processed):
                        next_section = processed[i + 1]
                        if next_section['type'] in ['verse', 'pre_chorus', 'bridge']:
                            processed[i + 1]['type'] = 'chorus'
                            processed[i + 1]['ascii_label'] = self.JP_ASCII_LABELS.get('chorus', 'chorus')
                            continue
                
                # Check if followed by chorus within 3 sections
                has_following_chorus = False
                for j in range(i + 1, min(i + 4, len(processed))):  # Check next 3 sections
                    if processed[j]['type'] in ['chorus', 'bridge']:
                        has_following_chorus = True
                        break
                
                # If no following chorus/bridge, downgrade to verse
                if not has_following_chorus:
                    processed[i]['type'] = 'verse'
                    processed[i]['ascii_label'] = self.JP_ASCII_LABELS.get('verse', 'verse')
        
        # Second pass: Ensure chorus sections have preceding pre-chorus
        for i in range(1, len(processed)):
            if processed[i]['type'] == 'chorus':
                # Check if preceded by pre-chorus
                has_preceding_pre_chorus = (processed[i - 1]['type'] == 'pre_chorus')
                
                # If no preceding pre-chorus, upgrade previous section if suitable
                if not has_preceding_pre_chorus:
                    prev_section = processed[i - 1]
                    
                    # Locked sections are not upgraded
                    if prev_section.get('_locked'):
                        continue
                    
                    # Only upgrade verse to pre-chorus if energy is building
                    if (prev_section['type'] == 'verse' and
                        prev_section.get('energy_level', 0.5) > 0.4):
                        processed[i - 1]['type'] = 'pre_chorus'
                        processed[i - 1]['ascii_label'] = self.JP_ASCII_LABELS.get('pre_chorus', 'pre_chorus')
        
        return processed
    
    def collapse_alternating_ar_patterns(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse A-R alternating patterns to A-R-B structure.
        
        Args:
            sections: List of sections to process
            
        Returns:
            Processed sections with A-R alternation collapsed
        """
        if len(sections) < 3:
            return sections
        
        # A-R alternating pattern collapse
        out = sections.copy()
        k = 0
        while k + 3 <= len(out):
            block = [s['type'] for s in out[k:k+3]]
            
            # Pattern: verse -> pre_chorus -> verse → convert to verse -> pre_chorus -> chorus
            if block == ['verse', 'pre_chorus', 'verse']:
                out[k+2]['type'] = 'chorus'
                out[k+2]['ascii_label'] = self.JP_ASCII_LABELS.get('chorus', 'chorus')
            
            k += 1
        
        return out
    
    def break_consecutive_chorus_chains(self, sections: List[Dict[str, Any]],
                                      y: np.ndarray, sr: int, bpm: float) -> List[Dict[str, Any]]:
        """Break up consecutive chorus chains and restore instrumentals.
        
        Args:
            sections: List of sections to process
            y: Audio signal
            sr: Sample rate
            bpm: BPM for bar calculation
            
        Returns:
            Processed sections with consecutive chorus chains broken
        """
        if len(sections) < 2:
            return sections
        
        processed = sections.copy()
        
        # Add vocal presence information to all sections
        for section in processed:
            vocal_ratio = self._detect_vocal_presence(y, sr, section['start_time'], section['end_time'])
            section['vocal_ratio'] = vocal_ratio
        
        # Find and break consecutive chorus chains
        i = 0
        while i < len(processed):
            # Look for consecutive chorus sections
            chorus_chain = []
            j = i
            while j < len(processed) and processed[j]['type'] == 'chorus':
                chorus_chain.append(j)
                j += 1
            
            # Process chains of 2 or more consecutive choruses
            if len(chorus_chain) >= 2:
                self._process_chorus_chain(processed, chorus_chain, bpm)
                i = j
            else:
                i += 1
        
        return processed

    def label_special_chorus_sections(
        self,
        sections: List[Dict[str, Any]],
        y: np.ndarray,
        sr: int,
        modulations: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Label drop chorus and last chorus modulation."""
        processed = sections.copy()

        energies = []
        for s in processed:
            start = int(s.get("start_time", 0) * sr)
            end = int(s.get("end_time", 0) * sr)
            seg = y[start:end]
            if len(seg) == 0:
                energies.append(-120.0)
            else:
                rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
                energies.append(20 * np.log10(rms))

        for idx, s in enumerate(processed):
            if s.get("type") == "chorus" and energies[idx] <= -30.0:
                if modulations and any(
                    m.get("time", 0) >= s.get("start_time", 0)
                    and m.get("time", 0) < s.get("end_time", 0)
                    for m in modulations
                ):
                    s["type"] = "drop_chorus"
                    s["ascii_label"] = self.JP_ASCII_LABELS.get(
                        "drop_chorus", "drop_chorus"
                    )

        if modulations:
            last_mod = modulations[-1]
            if processed and processed[-1].get("type") == "chorus":
                if last_mod.get("time", 0) >= processed[-1].get("start_time", 0):
                    processed[-1]["last_chorus_key"] = last_mod.get("to_key")

        return processed
    
    def _process_chorus_chain(self, sections: List[Dict[str, Any]],
                            chorus_indices: List[int], bpm: float) -> None:
        """Process a chain of consecutive chorus sections.
        
        Args:
            sections: List of all sections (modified in place)
            chorus_indices: Indices of consecutive chorus sections
            bpm: BPM for bar calculation
        """
        if len(chorus_indices) < 2:
            return
        
        # Calculate 8-bar duration for reference
        eight_bar_duration = (8 * 4 * 60.0) / bpm
        
        if len(chorus_indices) == 2:
            # B,B → check if second should be D based on energy and vocal presence
            second_idx = chorus_indices[1]
            section = sections[second_idx]
            
            # Convert to instrumental if:
            # 1. Low vocal presence (< 0.25) OR low energy (< 0.4)
            # 2. Duration is reasonable (4-16 bars)
            vocal_ratio = section.get('vocal_ratio', 0.5)
            energy_level = section.get('energy_level', 0.5)
            
            should_convert = (
                (vocal_ratio < 0.25 or energy_level < 0.4) and
                eight_bar_duration * 0.5 <= section['duration'] <= eight_bar_duration * 2
            )
            
            if should_convert:
                sections[second_idx]['type'] = 'instrumental'
                sections[second_idx]['ascii_label'] = self.JP_ASCII_LABELS['instrumental']
        
        elif len(chorus_indices) == 3:
            # B,B,B → convert middle to D (always for 3-chain)
            middle_idx = chorus_indices[1]
            section = sections[middle_idx]
            
            # Convert middle to instrumental if duration is reasonable
            if eight_bar_duration * 0.25 <= section['duration'] <= eight_bar_duration * 3:
                sections[middle_idx]['type'] = 'instrumental'
                sections[middle_idx]['ascii_label'] = self.JP_ASCII_LABELS['instrumental']
        
        elif len(chorus_indices) >= 4:
            # B,B,B,B+ → convert middle sections to D, keep max 2 consecutive B
            # Strategy: Keep first and last as chorus, convert middle ones to instrumental
            for i in range(1, len(chorus_indices) - 1):
                idx = chorus_indices[i]
                section = sections[idx]
                
                # Convert to instrumental if duration is reasonable
                if eight_bar_duration * 0.25 <= section['duration'] <= eight_bar_duration * 3:
                    sections[idx]['type'] = 'instrumental'
                    sections[idx]['ascii_label'] = self.JP_ASCII_LABELS['instrumental']
    
    def _detect_vocal_presence(self, y: np.ndarray, sr: int, start_time: float, end_time: float) -> float:
        """Detect vocal presence ratio in a given time segment.
        
        Args:
            y: Audio signal
            sr: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Vocal presence ratio (0.0 = no vocals, 1.0 = strong vocals)
        """
        import librosa
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]
        
        if len(segment) < sr:  # Less than 1 second
            return 0.5  # Default assumption
        
        # Simple vocal detection using spectral features
        # Extract spectral features
        stft = librosa.stft(segment, hop_length=512)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
        
        # Vocal characteristics:
        # - Higher spectral centroid (1000-4000 Hz range)
        # - Moderate zero crossing rate
        # - Spectral rolloff in vocal range
        
        # Normalize features
        avg_centroid = np.mean(spectral_centroids)
        avg_rolloff = np.mean(spectral_rolloff)
        avg_zcr = np.mean(zero_crossing_rate)
        
        # Vocal presence indicators
        vocal_score = 0.0
        
        # Spectral centroid in vocal range (1000-4000 Hz)
        if 1000 <= avg_centroid <= 4000:
            vocal_score += 0.4
        elif 500 <= avg_centroid <= 6000:
            vocal_score += 0.2
        
        # Zero crossing rate (vocals typically have moderate ZCR)
        if 0.05 <= avg_zcr <= 0.15:
            vocal_score += 0.3
        elif 0.02 <= avg_zcr <= 0.25:
            vocal_score += 0.1
        
        # Spectral rolloff (vocals have energy distributed across frequencies)
        if 2000 <= avg_rolloff <= 8000:
            vocal_score += 0.3
        elif 1000 <= avg_rolloff <= 12000:
            vocal_score += 0.1
        
        return min(1.0, vocal_score)