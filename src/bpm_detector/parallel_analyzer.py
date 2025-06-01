"""Parallel audio analyzer with progress tracking and auto-optimization."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .auto_parallel import AutoParallelConfig, ParallelConfig, PerformanceProfiler, SystemMonitor
from .music_analyzer import AudioAnalyzer
from .progress_manager import ProgressCallback, ProgressManager


class SmartParallelAudioAnalyzer(AudioAnalyzer):
    """Smart parallel audio analyzer with automatic optimization."""

    def __init__(self, auto_parallel: bool = True, max_workers: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.auto_parallel = auto_parallel
        self.system_monitor = SystemMonitor()
        self.performance_profiler = PerformanceProfiler()
        self._parallel_config: Optional[ParallelConfig] = None
        self._manual_max_workers = max_workers

        if auto_parallel:
            self._configure_auto_parallel()

    def _configure_auto_parallel(self):
        """Configure automatic parallelization."""
        self._parallel_config = AutoParallelConfig.get_optimal_config()

        # Override with manual setting if provided
        if self._manual_max_workers:
            self._parallel_config.max_workers = self._manual_max_workers
            self._parallel_config.reason += f" (manual override: {self._manual_max_workers} workers)"

        if self._parallel_config.enable_parallel:
            # Calculate physical memory info
            import psutil

            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_limit_gb = self._parallel_config.memory_limit_mb / 1024
            memory_percentage = (memory_limit_gb / total_memory_gb) * 100

            print(f"🚀 Auto-parallel enabled: {self._parallel_config.reason}")
            print(f"   Workers: {self._parallel_config.max_workers}")
            print(
                f"   Memory limit: {self._parallel_config.memory_limit_mb}MB "
                f"({memory_limit_gb:.1f}GB, {memory_percentage:.0f}% of {total_memory_gb:.1f}GB physical)"
            )
            print(f"   Strategy: {self._parallel_config.strategy.value}")

            # Start load monitoring
            self.system_monitor.start_monitoring()
        else:
            print(f"⚡ Sequential processing: {self._parallel_config.reason}")

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
        """Analyze audio file with smart parallelization."""
        return self._analyze_single_file(
            path, detect_key, comprehensive, min_bpm, max_bpm, start_bpm, progress_callback
        )

    def analyze_files(
        self,
        paths: List[str],
        comprehensive: bool = True,
        progress_callback: Optional[Callable] = None,
        progress_display=None,
        detailed_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple audio files with smart parallelization."""
        return self._analyze_multiple_files(
            paths, comprehensive, progress_callback, progress_display, detailed_progress, **kwargs
        )

    def _analyze_single_file(
        self,
        path: str,
        detect_key: bool = True,
        comprehensive: bool = True,
        min_bpm: float = 40.0,
        max_bpm: float = 300.0,
        start_bpm: float = 150.0,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Analyze single file with parallel processing."""

        if not comprehensive or not self._should_use_parallel():
            return super().analyze_file(path, detect_key, comprehensive, min_bpm, max_bpm, start_bpm, progress_callback)

        # Setup progress management
        progress_manager = ProgressManager()

        # Setup progress callback with safety measures
        if progress_callback:
            # Use simple callback for basic progress
            def safe_progress_callback(pm):
                try:
                    overall_progress = pm.get_overall_progress()
                    running_tasks = pm.get_running_tasks()
                    if running_tasks:
                        message = f"Running: {', '.join(running_tasks[:2])}"
                    else:
                        with pm._lock:
                            completed = sum(1 for t in pm._tasks.values() if t.status.value == "completed")
                            total = len(pm._tasks)
                            message = f"Completed: {completed}/{total} tasks"
                    progress_callback(overall_progress, message)
                except Exception:
                    # Fallback to simple message
                    progress_callback(pm.get_overall_progress(), "Processing...")

            progress_manager.add_callback(safe_progress_callback)

        try:
            # Load audio first
            if progress_callback:
                progress_callback(5, "Loading audio file...")

            import librosa

            y, sr = librosa.load(
                path,
                sr=self.sr,
                mono=True,
                dtype=np.float32,  # Use float32 for better memory efficiency
                res_type='kaiser_fast',  # Faster resampling
            )
            sr = int(sr)  # Ensure sr is int for type checking

            if progress_callback:
                progress_callback(10, "Starting parallel analysis...")

            # Parallel comprehensive analysis
            results = self._parallel_comprehensive_analysis(
                y,
                sr,
                path,
                progress_manager,
                detect_key=detect_key,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                start_bpm=start_bpm,
            )

            # Clear audio data from memory early
            del y
            import gc

            gc.collect()

            if progress_callback:
                progress_callback(100, "Analysis completed!")

            return results

        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {e}")
            raise
        finally:
            progress_manager.reset()

    def _analyze_multiple_files(
        self,
        paths: List[str],
        comprehensive: bool = True,
        progress_callback: Optional[Callable] = None,
        progress_display=None,
        detailed_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple files with parallel processing."""

        # Adjust configuration for multiple files
        if self._parallel_config:
            adjusted_config = AutoParallelConfig.get_file_count_adjustment(len(paths), self._parallel_config)
        else:
            adjusted_config = AutoParallelConfig.get_optimal_config()

        if not adjusted_config.enable_parallel:
            # Sequential processing
            results = {}
            for i, path in enumerate(paths):
                if progress_callback:
                    progress_callback(int(100 * i / len(paths)), f"Processing {path} ({i+1}/{len(paths)})")
                results[path] = self.analyze_file(path, **kwargs)

            if progress_callback:
                progress_callback(100, "All files completed!")
            return results

        # Parallel processing
        return self._parallel_analyze_multiple_files(paths, comprehensive, adjusted_config, progress_callback, **kwargs)

    def _parallel_comprehensive_analysis(
        self,
        y,
        sr,
        path: str,
        progress_manager: ProgressManager,
        detect_key: bool = True,
        min_bpm: float = 40.0,
        max_bpm: float = 300.0,
        start_bpm: float = 150.0,
    ) -> Dict[str, Any]:
        """Perform parallel comprehensive analysis."""

        # Register analysis tasks
        analysis_tasks = [
            ('basic_info', 'Basic Analysis'),
            ('chord_progression', 'Chord Progression'),
            ('structure', 'Structure Analysis'),
            ('rhythm', 'Rhythm Analysis'),
            ('timbre', 'Timbre Analysis'),
            ('melody_harmony', 'Melody & Harmony'),
            ('dynamics', 'Dynamics Analysis'),
        ]

        for task_id, name in analysis_tasks:
            progress_manager.register_task(task_id, name)

        # Basic analysis first (needed by other analyzers)
        basic_callback = ProgressCallback(progress_manager, 'basic_info')
        basic_info = self._analyze_basic_info_with_progress(
            y, sr, path, basic_callback, detect_key=detect_key, min_bpm=min_bpm, max_bpm=max_bpm, start_bpm=start_bpm
        )
        basic_callback.complete()

        key = basic_info.get('key')
        max_workers = self._get_current_max_workers()

        # Use more aggressive parallelization for better performance
        effective_workers = min(max_workers * 3, 24)  # Triple the workers for analysis tasks

        # Parallel analysis execution
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                'chord_progression': executor.submit(
                    self._analyze_with_progress,
                    'chord_progression',
                    progress_manager,
                    self.chord_analyzer.analyze,
                    y,
                    sr,
                    key,
                    float(basic_info.get('bpm', 130.0)),
                ),
                'structure': executor.submit(
                    self._analyze_with_progress,
                    'structure',
                    progress_manager,
                    self._analyze_structure_with_progress,
                    y,
                    sr,
                    float(basic_info.get('bpm', 130.0)),
                ),
                'rhythm': executor.submit(
                    self._analyze_with_progress, 'rhythm', progress_manager, self.rhythm_analyzer.analyze, y, sr
                ),
                'timbre': executor.submit(
                    self._analyze_with_progress, 'timbre', progress_manager, self.timbre_analyzer.analyze, y, sr
                ),
                'melody_harmony': executor.submit(
                    self._analyze_with_progress,
                    'melody_harmony',
                    progress_manager,
                    self.melody_harmony_analyzer.analyze,
                    y,
                    sr,
                ),
                'dynamics': executor.submit(
                    self._analyze_with_progress, 'dynamics', progress_manager, self.dynamics_analyzer.analyze, y, sr
                ),
            }

            # Collect results
            results = {'basic_info': basic_info}

            for task_id, future in futures.items():
                try:
                    results[task_id] = future.result(timeout=None)  # No timeout
                except Exception as e:
                    print(f"❌ Warning: Error in {task_id} analysis: {e}")
                    progress_manager.complete_task(task_id, False, str(e))
                    results[task_id] = {}

        # Generate additional features
        try:
            feature_vector = self.similarity_engine.extract_feature_vector(results)
            results["similarity_features"] = {
                "feature_vector": feature_vector.tolist(),
                "feature_weights": self.similarity_engine.feature_weights,
            }

            results["reference_tags"] = self._generate_reference_tags(results)
            results["production_notes"] = self._generate_production_notes(results)

        except Exception as e:
            print(f"Warning: Error generating additional features: {e}")

        return results

    def _parallel_analyze_multiple_files(
        self,
        paths: List[str],
        comprehensive: bool,
        config: ParallelConfig,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Parallel analysis of multiple files."""

        max_workers = config.max_workers

        if config.use_process_pool:
            # Process parallelization for multiple files
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    path: executor.submit(self._analyze_single_file_worker, path, comprehensive, **kwargs)
                    for path in paths
                }

                results = {}
                completed = 0

                for path, future in futures.items():
                    try:
                        results[path] = future.result()
                        completed += 1

                        if progress_callback:
                            progress_callback(
                                int(100 * completed / len(paths)), f"Completed {completed}/{len(paths)} files"
                            )
                    except Exception as e:
                        results[path] = {'error': str(e)}
                        completed += 1

                return results
        else:
            # Thread parallelization
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {path: executor.submit(self.analyze_file, path, **kwargs) for path in paths}

                results = {}
                completed = 0

                for path, future in futures.items():
                    try:
                        results[path] = future.result()
                        completed += 1

                        if progress_callback:
                            progress_callback(
                                int(100 * completed / len(paths)), f"Completed {completed}/{len(paths)} files"
                            )
                    except Exception as e:
                        results[path] = {'error': str(e)}
                        completed += 1

                return results

    def _analyze_with_progress(
        self, task_id: str, progress_manager: ProgressManager, analyzer_func: Callable, *args, **kwargs
    ):
        """Execute analysis with progress tracking."""
        callback = ProgressCallback(progress_manager, task_id)
        profile_data = self.performance_profiler.start_profiling(task_id)

        try:
            callback(0, "Starting...")

            # Special handling for detailed progress analyzers
            if task_id == 'structure':
                # Extract arguments: y, sr, bpm
                y, sr, bpm = args
                result = self._analyze_structure_with_progress(y, sr, callback, bpm)
            elif task_id == 'chord_progression':
                # Extract arguments: y, sr, key, bpm
                y, sr, key, bpm = args
                result = self._analyze_chords_with_progress(y, sr, key, bpm, callback)
            else:
                # Enhanced progress for other analyzers with callback injection
                callback(10, "Initializing...")
                callback(25, "Processing audio...")

                # Try to inject progress callback into analyzer function
                try:
                    # Check if analyzer accepts progress_callback parameter
                    import inspect

                    sig = inspect.signature(analyzer_func)
                    if 'progress_callback' in sig.parameters:
                        result = analyzer_func(*args, progress_callback=callback, **kwargs)
                    else:
                        result = analyzer_func(*args, **kwargs)
                        callback(75, "Processing completed...")
                except Exception:
                    # Fallback to original call
                    result = analyzer_func(*args, **kwargs)
                    callback(75, "Processing completed...")

                callback(90, "Finalizing results...")

            callback.complete(True)
            self.performance_profiler.end_profiling(profile_data)
            return result

        except Exception as e:
            callback.complete(False, str(e))
            self.performance_profiler.end_profiling(profile_data)
            raise

    def _analyze_basic_info_with_progress(
        self,
        y,
        sr,
        path,
        callback,
        detect_key: bool = True,
        min_bpm: float = 40.0,
        max_bpm: float = 300.0,
        start_bpm: float = 150.0,
    ):
        """Basic analysis with progress tracking."""
        callback(5, "Starting basic analysis...")

        callback(10, "Calculating duration...")
        duration = len(y) / sr
        callback(15, "Analyzing BPM...")

        bpm, bpm_conf, top_bpms, top_hits = self.bpm_detector.detect(y, sr, min_bpm, max_bpm, start_bpm)
        callback(50, "BPM analysis completed")

        callback(60, "Detecting key...")

        key = None
        key_conf = 0.0
        key_detection_result = None
        if detect_key:
            # Use the improved key detection from new KeyDetector
            key_detection_result = self.key_detector.detect_key(y, sr)
            # Check for valid key detection (not Unknown format)
            if not key_detection_result['key'].startswith('Unknown'):
                key = f"{key_detection_result['key']} {key_detection_result['mode']}"
                key_conf = key_detection_result['confidence']  # Already 0-100 scale
            else:
                # Fallback to melody_harmony_analyzer
                fallback_result = self.melody_harmony_analyzer.detect_key(y, sr)
                key = f"{fallback_result['key']} {fallback_result['mode']}"
                key_conf = fallback_result['confidence']
            callback(90, "Key detection completed")
        else:
            callback(90, "Key detection skipped")

        callback(100, "Basic analysis completed")

        return {
            "filename": path,
            "duration": duration,
            "bpm": bpm,
            "bpm_confidence": bpm_conf,
            "bpm_candidates": list(zip(top_bpms, top_hits)),
            "key": key,
            "key_confidence": key_conf,
        }

    def _analyze_structure_with_progress(self, y, sr, callback, bpm=130.0):
        """Structure analysis with detailed progress."""
        callback(5, "Starting structure analysis...")

        callback(10, "Extracting features...")
        features = self.structure_analyzer.extract_structural_features(y, sr)
        callback(25, "Features extracted")

        callback(30, "Computing similarity matrix...")
        similarity_matrix = self.structure_analyzer.compute_self_similarity_matrix(features)
        callback(50, "Similarity matrix computed")

        callback(60, "Detecting boundaries...")
        self.structure_analyzer.detect_boundaries(similarity_matrix, sr, bpm=bpm)
        callback(75, "Boundaries detected")

        callback(80, "Analyzing structure...")
        structure_result = self.structure_analyzer.analyze(y, sr, bpm)
        sections = structure_result['sections']
        callback(90, "Structure analyzed")

        callback(90, "Using form analysis from structure result...")
        # Use form analysis from the complete structure analysis
        form_analysis = {
            'form': structure_result['form'],
            'repetition_ratio': structure_result['repetition_ratio'],
            'structural_complexity': structure_result['structural_complexity'],
            'section_count': structure_result['section_count'],
            'unique_sections': structure_result['unique_sections'],
        }

        callback(95, "Analyzing section-wise chord progressions...")
        # Enhance sections with chord progression information
        enhanced_sections = self._analyze_section_chord_progressions(y, sr, sections)

        callback(98, "Form analysis completed")

        return {
            'sections': enhanced_sections,
            'form': form_analysis['form'],
            'repetition_ratio': form_analysis['repetition_ratio'],
            'structural_complexity': form_analysis['structural_complexity'],
            'section_count': form_analysis['section_count'],
            'unique_sections': form_analysis.get('unique_sections', 0),
        }

    def _analyze_section_chord_progressions(self, y, sr, sections):
        """Analyze chord progressions for each section."""
        enhanced_sections = []

        for section in sections:
            # Extract audio segment for this section
            start_sample = int(section['start_time'] * sr)
            end_sample = int(section['end_time'] * sr)
            segment = y[start_sample:end_sample]

            if len(segment) > sr:  # Only analyze segments longer than 1 second
                try:
                    # Extract chroma features for this segment
                    chroma = self.chord_analyzer.extract_chroma_features(segment, sr)

                    # Detect chords for this segment
                    segment_chords = self.chord_analyzer.detect_chords(chroma)

                    # Get the most common chord progression for this section
                    if segment_chords:
                        # Take the most frequent chords (simplified)
                        unique_chords = list(set(segment_chords))[:4]  # Max 4 chords per section
                        section_progression = (
                            ' → '.join(str(chord) for chord in unique_chords) if unique_chords else 'Unknown'
                        )
                    else:
                        section_progression = 'Unknown'

                except Exception:
                    section_progression = 'Unknown'
            else:
                section_progression = 'Unknown'

            # Add chord progression to section info
            enhanced_section = section.copy()
            enhanced_section['chord_progression'] = section_progression
            enhanced_sections.append(enhanced_section)

        return enhanced_sections

    def _analyze_chords_with_progress(self, y, sr, key, bpm, callback):
        """Chord analysis with detailed progress."""
        callback(10, "Starting chord analysis...")

        callback(20, "Extracting chroma features...")
        chroma = self.chord_analyzer.extract_chroma_features(y, sr)
        callback(40, "Chroma features extracted")

        callback(50, "Detecting chords...")
        chords = self.chord_analyzer.detect_chords(chroma, bpm)
        callback(70, "Chords detected")

        callback(80, "Analyzing progression...")
        progression_analysis = self.chord_analyzer.analyze_progression(chords)
        callback(95, "Progression analysis completed")

        return {
            'chords': chords,
            'main_progression': progression_analysis['main_progression'],
            'chord_complexity': progression_analysis['chord_complexity'],
            'harmonic_rhythm': progression_analysis['harmonic_rhythm'],
        }

    def _should_use_parallel(self) -> bool:
        """Check if parallel processing should be used."""
        if not self.auto_parallel or not self._parallel_config or not self._parallel_config.enable_parallel:
            return False

        # Dynamic load check
        if self.system_monitor.monitoring:
            return not self.system_monitor.should_reduce_parallelism()

        return True

    def _get_current_max_workers(self) -> int:
        """Get current recommended worker count."""
        if not self._parallel_config:
            return 1

        base_workers = self._parallel_config.max_workers

        if self.system_monitor.monitoring:
            return self.system_monitor.get_recommended_workers(base_workers)

        return base_workers

    def _get_progress_message(self, progress_manager: ProgressManager) -> str:
        """Generate progress message."""
        status = progress_manager.get_status_summary()
        running_tasks = progress_manager.get_running_tasks()

        if running_tasks:
            return f"Running: {', '.join(running_tasks[:2])}"
        else:
            return f"Completed: {status['completed']}/{status['total']} tasks"

    def _analyze_single_file_worker(self, path: str, comprehensive: bool, **kwargs):
        """Worker function for process-based parallelization."""
        # Create new analyzer instance for process isolation
        analyzer = AudioAnalyzer(sr=self.sr, hop_length=self.hop_length)
        return analyzer.analyze_file(path, comprehensive=comprehensive, **kwargs)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance profiling summary."""
        return self.performance_profiler.get_performance_summary()

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'system_monitor') and self.system_monitor:
                self.system_monitor.stop_monitoring()
        except Exception:
            # Ignore errors during cleanup
            pass

    def cleanup(self):
        """Explicit cleanup method for graceful shutdown."""
        try:
            if hasattr(self, 'system_monitor') and self.system_monitor:
                self.system_monitor.stop_monitoring()
        except Exception:
            pass
