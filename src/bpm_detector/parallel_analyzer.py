"""Parallel audio analyzer with progress tracking and auto-optimization."""

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, cast

from .auto_parallel import AutoParallelConfig, ParallelConfig, PerformanceProfiler, SystemMonitor
from .music_analyzer import AudioAnalyzer
from .progress_manager import ProgressCallback, ProgressManager


def _analyze_single_file_worker(
    path: str, sr: int, hop_length: int, comprehensive: bool, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze one file in a spawned process.

    Keep this function at module scope so the process executor only needs to
    pickle simple configuration values.  A bound method would pickle the
    analyzer, including its monitor thread and other non-process state.
    """
    analyzer = AudioAnalyzer(sr=sr, hop_length=hop_length)
    return analyzer.analyze_file(path, comprehensive=comprehensive, **kwargs)


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
        if self._manual_max_workers is not None:
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
                f"   Memory budget (advisory): {self._parallel_config.memory_limit_mb}MB "
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
        analyze_rhythm: bool = False,
        analyze_chords: bool = False,
        analyze_structure: bool = False,
        analyze_timbre: bool = False,
        analyze_melody: bool = False,
        analyze_dynamics: bool = False,
        detect_modulation: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze audio file with smart parallelization."""
        if detect_modulation and not detect_key:
            raise ValueError("detect_modulation requires detect_key=True")

        return self._analyze_single_file(
            path,
            detect_key,
            comprehensive,
            min_bpm,
            max_bpm,
            start_bpm,
            progress_callback,
            analyze_rhythm,
            analyze_chords,
            analyze_structure,
            analyze_timbre,
            analyze_melody,
            analyze_dynamics,
            detect_modulation,
            **kwargs,
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
        analyze_rhythm: bool = False,
        analyze_chords: bool = False,
        analyze_structure: bool = False,
        analyze_timbre: bool = False,
        analyze_melody: bool = False,
        analyze_dynamics: bool = False,
        detect_modulation: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze single file with parallel processing."""

        # Check if any selective analysis is requested
        has_selective = (
            analyze_rhythm
            or analyze_chords
            or analyze_structure
            or analyze_timbre
            or analyze_melody
            or analyze_dynamics
            or detect_modulation
        )

        # Use regular (non-parallel) analysis if:
        # - Not comprehensive AND no selective analysis requested, OR
        # - Should not use parallel processing
        if (not comprehensive and not has_selective) or not self._should_use_parallel():
            base_kwargs = {}
            # ``detect_modulation`` is added by AudioAnalyzer in the core
            # workstream.  Keep the false default compatible with older
            # instances while forwarding an explicit request to that API.
            if detect_modulation:
                base_kwargs['detect_modulation'] = True

            base_progress_callback = None
            if progress_callback:
                # AudioAnalyzer reports one positional progress value.  The
                # Smart analyzer's public callback also receives a message.
                # Adapt at this boundary without changing the base contract.
                def adapted_progress_callback(progress):
                    progress_callback(progress, 'Processing...')

                base_progress_callback = adapted_progress_callback

            return super().analyze_file(
                path,
                detect_key,
                comprehensive,
                min_bpm,
                max_bpm,
                start_bpm,
                base_progress_callback,
                analyze_rhythm,
                analyze_chords,
                analyze_structure,
                analyze_timbre,
                analyze_melody,
                analyze_dynamics,
                **base_kwargs,
            )

        # Setup progress management with debug mode
        import os

        debug_mode = os.environ.get('BPM_DEBUG', '').lower() in ('1', 'true', 'yes')
        progress_manager = ProgressManager(debug=debug_mode)

        # Setup progress callback with safety measures
        if progress_callback:
            progress_dispatch_lock = threading.Lock()
            last_dispatched_progress = -float('inf')

            # Use simple callback for basic progress with improved calculation
            def safe_progress_callback(pm):
                nonlocal last_dispatched_progress

                # ProgressManager can notify from several analysis workers.
                # Read the snapshot and call user code under one per-analysis
                # lock so a slow callback cannot be overtaken by a later one.
                with progress_dispatch_lock:
                    callback_progress = 0.0
                    callback_message = 'Processing...'
                    try:
                        # Get overall progress with better calculation
                        overall_progress = pm.get_overall_progress()

                        # Apply scaling to account for initial setup (10%) and final processing (5%)
                        # The parallel analysis represents 85% of the total work
                        callback_progress = 10 + (overall_progress * 0.85)

                        running_tasks = pm.get_running_tasks()
                        if running_tasks:
                            callback_message = f"Running: {', '.join(running_tasks[:2])}"
                        else:
                            with pm._lock:
                                completed = sum(1 for t in pm._tasks.values() if t.status.value == 'completed')
                                total = len(pm._tasks)
                                callback_message = f"Completed: {completed}/{total} tasks"
                    except Exception:
                        # Metadata formatting is best effort.  User callback
                        # errors are deliberately outside this try block so
                        # they are never retried with a second invocation.
                        try:
                            callback_progress = pm.get_overall_progress()
                        except Exception:
                            callback_progress = 0.0

                    if callback_progress < last_dispatched_progress:
                        return
                    last_dispatched_progress = callback_progress
                    progress_callback(callback_progress, callback_message)

            progress_manager.add_callback(safe_progress_callback)

        try:
            # Load audio first
            if progress_callback:
                progress_callback(5, "Loading audio file...")

            y, sr = self._load_audio(path)

            if progress_callback:
                progress_callback(10, "Starting parallel analysis...")

            # Parallel comprehensive or selective analysis
            results = self._parallel_comprehensive_analysis(
                y,
                sr,
                path,
                progress_manager,
                detect_key=detect_key,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                start_bpm=start_bpm,
                progress_callback=progress_callback,
                comprehensive=comprehensive,
                analyze_rhythm=analyze_rhythm,
                analyze_chords=analyze_chords,
                analyze_structure=analyze_structure,
                analyze_timbre=analyze_timbre,
                analyze_melody=analyze_melody,
                analyze_dynamics=analyze_dynamics,
                detect_modulation=detect_modulation,
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

        if not paths:
            if progress_callback:
                progress_callback(100, 'All files completed!')
            return {}

        if kwargs.get('detect_modulation') and kwargs.get('detect_key', True) is False:
            raise ValueError("detect_modulation requires detect_key=True")

        # ``auto_parallel=False`` is an explicit sequential mode.  Do not
        # probe host resources here: doing so both changes behavior and can
        # unexpectedly create a process pool later in this method.
        if not self.auto_parallel:
            results = {}
            for i, path in enumerate(paths):
                path_str = str(path)
                if progress_callback:
                    progress_callback(int(100 * i / len(paths)), f"Processing {path_str} ({i+1}/{len(paths)})")
                results[path_str] = self.analyze_file(path_str, comprehensive=comprehensive, **kwargs)

            if progress_callback:
                progress_callback(100, "All files completed!")
            return results

        # Adjust configuration for multiple files
        base_config = self._parallel_config
        if base_config is None:
            # A failed or deferred auto setup must fail closed instead of
            # probing the host a second time from the batch path.
            adjusted_config = ParallelConfig(enable_parallel=False, max_workers=1)
        else:
            adjusted_config = AutoParallelConfig.get_file_count_adjustment(len(paths), base_config)

        # File-count tuning can raise the worker count for large batches.  A
        # manual setting remains a hard ceiling across every batch shape.
        if self._manual_max_workers is not None:
            adjusted_config.max_workers = min(adjusted_config.max_workers, self._manual_max_workers)
            adjusted_config.max_workers = max(1, adjusted_config.max_workers)

        if not adjusted_config.enable_parallel:
            # Sequential processing
            results = {}
            for i, path in enumerate(paths):
                path_str = str(path)
                if progress_callback:
                    progress_callback(int(100 * i / len(paths)), f"Processing {path_str} ({i+1}/{len(paths)})")
                results[path_str] = self.analyze_file(path_str, comprehensive=comprehensive, **kwargs)

            if progress_callback:
                progress_callback(100, "All files completed!")
            return results

        # Parallel processing
        return self._parallel_analyze_multiple_files(paths, comprehensive, adjusted_config, progress_callback, **kwargs)

    def _analyze_batch_file(self, path: str, comprehensive: bool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze one batch file through the base path.

        The outer batch executor owns file-level concurrency.  Calling the
        base implementation directly keeps a file from creating another
        per-feature executor and avoids changing shared analyzer configuration
        while sibling files are running.
        """
        profile_data = self.performance_profiler.start_profiling('file_analysis')
        try:
            return AudioAnalyzer.analyze_file(self, path, comprehensive=comprehensive, **kwargs)
        finally:
            self.performance_profiler.end_profiling(profile_data)

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
        progress_callback=None,
        comprehensive: bool = True,
        analyze_rhythm: bool = False,
        analyze_chords: bool = False,
        analyze_structure: bool = False,
        analyze_timbre: bool = False,
        analyze_melody: bool = False,
        analyze_dynamics: bool = False,
        detect_modulation: bool = False,
    ) -> Dict[str, Any]:
        """Perform parallel comprehensive or selective analysis."""

        if detect_modulation and not detect_key:
            raise ValueError("detect_modulation requires detect_key=True")

        # Determine which analyses to run
        run_chords = comprehensive or analyze_chords
        run_structure = comprehensive or analyze_structure
        run_rhythm = comprehensive or analyze_rhythm
        run_timbre = comprehensive or analyze_timbre
        run_melody = comprehensive or analyze_melody
        run_dynamics = comprehensive or analyze_dynamics

        # Register analysis tasks based on what's requested
        analysis_tasks = [('basic_info', 'Basic Analysis')]

        if run_chords:
            analysis_tasks.append(('chord_progression', 'Chord Progression'))
        if run_structure:
            analysis_tasks.append(('structure', 'Structure Analysis'))
        if run_rhythm:
            analysis_tasks.append(('rhythm', 'Rhythm Analysis'))
        if run_timbre:
            analysis_tasks.append(('timbre', 'Timbre Analysis'))
        if run_melody:
            analysis_tasks.append(('melody_harmony', 'Melody & Harmony'))
        if run_dynamics:
            analysis_tasks.append(('dynamics', 'Dynamics Analysis'))
        if detect_modulation:
            analysis_tasks.append(('modulation_analysis', 'Modulation Analysis'))

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

        # Parallel analysis execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # Only submit tasks that are requested
            if run_chords:
                futures['chord_progression'] = executor.submit(
                    self._analyze_with_progress,
                    'chord_progression',
                    progress_manager,
                    self.chord_analyzer.analyze,
                    y,
                    sr,
                    key,
                    float(basic_info.get('bpm', 130.0)),
                )

            if run_structure:
                futures['structure'] = executor.submit(
                    self._analyze_with_progress,
                    'structure',
                    progress_manager,
                    self._analyze_structure_with_progress,
                    y,
                    sr,
                    float(basic_info.get('bpm', 130.0)),
                )

            if run_rhythm:
                futures['rhythm'] = executor.submit(
                    self._analyze_with_progress, 'rhythm', progress_manager, self.rhythm_analyzer.analyze, y, sr
                )

            if run_timbre:
                futures['timbre'] = executor.submit(
                    self._analyze_with_progress, 'timbre', progress_manager, self.timbre_analyzer.analyze, y, sr
                )

            if run_melody:
                futures['melody_harmony'] = executor.submit(
                    self._analyze_with_progress,
                    'melody_harmony',
                    progress_manager,
                    self.melody_harmony_analyzer.analyze,
                    y,
                    sr,
                )

            if run_dynamics:
                futures['dynamics'] = executor.submit(
                    self._analyze_with_progress, 'dynamics', progress_manager, self.dynamics_analyzer.analyze, y, sr
                )

            if detect_modulation:
                futures['modulation_analysis'] = executor.submit(
                    self._analyze_with_progress,
                    'modulation_analysis',
                    progress_manager,
                    self.key_detector.compute_modulation_timeseries,
                    y,
                    sr,
                    bpm=float(basic_info.get('bpm', 130.0)),
                )

            # Collect results
            results = {'basic_info': basic_info}

            for task_id, future in futures.items():
                try:
                    results[task_id] = future.result(timeout=None)  # No timeout
                except Exception as e:
                    print(f"❌ Warning: Error in {task_id} analysis: {e}")
                    progress_manager.complete_task(task_id, False, str(e))
                    results[task_id] = {}

        # Generate additional features with progress updates (only in comprehensive mode)
        if comprehensive:
            try:
                # Update progress for final processing
                if progress_callback:
                    progress_callback(95, "Generating similarity features...")

                feature_vector = self.similarity_engine.extract_feature_vector(results)
                results["similarity_features"] = {
                    "feature_vector": feature_vector.tolist(),
                    "feature_weights": self.similarity_engine.feature_weights,
                }

                if progress_callback:
                    progress_callback(97, "Generating reference tags...")
                results["reference_tags"] = self._generate_reference_tags(results)

                if progress_callback:
                    progress_callback(99, "Finalizing analysis...")
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

        if not paths:
            if progress_callback:
                progress_callback(100, 'All files completed!')
            return {}

        if not self.auto_parallel:
            results = {}
            for index, path in enumerate(paths, start=1):
                path_str = str(path)
                results[path_str] = self.analyze_file(path_str, comprehensive=comprehensive, **kwargs)
                if progress_callback:
                    progress_callback(int(100 * index / len(paths)), f"Completed {index}/{len(paths)} files")
            return results

        max_workers = max(1, int(config.max_workers))
        if self.system_monitor.monitoring:
            max_workers = max(1, self.system_monitor.get_recommended_workers(max_workers))
        if self._manual_max_workers is not None:
            max_workers = max(1, min(max_workers, self._manual_max_workers))

        if config.use_process_pool:
            # Process parallelization for multiple files
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for path in paths:
                    # Ensure path is a string
                    path_str = str(path)
                    futures[path_str] = executor.submit(
                        _analyze_single_file_worker, path_str, self.sr, self.hop_length, comprehensive, dict(kwargs)
                    )

                results = {}
                completed = 0

                for path, future in futures.items():
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        results[path] = {'error': str(e)}
                    finally:
                        completed += 1
                        if progress_callback:
                            progress_callback(
                                int(100 * completed / len(paths)), f"Completed {completed}/{len(paths)} files"
                            )

                return results
        else:
            # Thread parallelization
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for path in paths:
                    # Ensure path is a string
                    path_str = str(path)
                    futures[path_str] = executor.submit(self._analyze_batch_file, path_str, comprehensive, dict(kwargs))

                results = {}
                completed = 0

                for path, future in futures.items():
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        results[path] = {'error': str(e)}
                    finally:
                        completed += 1
                        if progress_callback:
                            progress_callback(
                                int(100 * completed / len(paths)), f"Completed {completed}/{len(paths)} files"
                            )

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

                # Signature lookup can fail for extension/builtin callables,
                # but an error from the DSP call itself must not be retried.
                import inspect

                try:
                    sig = inspect.signature(analyzer_func)
                except Exception:
                    result = analyzer_func(*args, **kwargs)
                else:
                    if 'progress_callback' in sig.parameters:
                        result = analyzer_func(*args, progress_callback=callback, **kwargs)
                    else:
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
            callback(90, "Key detection completed")
        else:
            callback(90, "Key detection skipped")

        callback(100, "Basic analysis completed")

        basic_info = {
            "filename": path,
            "duration": duration,
            "bpm": bpm,
            "bpm_confidence": bpm_conf,
            "bpm_candidates": list(zip(top_bpms, top_hits)),
            "key": key,
            "key_confidence": key_conf,
        }
        if key_detection_result:
            basic_info["key_detection_details"] = key_detection_result
        return basic_info

    def _analyze_structure_with_progress(self, y, sr, callback, bpm=130.0):
        """Structure analysis with detailed progress."""
        callback(5, "Starting structure analysis...")
        callback(20, "Running complete structure analysis...")
        # Run the canonical structure analysis once.  The former helper
        # rebuilt a partial result and repeated feature extraction before
        # adding the section-level chord annotations.
        result = self.structure_analyzer.analyze(y, sr, bpm)
        # Preserve the shipped section-level chord annotations while keeping
        # every field returned by the canonical structure analyzer.
        if result.get('sections'):
            result = dict(result)
            result['sections'] = self._analyze_section_chord_progressions(y, sr, result['sections'], bpm=bpm)
        callback(95, "Structure analysis completed")
        return result

    def _analyze_chords_with_progress(self, y, sr, key, bpm, callback):
        """Chord analysis with detailed progress."""
        callback(10, "Starting chord analysis...")

        callback(20, "Running complete chord analysis...")
        # Keep this call identical to AudioAnalyzer's base path.  Rebuilding
        # the result from individual helper calls silently drops schema keys
        # and can use a different default BPM.
        result = self.chord_analyzer.analyze(y, sr, key or "C Major", bpm=bpm)
        callback(95, "Progression analysis completed")
        return result

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

        base_workers = max(1, int(self._parallel_config.max_workers))

        if self.system_monitor.monitoring:
            recommended_workers = self.system_monitor.get_recommended_workers(base_workers)
            if self._manual_max_workers is not None:
                recommended_workers = min(recommended_workers, self._manual_max_workers)
            return max(1, recommended_workers)

        if self._manual_max_workers is not None:
            return max(1, min(base_workers, self._manual_max_workers))

        return base_workers

    def _get_progress_message(self, progress_manager: ProgressManager) -> str:
        """Generate progress message."""
        status = progress_manager.get_status_summary()
        running_tasks = progress_manager.get_running_tasks()

        if running_tasks:
            return f"Running: {', '.join(running_tasks[:2])}"
        else:
            return f"Completed: {status['completed']}/{status['total']} tasks"

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
