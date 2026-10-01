"""Tests for parallel analyzer functionality."""

import os
import shutil
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.bpm_detector.auto_parallel import AutoParallelConfig, ParallelConfig, SystemMonitor
from src.bpm_detector.music_analyzer import AudioAnalyzer
from src.bpm_detector.parallel_analyzer import SmartParallelAudioAnalyzer, _analyze_single_file_worker
from src.bpm_detector.progress_manager import ProgressManager, TaskStatus

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


class TestAutoParallelConfig(unittest.TestCase):
    """Test automatic parallel configuration."""

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_high_performance_config(self, mock_psutil, mock_cpu_count):
        """Test configuration for high-performance systems."""
        mock_cpu_count.return_value = 12
        mock_psutil.cpu_count.return_value = 6
        mock_psutil.virtual_memory.return_value.available = 16 * 1024**3  # 16GB
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value.percent = 50.0

        config = AutoParallelConfig.get_optimal_config()

        self.assertTrue(config.enable_parallel)
        self.assertEqual(config.max_workers, 10)  # 12 - 2
        self.assertTrue(config.use_process_pool)
        self.assertGreater(config.memory_limit_mb, 1000)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_low_performance_config(self, mock_psutil, mock_cpu_count):
        """Test configuration for low-performance systems."""
        mock_cpu_count.return_value = 2
        mock_psutil.cpu_count.return_value = 2
        mock_psutil.virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        mock_psutil.cpu_percent.return_value = 80.0
        mock_psutil.virtual_memory.return_value.percent = 75.0

        config = AutoParallelConfig.get_optimal_config()

        self.assertFalse(config.enable_parallel)
        self.assertEqual(config.max_workers, 1)

    def test_file_count_adjustment(self):
        """Test configuration adjustment based on file count."""
        base_config = ParallelConfig(enable_parallel=True, max_workers=8, use_process_pool=False)

        # Single file
        single_config = AutoParallelConfig.get_file_count_adjustment(1, base_config)
        self.assertLessEqual(single_config.max_workers, 6)

        # Many files
        many_config = AutoParallelConfig.get_file_count_adjustment(20, base_config)
        self.assertTrue(many_config.use_process_pool)
        self.assertGreater(many_config.max_workers, base_config.max_workers)


class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = SystemMonitor(check_interval=0.1)
        self.assertFalse(monitor.monitoring)
        self.assertEqual(monitor.current_load['cpu'], 0)
        self.assertEqual(monitor.current_load['memory'], 0)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_load_detection(self, mock_psutil):
        """Test load detection logic."""
        monitor = SystemMonitor()

        # High load scenario
        monitor.current_load = {'cpu': 95, 'memory': 90}
        self.assertTrue(monitor.should_reduce_parallelism())

        # Low load scenario
        monitor.current_load = {'cpu': 30, 'memory': 40}
        self.assertFalse(monitor.should_reduce_parallelism())

    def test_worker_recommendation(self):
        """Test worker count recommendation."""
        monitor = SystemMonitor()

        # High load - should reduce
        monitor.current_load = {'cpu': 95, 'memory': 90}
        recommended = monitor.get_recommended_workers(8)
        self.assertEqual(recommended, 4)  # Half of current

        # Low load - should increase
        monitor.current_load = {'cpu': 30, 'memory': 40}
        recommended = monitor.get_recommended_workers(4)
        self.assertEqual(recommended, 5)  # Increase by 1


class TestProgressManager(unittest.TestCase):
    """Test progress management functionality."""

    def test_task_registration(self):
        """Test task registration."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        tasks = manager.get_task_details()
        self.assertIn("test_task", tasks)
        self.assertEqual(tasks["test_task"].name, "Test Task")
        self.assertEqual(tasks["test_task"].status, TaskStatus.PENDING)

    def test_progress_updates(self):
        """Test progress updates."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        # Update progress
        manager.update_progress("test_task", 50.0, "Processing...")

        tasks = manager.get_task_details()
        task = tasks["test_task"]
        self.assertEqual(task.progress, 50.0)
        self.assertEqual(task.message, "Processing...")
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task.start_time)

    def test_task_completion(self):
        """Test task completion."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        # Complete task
        manager.complete_task("test_task", True)

        tasks = manager.get_task_details()
        task = tasks["test_task"]
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.progress, 100.0)
        self.assertIsNotNone(task.end_time)

    def test_overall_progress(self):
        """Test overall progress calculation."""
        manager = ProgressManager()
        manager.register_task("task1", "Task 1")
        manager.register_task("task2", "Task 2")

        manager.update_progress("task1", 100.0)
        manager.update_progress("task2", 50.0)

        overall = manager.get_overall_progress()
        self.assertEqual(overall, 75.0)  # (100 + 50) / 2

    def test_status_summary(self):
        """Test status summary."""
        manager = ProgressManager()
        manager.register_task("task1", "Task 1")
        manager.register_task("task2", "Task 2")
        manager.register_task("task3", "Task 3")

        manager.complete_task("task1", True)
        manager.update_progress("task2", 50.0)
        manager.complete_task("task3", False)

        summary = manager.get_status_summary()
        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['running'], 1)
        self.assertEqual(summary['failed'], 1)


class TestSmartParallelAudioAnalyzer(unittest.TestCase):
    """Test smart parallel audio analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.wav")

        # Generate test audio (1 second of sine wave)
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        sf.write(self.test_file, audio, sr)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_analyzer_initialization(self, mock_psutil, mock_cpu_count):
        """Test analyzer initialization."""
        mock_cpu_count.return_value = 8
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 50.0

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        self.assertIsNotNone(analyzer._parallel_config)
        if analyzer._parallel_config is not None:
            self.assertTrue(analyzer._parallel_config.enable_parallel)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_parallel_disabled(self, mock_psutil, mock_cpu_count):
        """Test analyzer with parallel processing disabled."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        self.assertIsNone(analyzer._parallel_config)

    def test_manual_worker_override(self):
        """Test manual worker count override."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=4)

        if analyzer._parallel_config:
            self.assertEqual(analyzer._parallel_config.max_workers, 4)

    def test_missing_file_raises_file_not_found(self):
        """A path that does not exist surfaces as FileNotFoundError."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        with self.assertRaises(FileNotFoundError):
            analyzer.analyze_file('non_existent_file.wav')

    @patch('src.bpm_detector.parallel_analyzer.SmartParallelAudioAnalyzer._should_use_parallel')
    def test_fallback_to_sequential(self, mock_should_use_parallel):
        """Test fallback to sequential processing."""
        mock_should_use_parallel.return_value = False

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        # Should fall back to parent class method
        with patch.object(analyzer.__class__.__bases__[0], 'analyze_file') as mock_parent:
            mock_parent.return_value = {"test": "result"}

            analyzer.analyze_file(self.test_file, comprehensive=True)

            mock_parent.assert_called_once()

    def test_multiple_file_processing(self):
        """Test multiple file processing."""
        # Create additional test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.wav")
            sr = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            sf.write(file_path, audio, sr)
            test_files.append(file_path)

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)  # Disable for testing

        # Test multiple file analysis
        results = analyzer.analyze_files(test_files, comprehensive=False)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)

        # Clean up
        for file_path in test_files:
            os.remove(file_path)

    def test_batch_sequential_mode_honors_comprehensive_flag(self):
        """A sequential batch must pass its explicit comprehensive setting through."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        results = analyzer.analyze_files([self.test_file], comprehensive=False, detect_key=False)

        self.assertIn(self.test_file, results)
        self.assertEqual(set(results[self.test_file]), {'basic_info'})

    def test_auto_parallel_false_never_queries_auto_configuration(self):
        """Disabling auto parallelism must not inspect the host or create a pool."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        with patch(
            'src.bpm_detector.parallel_analyzer.AutoParallelConfig.get_optimal_config',
            side_effect=AssertionError('auto configuration was queried'),
        ):
            results = analyzer.analyze_files([self.test_file], comprehensive=False, detect_key=False)

        self.assertIn(self.test_file, results)

    def test_two_argument_callback_works_in_auto_parallel_fallback(self):
        """The Smart callback contract must survive the sequential fallback."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)
        events = []

        def callback(progress, message):
            events.append((progress, message))

        result = analyzer.analyze_file(
            self.test_file, comprehensive=False, detect_key=False, progress_callback=callback
        )

        self.assertIn('basic_info', result)
        self.assertTrue(events)
        self.assertTrue(all(len(event) == 2 for event in events))

    def test_two_argument_callback_works_for_basic_only_fallback(self):
        """Basic-only analysis must adapt the base analyzer's callback shape."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.stop_monitoring()
        analyzer.system_monitor.monitoring = True
        analyzer.system_monitor.current_load = {'cpu': 95, 'memory': 0}
        events = []

        def callback(progress, message):
            events.append((progress, message))

        result = analyzer.analyze_file(
            self.test_file, comprehensive=False, detect_key=False, progress_callback=callback
        )

        self.assertIn('basic_info', result)
        self.assertTrue(events)
        self.assertTrue(all(len(event) == 2 for event in events))

    def test_two_argument_callback_works_when_load_forces_fallback(self):
        """High host load must not bypass the Smart callback adapter."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.stop_monitoring()
        analyzer.system_monitor.monitoring = True
        analyzer.system_monitor.current_load = {'cpu': 95, 'memory': 0}
        events = []

        def callback(progress, message):
            events.append((progress, message))

        result = analyzer.analyze_file(self.test_file, comprehensive=True, detect_key=False, progress_callback=callback)

        self.assertIn('basic_info', result)
        self.assertTrue(events)
        self.assertTrue(all(len(event) == 2 for event in events))

    def test_process_pool_worker_is_module_level_and_keeps_audio_configuration(self):
        """Spawned workers must be pickleable and receive sr/hop_length explicitly."""
        # A direct function lookup documents the worker boundary used by ProcessPoolExecutor.
        self.assertTrue(callable(_analyze_single_file_worker))

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2, sr=8000, hop_length=64)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=2, use_process_pool=True)
        second_file = os.path.join(self.temp_dir, 'test-second.wav')
        shutil.copyfile(self.test_file, second_file)
        try:
            expected_analyzer = AudioAnalyzer(sr=8000, hop_length=64)
            expected = {
                path: expected_analyzer.analyze_file(path, comprehensive=False, detect_key=False, analyze_chords=True)
                for path in (self.test_file, second_file)
            }
            results = analyzer.analyze_files(
                [self.test_file, second_file], comprehensive=False, detect_key=False, analyze_chords=True
            )

            self.assertEqual(results, expected)
            for result in results.values():
                self.assertNotIn('error', result)
        finally:
            analyzer.cleanup()
            if os.path.exists(second_file):
                os.remove(second_file)

    def test_thread_batch_owns_file_concurrency_and_profiles_each_file(self):
        """A thread batch must not create nested per-file analysis pools."""
        second_file = os.path.join(self.temp_dir, 'test-second.wav')
        shutil.copyfile(self.test_file, second_file)
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=2, use_process_pool=False)
        analyzer.system_monitor.stop_monitoring()
        analyzer.system_monitor.current_load = {'cpu': 0, 'memory': 0}

        active = 0
        peak = 0
        active_lock = threading.Lock()
        executor_workers = []
        original_chord = analyzer.chord_analyzer.analyze
        original_dynamics = analyzer.dynamics_analyzer.analyze

        def tracked(original):
            def wrapped(*args, **kwargs):
                nonlocal active, peak
                with active_lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    return original(*args, **kwargs)
                finally:
                    with active_lock:
                        active -= 1

            return wrapped

        analyzer.chord_analyzer.analyze = tracked(original_chord)
        analyzer.dynamics_analyzer.analyze = tracked(original_dynamics)

        class RecordingExecutor:
            def __init__(self, *args, **kwargs):
                workers = kwargs.get('max_workers')
                if workers is None and args:
                    workers = args[0]
                executor_workers.append(workers)
                self._executor = ThreadPoolExecutor(*args, **kwargs)

            def __enter__(self):
                self._executor.__enter__()
                return self

            def __exit__(self, *args):
                return self._executor.__exit__(*args)

            def submit(self, *args, **kwargs):
                return self._executor.submit(*args, **kwargs)

        try:
            with patch('src.bpm_detector.parallel_analyzer.ThreadPoolExecutor', RecordingExecutor):
                actual = analyzer.analyze_files(
                    [self.test_file, second_file],
                    comprehensive=False,
                    detect_key=False,
                    analyze_chords=True,
                    analyze_dynamics=True,
                )

            expected_analyzer = AudioAnalyzer(sr=analyzer.sr, hop_length=analyzer.hop_length)
            expected = {
                path: expected_analyzer.analyze_file(
                    path, comprehensive=False, detect_key=False, analyze_chords=True, analyze_dynamics=True
                )
                for path in (self.test_file, second_file)
            }
            self.assertEqual(actual, expected)
            self.assertEqual(executor_workers, [2])
            self.assertLessEqual(peak, 2)
            summary = analyzer.get_performance_summary()
            self.assertEqual(summary['task_summaries']['file_analysis']['count'], 2)
        finally:
            analyzer.cleanup()
            if os.path.exists(second_file):
                os.remove(second_file)

    def test_parallel_batch_reports_progress_for_failed_futures(self):
        """A failed file still advances the batch progress to the terminal value."""
        missing_file = os.path.join(self.temp_dir, 'missing.wav')

        for use_process_pool in (False, True):
            analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)
            analyzer._parallel_config = ParallelConfig(
                enable_parallel=True, max_workers=2, use_process_pool=use_process_pool
            )
            analyzer.system_monitor.stop_monitoring()
            updates = []
            try:
                results = analyzer.analyze_files(
                    [self.test_file, missing_file],
                    comprehensive=False,
                    progress_callback=lambda progress, message: updates.append((progress, message)),
                    detect_key=False,
                )

                self.assertIn(self.test_file, results)
                self.assertIn(missing_file, results)
                self.assertIn('error', results[missing_file])
                self.assertEqual(updates[-1][0], 100)
            finally:
                analyzer.cleanup()

    def test_empty_parallel_batch_reports_completion(self):
        """An empty batch has a deterministic terminal progress notification."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)
        updates = []

        result = analyzer.analyze_files([], progress_callback=lambda progress, message: updates.append(progress))

        self.assertEqual(result, {})
        self.assertEqual(updates, [100])

    def test_parallel_progress_callbacks_are_serialized(self):
        """A slower callback cannot be overtaken by a later task update."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=2, use_process_pool=False)
        analyzer.system_monitor.stop_monitoring()
        values = []
        callback_52 = threading.Event()
        callback_95 = threading.Event()
        release_52 = threading.Event()

        def callback(progress, message):
            if progress == 52.5:
                callback_52.set()
                release_52.wait(timeout=2)
            elif progress == 95.0:
                callback_95.set()
            values.append(progress)

        def fake_parallel(y, sr, path, progress_manager, **kwargs):
            progress_manager.register_task('a', 'A')
            progress_manager.register_task('b', 'B')

            first = threading.Thread(target=lambda: progress_manager.complete_task('a', True))
            first.start()
            self.assertTrue(callback_52.wait(timeout=2))

            second = threading.Thread(target=lambda: progress_manager.complete_task('b', True))
            second.start()
            # Give an unsynchronized callback a chance to overtake the first
            # callback.  A synchronized implementation keeps the second task
            # behind the dispatch lock, then the timeout releases the first.

            def wait_for_second_callback():
                callback_95.wait(timeout=0.25)
                release_52.set()

            threading.Thread(target=wait_for_second_callback).start()
            second.join(timeout=2)
            first.join(timeout=2)
            release_52.set()
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            return {'basic_info': {'filename': path}}

        analyzer._load_audio = lambda path: (np.zeros(8), 22050)
        analyzer._parallel_comprehensive_analysis = fake_parallel
        try:
            analyzer.analyze_file('synthetic.wav', progress_callback=callback)
        finally:
            release_52.set()
            analyzer.cleanup()

        self.assertEqual(values[-1], 100)
        self.assertTrue(all(previous <= current for previous, current in zip(values, values[1:])), values)

    def test_progress_callback_error_is_not_retried(self):
        """A user callback failure is delivered once for each manager update."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.stop_monitoring()
        callback_calls = []

        def callback(progress, message):
            if message not in {'Loading audio file...', 'Starting parallel analysis...', 'Analysis completed!'}:
                callback_calls.append((progress, message))
                raise TypeError('user callback failed')

        def fake_parallel(y, sr, path, progress_manager, **kwargs):
            progress_manager.register_task('one', 'One')
            progress_manager.complete_task('one', True)
            return {'basic_info': {'filename': path}}

        analyzer._load_audio = lambda path: (np.zeros(8), 22050)
        analyzer._parallel_comprehensive_analysis = fake_parallel
        try:
            result = analyzer.analyze_file('synthetic.wav', progress_callback=callback)
        finally:
            analyzer.cleanup()

        self.assertEqual(result, {'basic_info': {'filename': 'synthetic.wav'}})
        self.assertEqual(len(callback_calls), 2)

    def test_selective_chord_analysis_matches_base_schema(self):
        """The parallel chord path must return the same complete schema as AudioAnalyzer."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.current_load = {'cpu': 0, 'memory': 0}
        try:
            expected = AudioAnalyzer(sr=analyzer.sr, hop_length=analyzer.hop_length).analyze_file(
                self.test_file, comprehensive=False, detect_key=False, analyze_chords=True
            )
            actual = analyzer.analyze_file(self.test_file, comprehensive=False, detect_key=False, analyze_chords=True)

            self.assertEqual(actual, expected)
            self.assertEqual(
                set(actual['chord_progression']),
                {
                    'chords',
                    'main_progression',
                    'progression_pattern',
                    'harmonic_rhythm',
                    'chord_complexity',
                    'unique_chords',
                    'chord_changes',
                    'functional_analysis',
                    'modulations',
                    'substitute_chords_ratio',
                },
            )
        finally:
            analyzer.cleanup()

    def test_structure_progress_helper_returns_full_analyzer_result_once(self):
        """Structure progress must wrap the canonical analyzer without a second pass."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        class StructureSpy:
            def __init__(self):
                self.calls = 0
                self.result = {
                    'sections': [{'start_time': 0.0, 'end_time': 1.0}],
                    'section_summary': 'none',
                    'form': 'A',
                    'repetition_ratio': 0.0,
                    'structural_complexity': 0.0,
                    'section_count': 0,
                    'unique_sections': 0,
                    'repetitions': [],
                    'boundaries': [],
                }

            def analyze(self, y, sr, bpm=130.0):
                self.calls += 1
                return self.result

            def __getattr__(self, name):
                raise AssertionError(f'unexpected redundant structure pass: {name}')

        spy = StructureSpy()
        analyzer.structure_analyzer = spy
        helper_calls = []

        def annotate_sections(y, sr, sections, bpm=130.0):
            helper_calls.append((y, sr, bpm))
            annotated = [dict(section, chord_progression='Unknown') for section in sections]
            return annotated

        analyzer._analyze_section_chord_progressions = annotate_sections
        callback_updates = []

        result = analyzer._analyze_structure_with_progress(
            np.zeros(4), 22050, lambda progress, message: callback_updates.append((progress, message)), bpm=97.0
        )

        expected = dict(spy.result)
        expected['sections'] = [{'start_time': 0.0, 'end_time': 1.0, 'chord_progression': 'Unknown'}]
        self.assertEqual(result, expected)
        self.assertEqual(spy.calls, 1)
        self.assertEqual(helper_calls[0][2], 97.0)

    def test_selective_structure_analysis_matches_base_schema(self):
        """Parallel structure output keeps the base fields and section annotations."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.current_load = {'cpu': 0, 'memory': 0}
        try:
            expected = AudioAnalyzer(sr=analyzer.sr, hop_length=analyzer.hop_length).analyze_file(
                self.test_file, comprehensive=False, detect_key=False, analyze_structure=True
            )
            actual = analyzer.analyze_file(
                self.test_file, comprehensive=False, detect_key=False, analyze_structure=True
            )

            self.assertEqual(actual, expected)
            self.assertIn('sections', actual['structure'])
            self.assertIn('section_summary', actual['structure'])
            self.assertIn('repetitions', actual['structure'])
            self.assertIn('boundaries', actual['structure'])
        finally:
            analyzer.cleanup()

    def test_dsp_failure_is_not_retried_when_signature_is_known(self):
        """An analyzer error must propagate after one call, without a fallback retry."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)
        manager = ProgressManager()
        manager.register_task('test', 'Test')
        calls = []

        def failing_analyzer(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError('DSP failure')

        with self.assertRaisesRegex(RuntimeError, 'DSP failure'):
            analyzer._analyze_with_progress('test', manager, failing_analyzer, np.zeros(4), 22050)

        self.assertEqual(len(calls), 1)

    def test_signature_lookup_failure_still_runs_analyzer_once(self):
        """Unsupported signatures use one direct call instead of swallowing it."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)
        manager = ProgressManager()
        manager.register_task('test', 'Test')

        class SignatureUnavailable:
            def __init__(self):
                self.calls = 0

            @property
            def __signature__(self):
                raise RuntimeError('signature unavailable')

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return {'ok': True}

        callback = SignatureUnavailable()
        result = analyzer._analyze_with_progress('test', manager, callback, np.zeros(4), 22050)

        self.assertEqual(result, {'ok': True})
        self.assertEqual(callback.calls, 1)

    def test_manual_worker_limit_is_preserved_after_batch_adjustment(self):
        """File-count tuning may not raise an explicitly requested worker ceiling."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=2, use_process_pool=True)
        adjusted = ParallelConfig(enable_parallel=True, max_workers=16, use_process_pool=True)

        with (
            patch(
                'src.bpm_detector.parallel_analyzer.AutoParallelConfig.get_file_count_adjustment', return_value=adjusted
            ),
            patch.object(analyzer, '_parallel_analyze_multiple_files', return_value={}) as parallel,
        ):
            analyzer._analyze_multiple_files(['one.wav', 'two.wav'], comprehensive=False)

        passed_config = parallel.call_args.args[2]
        self.assertEqual(passed_config.max_workers, 2)

    def test_single_file_executor_uses_recommended_workers_once(self):
        """Single-file analysis must use the recommended count without multiplying it."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)
        manager = ProgressManager()
        analyzer._get_current_max_workers = lambda: 2
        analyzer._analyze_basic_info_with_progress = lambda *args, **kwargs: {
            'filename': 'test.wav',
            'bpm': 120.0,
            'key': None,
        }
        future = Future()
        future.set_result({})

        with patch('src.bpm_detector.parallel_analyzer.ThreadPoolExecutor') as executor_cls:
            executor = executor_cls.return_value.__enter__.return_value
            executor.submit.return_value = future
            analyzer._parallel_comprehensive_analysis(
                np.zeros(8), 22050, 'test.wav', manager, detect_key=False, comprehensive=False, analyze_rhythm=True
            )

        executor_cls.assert_called_once_with(max_workers=2)

    def test_modulation_requires_key_detection(self):
        """Modulation analysis needs a detected base key."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        with self.assertRaises(ValueError):
            analyzer.analyze_file(self.test_file, detect_key=False, detect_modulation=True)

    def test_parallel_modulation_matches_base_analysis(self):
        """The modulation result is produced by the same detector as the base path."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=1)
        analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=1, use_process_pool=False)
        analyzer.system_monitor.current_load = {'cpu': 0, 'memory': 0}
        try:
            expected = AudioAnalyzer(sr=analyzer.sr, hop_length=analyzer.hop_length).analyze_file(
                self.test_file, comprehensive=False, detect_key=True, detect_modulation=True
            )
            actual = analyzer.analyze_file(self.test_file, comprehensive=False, detect_key=True, detect_modulation=True)

            self.assertEqual(actual, expected)
            self.assertIn('modulation_analysis', actual)
        finally:
            analyzer.cleanup()


class TestAdvancedParallelFeatures(unittest.TestCase):
    """Test advanced parallel processing features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sr = 22050

        # Create test audio file
        duration = 2.0
        t = np.linspace(0, duration, int(self.sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        self.test_file = os.path.join(self.temp_dir, "test.wav")
        sf.write(self.test_file, audio, self.sr)

        self.analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.analyzer, 'cleanup'):
            self.analyzer.cleanup()
        shutil.rmtree(self.temp_dir)

    def test_progress_manager_integration(self):
        """Test integration with progress manager."""
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        self.analyzer.analyze_file(
            self.test_file, comprehensive=True, progress_callback=progress_callback, detailed_progress=True
        )

        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)

        # Progress should be between 0 and 100
        for progress, message in progress_updates:
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 100.0)
            self.assertIsInstance(message, str)

        self.assertEqual(progress_updates[-1][0], 100)
        progress_values = [progress for progress, _ in progress_updates]
        self.assertTrue(all(previous <= current for previous, current in zip(progress_values, progress_values[1:])))

    def test_full_comprehensive_parallel_matches_sequential(self):
        """The enabled single-file pool preserves the complete base result."""
        sequential = AudioAnalyzer(sr=self.sr, hop_length=128).analyze_file(
            self.test_file, comprehensive=True, detect_key=True
        )
        parallel_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2, sr=self.sr, hop_length=128)
        parallel_analyzer._parallel_config = ParallelConfig(enable_parallel=True, max_workers=2, use_process_pool=False)
        parallel_analyzer.system_monitor.stop_monitoring()
        parallel_analyzer.system_monitor.current_load = {'cpu': 0, 'memory': 0}
        try:
            parallel = parallel_analyzer.analyze_file(self.test_file, comprehensive=True, detect_key=True)

            self.assertEqual(set(parallel), set(sequential))
            self.assertEqual(parallel['basic_info'], sequential['basic_info'])
            self.assertEqual(parallel['structure'], sequential['structure'])
            self.assertEqual(parallel['chord_progression'], sequential['chord_progression'])
            self.assertEqual(set(parallel['structure']), set(sequential['structure']))
            self.assertEqual(set(parallel['chord_progression']), set(sequential['chord_progression']))
            np.testing.assert_allclose(
                parallel['similarity_features']['feature_vector'],
                sequential['similarity_features']['feature_vector'],
                rtol=1e-6,
                atol=1e-7,
            )
        finally:
            parallel_analyzer.cleanup()

    def test_system_resource_monitoring(self):
        """Test system resource monitoring during analysis."""
        from src.bpm_detector.auto_parallel import SystemMonitor

        # Enable system monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()

        try:
            results = self.analyzer.analyze_file(self.test_file, comprehensive=True)

            # Should complete successfully even with monitoring
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

            # Check monitoring functionality
            should_reduce = monitor.should_reduce_parallelism()
            self.assertIsInstance(should_reduce, bool)

        finally:
            monitor.stop_monitoring()

    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        from src.bpm_detector.auto_parallel import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Start profiling
        profile_data = profiler.start_profiling("test_analysis")

        # Run analysis
        results = self.analyzer.analyze_file(self.test_file, comprehensive=True)

        # End profiling
        profiler.end_profiling(profile_data)

        # Should complete successfully with profiling
        self.assertIsInstance(results, dict)

        # Check profiling data
        summary = profiler.get_performance_summary()
        self.assertIsInstance(summary, dict)

    def test_adaptive_worker_adjustment(self):
        """Test adaptive worker count adjustment."""
        from src.bpm_detector.auto_parallel import AutoParallelConfig

        # Test with different file counts
        # Create multiple test files
        large_batch = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"batch_test_{i}.wav")
            duration = 1.0
            t = np.linspace(0, duration, int(self.sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            sf.write(file_path, audio, self.sr)
            large_batch.append(file_path)

        try:
            # Test file count adjustment
            base_config = AutoParallelConfig.get_optimal_config()
            adjusted_config = AutoParallelConfig.get_file_count_adjustment(len(large_batch), base_config)

            self.assertIsInstance(adjusted_config.max_workers, int)
            self.assertGreater(adjusted_config.max_workers, 0)

        finally:
            # Clean up batch files
            for file_path in large_batch:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def test_error_handling_in_parallel(self):
        """Test error handling in parallel processing."""
        # Create a corrupted file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid audio file")

        test_files = [self.test_file, corrupted_file]

        try:
            results = self.analyzer.analyze_files(test_files, comprehensive=False)

            # Should handle errors gracefully
            self.assertIsInstance(results, dict)

            # Good file should have results
            self.assertIn(self.test_file, results)

            # Results should be valid for good file
            if self.test_file in results:
                self.assertIsInstance(results[self.test_file], dict)

        except Exception as e:
            # If exception is raised, it should be handled gracefully
            self.assertIsInstance(e, Exception)

        finally:
            if os.path.exists(corrupted_file):
                os.remove(corrupted_file)

    def test_memory_management(self):
        """Test memory management during parallel processing."""
        # Test with limited workers to check memory management
        limited_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)

        try:
            results = limited_analyzer.analyze_file(self.test_file, comprehensive=True)

            # Should complete without memory issues
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

        finally:
            if hasattr(limited_analyzer, 'cleanup'):
                limited_analyzer.cleanup()

    def test_parallel_strategy_selection(self):
        """Test automatic selection of parallel strategy."""
        from src.bpm_detector.auto_parallel import AutoParallelConfig, ParallelStrategy

        # Get optimal configuration
        config = AutoParallelConfig.get_optimal_config()

        # Should have a valid strategy
        self.assertIsInstance(config.strategy, ParallelStrategy)
        # Should have a valid strategy (including AGGRESSIVE_PARALLEL)
        valid_strategies = [
            ParallelStrategy.THREAD_POOL,
            ParallelStrategy.PROCESS_POOL,
            ParallelStrategy.AGGRESSIVE_PARALLEL,
            ParallelStrategy.BALANCED_PARALLEL,
            ParallelStrategy.CONSERVATIVE_PARALLEL,
            ParallelStrategy.SEQUENTIAL_ONLY,
        ]
        self.assertIn(config.strategy, valid_strategies)

        # Should have reasonable worker count
        self.assertGreater(config.max_workers, 0)
        self.assertLessEqual(config.max_workers, 32)  # Reasonable upper bound

    def test_cleanup_functionality(self):
        """Test proper cleanup of parallel resources."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        # Run some analysis
        results = analyzer.analyze_file(self.test_file, comprehensive=False)
        self.assertIsInstance(results, dict)

        # Test explicit cleanup
        analyzer.cleanup()

        # Should not raise errors after cleanup
        try:
            analyzer.cleanup()  # Second cleanup should be safe
        except Exception as e:
            self.fail(f"Cleanup should not raise errors: {e}")

    def test_detailed_progress_tracking(self):
        """Test detailed progress tracking functionality."""
        from src.bpm_detector.progress_manager import DetailedProgressDisplay, ProgressManager

        progress_manager = ProgressManager()
        detailed_display = DetailedProgressDisplay()

        # Register some tasks
        progress_manager.register_task("task1", "Test Task 1")
        progress_manager.register_task("task2", "Test Task 2")

        # Update progress
        progress_manager.update_progress("task1", 50.0, "Halfway done")
        progress_manager.update_progress("task2", 25.0, "Quarter done")

        # Test display update
        detailed_display.update(progress_manager)

        # Complete tasks
        progress_manager.complete_task("task1", success=True)
        progress_manager.complete_task("task2", success=True)

        # Get final status
        status = progress_manager.get_status_summary()
        self.assertIsInstance(status, dict)
        self.assertIn('total_tasks', status)
        self.assertIn('completed_tasks', status)


if __name__ == '__main__':
    unittest.main()
