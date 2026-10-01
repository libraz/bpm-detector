"""Concurrency and snapshot regression tests for progress management."""

import io
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

from src.bpm_detector.progress_manager import (
    DetailedProgressDisplay,
    ProgressManager,
    SimpleProgressDisplay,
    TaskProgress,
    TaskStatus,
)


class SnapshotOnlyManager:
    """Small public-interface test double for displays."""

    def __init__(self):
        self._task = TaskProgress(name="Snapshot task", status=TaskStatus.RUNNING, progress=25.0)

    def get_task_details(self):
        return {"snapshot": replace(self._task)}


class TestProgressSnapshots(unittest.TestCase):
    def test_task_details_are_independent_dataclass_snapshots(self):
        manager = ProgressManager()
        manager.register_task("task", "Task")
        manager.update_progress("task", 25.0, "working")

        details = manager.get_task_details()
        details["task"].progress = 100.0
        details["task"].status = TaskStatus.COMPLETED
        details["task"].message = "changed outside manager"

        current = manager.get_task_details()["task"]
        self.assertEqual(current.progress, 25.0)
        self.assertEqual(current.status, TaskStatus.RUNNING)
        self.assertEqual(current.message, "working")

    def test_simple_display_reads_only_public_snapshots(self):
        display = SimpleProgressDisplay()
        display.update_interval = 0.0
        output = io.StringIO()

        with redirect_stdout(output):
            display.update(SnapshotOnlyManager())

        self.assertIn("Progress:", output.getvalue())
        self.assertNotIn("Progress display error", output.getvalue())

    def test_detailed_display_reads_only_public_snapshots(self):
        display = DetailedProgressDisplay()
        display.update_interval = 0.0
        output = io.StringIO()

        with redirect_stdout(output):
            display.update(SnapshotOnlyManager())

        self.assertIn("Overall Progress:", output.getvalue())
        self.assertNotIn("Progress display error", output.getvalue())


class TestTerminalProgressNotifications(unittest.TestCase):
    def test_terminal_updates_bypass_throttle_and_callbacks_can_read_manager(self):
        manager = ProgressManager()
        manager.register_task("success", "Success")
        manager.register_task("failure", "Failure")
        events = []

        def callback(progress_manager):
            summary = progress_manager.get_status_summary()
            events.append((summary["completed"], summary["failed"], summary["total"]))

        manager.add_callback(callback)
        with patch("src.bpm_detector.progress_manager.time.time", return_value=123.0):
            manager.update_progress("success", 25.0)
            manager.complete_task("success", success=True)
            manager.complete_task("failure", success=False, error="failed")
            manager.reset()

        self.assertEqual(events, [(1, 0, 2), (1, 1, 2), (0, 0, 0)])

    def test_reentrant_terminal_callbacks_are_bounded(self):
        """A terminal callback must not notify forever when it repeats itself."""
        manager = ProgressManager()
        manager.register_task("task", "Task")
        complete_calls = []

        def complete_again(progress_manager):
            complete_calls.append(True)
            progress_manager.complete_task("task", success=True)

        manager.add_callback(complete_again)
        manager.complete_task("task", success=True)

        self.assertEqual(len(complete_calls), 1)

    def test_reentrant_terminal_callback_with_empty_error_is_bounded(self):
        """An explicitly empty error is still a recorded terminal value."""
        manager = ProgressManager()
        manager.register_task("task", "Task")
        complete_calls = []

        def complete_again(progress_manager):
            complete_calls.append(True)
            progress_manager.complete_task("task", success=True, error="")

        manager.add_callback(complete_again)
        manager.complete_task("task", success=True, error="")

        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(manager.get_task_details()["task"].error, "")

    def test_reentrant_reset_callback_is_bounded(self):
        """Resetting an already empty manager must not emit another event."""
        manager = ProgressManager()
        manager.register_task("task", "Task")
        reset_calls = []

        def reset_again(progress_manager):
            reset_calls.append(True)
            progress_manager.reset()

        manager.add_callback(reset_again)
        manager.reset()

        self.assertEqual(len(reset_calls), 1)


if __name__ == "__main__":
    unittest.main()
