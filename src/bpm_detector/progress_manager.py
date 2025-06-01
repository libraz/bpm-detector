"""Progress management module for parallel processing."""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskProgress:
    """Individual task progress tracking."""

    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


class ProgressManager:
    """Parallel processing progress manager."""

    def __init__(self):
        self._tasks: Dict[str, TaskProgress] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._overall_progress = 0.0

    def register_task(self, task_id: str, name: str):
        """Register a new task."""
        with self._lock:
            self._tasks[task_id] = TaskProgress(name=name)
        self._notify_update()

    def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.progress = min(100.0, max(0.0, progress))
                task.message = message
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
        self._notify_update()

    def complete_task(self, task_id: str, success: bool = True, error: Optional[str] = None):
        """Mark task as completed."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.progress = 100.0 if success else task.progress
                task.end_time = time.time()
                if error:
                    task.error = error
        self._notify_update()

    def get_overall_progress(self) -> float:
        """Calculate overall progress."""
        with self._lock:
            if not self._tasks:
                return 0.0

            total_progress = sum(task.progress for task in self._tasks.values())
            return total_progress / len(self._tasks)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary."""
        with self._lock:
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
            failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

            # Calculate overall progress directly to avoid recursion
            if not self._tasks:
                overall_progress = 0.0
            else:
                total_progress = sum(task.progress for task in self._tasks.values())
                overall_progress = total_progress / len(self._tasks)

            return {
                'total': len(self._tasks),
                'total_tasks': len(self._tasks),  # Field name expected by tests
                'completed': completed,
                'completed_tasks': completed,  # Field name expected by tests
                'running': running,
                'failed': failed,
                'pending': pending,
                'overall_progress': overall_progress,
            }

    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task names."""
        with self._lock:
            return [task.name for task in self._tasks.values() if task.status == TaskStatus.RUNNING]

    def get_task_details(self) -> Dict[str, TaskProgress]:
        """Get detailed task information."""
        with self._lock:
            return self._tasks.copy()

    def add_callback(self, callback: Callable):
        """Add progress update callback."""
        self._callbacks.append(callback)

    def _notify_update(self):
        """Notify all callbacks of progress update."""
        # Prevent infinite loops with update throttling
        current_time = time.time()
        if hasattr(self, '_last_notify_time'):
            if current_time - self._last_notify_time < 0.1:  # 100ms throttle for smooth updates
                return
        self._last_notify_time: float = current_time

        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Warning: Progress callback error: {e}")

    def reset(self):
        """Reset all tasks."""
        with self._lock:
            self._tasks.clear()
        self._notify_update()


class ProgressDisplay:
    """Base class for progress display."""

    def __init__(self):
        self.last_update = 0.0
        self.update_interval = 0.1  # 100ms

    def should_update(self) -> bool:
        """Check if display should be updated."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = float(current_time)
            return True
        return False

    def update(self, progress_manager: ProgressManager):
        """Update display with current progress."""
        raise NotImplementedError

    def close(self):
        """Close display."""


class SimpleProgressDisplay(ProgressDisplay):
    """Simple progress display using console output."""

    def __init__(self):
        super().__init__()
        self.last_progress = -1.0
        self.update_interval = 0.2  # 200ms for smooth updates

    def update(self, progress_manager: ProgressManager):
        """Update simple progress display."""
        if not self.should_update():
            return

        # Prevent recursive calls
        if hasattr(self, '_updating') and self._updating:
            return
        self._updating: bool = True

        try:
            # Get progress data without triggering callbacks
            with progress_manager._lock:
                overall_progress = (
                    sum(task.progress for task in progress_manager._tasks.values()) / len(progress_manager._tasks)
                    if progress_manager._tasks
                    else 0
                )
                running_tasks = [
                    task.name for task in progress_manager._tasks.values() if task.status == TaskStatus.RUNNING
                ]
                completed = sum(1 for task in progress_manager._tasks.values() if task.status == TaskStatus.COMPLETED)
                total = len(progress_manager._tasks)

            # Only update if progress changed significantly
            if abs(overall_progress - self.last_progress) < 0.5:
                return

            # Build progress bar
            bar_length = 30
            filled = int(overall_progress * bar_length / 100)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Build description
            if running_tasks:
                desc = f"Running: {', '.join(running_tasks[:2])}"
                if len(running_tasks) > 2:
                    desc += f" (+{len(running_tasks) - 2} more)"
            else:
                desc = f"Completed: {completed}/{total} tasks"

            # Clear line and show progress
            print(f"\r🔄 Progress: [{bar}] {overall_progress:5.1f}% - {desc}", end="", flush=True)
            self.last_progress = float(overall_progress)

        except Exception as e:
            print(f"Progress display error: {e}")
        finally:
            self._updating = False

    def close(self):
        """Close progress display."""
        print("\r" + " " * 80 + "\r", end="")  # Clear line


class DetailedProgressDisplay(ProgressDisplay):
    """Detailed hierarchical progress display using console output."""

    def __init__(self):
        super().__init__()
        self.last_output = ""
        self.update_interval = 0.3  # 300ms for smooth updates
        self.task_start_times = {}  # Track when tasks started
        self.estimated_durations = {  # Estimated task durations in seconds
            'basic_info': 2,
            'chord_progression': 3,
            'structure': 8,  # Longest task
            'rhythm': 2,
            'timbre': 5,
            'melody_harmony': 4,
            'dynamics': 1,
        }

    def update(self, progress_manager: ProgressManager):
        """Update detailed progress display."""
        if not self.should_update():
            return

        # Prevent recursive calls
        if hasattr(self, '_updating') and self._updating:
            return
        self._updating: bool = True

        try:
            # Get progress data without triggering callbacks
            with progress_manager._lock:
                task_details = progress_manager._tasks.copy()

            # Update estimated progress for running tasks
            current_time = time.time()
            for task_id, task in task_details.items():
                if task.status == TaskStatus.RUNNING:
                    # Track start time
                    if task_id not in self.task_start_times:
                        self.task_start_times[task_id] = current_time

                    # Estimate progress based on time if no recent updates
                    elapsed = current_time - self.task_start_times[task_id]
                    estimated_duration = self.estimated_durations.get(task_id, 5)

                    # If task hasn't updated progress recently, estimate based on time
                    if task.progress < 90:  # Don't override near-completion progress
                        time_based_progress = min(85, (elapsed / estimated_duration) * 100)
                        if time_based_progress > task.progress:
                            task.progress = time_based_progress

            # Calculate overall progress
            overall_progress = (
                sum(task.progress for task in task_details.values()) / len(task_details) if task_details else 0
            )

            # Build output
            lines = []
            lines.append(f"📊 Overall Progress: {overall_progress:.1f}%")
            lines.append("─" * 50)

            for task_id, task in task_details.items():
                # Status indicator
                if task.status == TaskStatus.COMPLETED:
                    status_icon = "✅"
                elif task.status == TaskStatus.FAILED:
                    status_icon = "❌"
                elif task.status == TaskStatus.RUNNING:
                    status_icon = "🔄"
                else:
                    status_icon = "⏸️"

                # Progress bar
                bar_length = 20
                filled = int(task.progress * bar_length / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                line = f"{status_icon} {task.name:<20} [{bar}] {task.progress:5.1f}%"
                if task.message:
                    line += f" {task.message}"
                elif task.status == TaskStatus.RUNNING and task.progress < 90:
                    line += " (estimating...)"
                lines.append(line)

            # Clear previous output and show new
            output = "\n".join(lines)
            if output != self.last_output:
                # Clear previous lines
                if self.last_output:
                    clear_lines = self.last_output.count('\n') + 1
                    print(f"\033[{clear_lines}A\033[J", end="")

                print(output, flush=True)
                self.last_output = output

        except Exception as e:
            print(f"Progress display error: {e}")
        finally:
            self._updating = False

    def close(self):
        """Close progress display."""
        if self.last_output:
            clear_lines = self.last_output.count('\n') + 1
            print(f"\033[{clear_lines}A\033[J", end="")
        print("✅ Analysis completed!")


class ProgressCallback:
    """Progress callback for integration with analyzers."""

    def __init__(self, progress_manager: ProgressManager, task_id: str):
        self.progress_manager = progress_manager
        self.task_id = task_id

    def __call__(self, progress: float, message: str = ""):
        """Update progress."""
        self.progress_manager.update_progress(self.task_id, progress, message)

    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark task as completed."""
        self.progress_manager.complete_task(self.task_id, success, error)


def create_progress_display(detailed: bool = False) -> ProgressDisplay:
    """Create appropriate progress display."""
    if detailed:
        return DetailedProgressDisplay()
    else:
        return SimpleProgressDisplay()
