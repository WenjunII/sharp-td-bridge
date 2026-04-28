"""
Frame Watcher
=============
Monitors a directory for new/updated image files from TouchDesigner
and triggers SHARP inference when a new frame arrives.

Uses watchdog for efficient filesystem event monitoring.
"""

import time
import logging
from pathlib import Path
from threading import Event, Lock
from queue import Queue

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class FrameHandler(FileSystemEventHandler):
    """Handles filesystem events for new/updated frames."""

    def __init__(self, frame_queue: Queue, debounce_ms: int = 50):
        """
        Args:
            frame_queue: Queue to push new frame paths to.
            debounce_ms: Minimum time between accepting frames (prevents rapid-fire).
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.debounce_s = debounce_ms / 1000.0
        self._last_event_time = 0.0
        self._lock = Lock()

    def _is_image(self, path: str) -> bool:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS

    def _handle_event(self, event):
        if event.is_directory:
            return
        if not self._is_image(event.src_path):
            return
        # Skip temp files (from double-buffering)
        if event.src_path.endswith(".tmp"):
            return

        with self._lock:
            now = time.time()
            if now - self._last_event_time < self.debounce_s:
                return  # Debounce
            self._last_event_time = now

        # Only keep the latest frame — discard old ones if queue is full
        frame_path = Path(event.src_path)
        if not self.frame_queue.full():
            self.frame_queue.put(frame_path)
        else:
            # Drop the oldest, add the newest
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass
            self.frame_queue.put(frame_path)

        logger.debug(f"Frame queued: {frame_path.name}")

    def on_created(self, event):
        self._handle_event(event)

    def on_modified(self, event):
        self._handle_event(event)


class FrameWatcher:
    """
    Watches a directory for new image frames from TouchDesigner.

    Usage:
        watcher = FrameWatcher("./frames")
        watcher.start()

        while True:
            frame_path = watcher.get_frame(timeout=1.0)
            if frame_path:
                process(frame_path)
    """

    def __init__(self, watch_dir: str | Path, queue_size: int = 2, debounce_ms: int = 50):
        """
        Args:
            watch_dir: Directory to monitor for new image files.
            queue_size: Max frames to buffer. Small = always process latest.
            debounce_ms: Minimum time between frame events.
        """
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.frame_queue = Queue(maxsize=queue_size)
        self.handler = FrameHandler(self.frame_queue, debounce_ms=debounce_ms)
        self.observer = Observer()
        self._running = False

    def start(self):
        """Start watching for frame changes."""
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        self._running = True
        logger.info(f"Frame watcher started on: {self.watch_dir}")

    def stop(self):
        """Stop watching."""
        self._running = False
        self.observer.stop()
        self.observer.join()
        logger.info("Frame watcher stopped")

    def get_frame(self, timeout: float = 1.0) -> Path | None:
        """
        Get the next available frame path.

        Args:
            timeout: Seconds to wait for a frame.

        Returns:
            Path to the frame image, or None if timeout.
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Exception:
            return None

    @property
    def is_running(self) -> bool:
        return self._running
