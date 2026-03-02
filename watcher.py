#!/usr/bin/env python3
import os
import time
import hashlib
import logging
import threading
import shutil
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from agents.db_hooks import init_db_pool
from graph.graph import main_graph
from agents.universal_parser_agent import init_worker
from utils.logger import get_logger

from dotenv import load_dotenv

load_dotenv()
init_worker()
PIPELINE = main_graph()

WATCH_DIR     = Path(os.getenv("WATCH_DIR", "/data/incoming")).resolve()   # FIX: always absolute
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/data/processed")).resolve()
FAILED_DIR    = Path(os.getenv("FAILED_DIR", "/data/failed")).resolve()
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", "4"))
STABLE_SECONDS = int(os.getenv("STABLE_SECONDS", "3"))
QUEUE_SIZE    = int(os.getenv("QUEUE_SIZE", "1000"))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO")
TASK_TIMEOUT  = int(os.getenv("TASK_TIMEOUT", "300"))
DEBOUNCE_SECONDS = 2  # events within this window are deduplicated

FAILED_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger("watcher")

# ---------------- utils ---------------- #

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_until_stable(path: Path):
    last = -1
    stable_cycles = 0
    start = time.time()

    while True:
        if time.time() - start > 300:
            raise TimeoutError(f"File never stabilized: {path}")

        if not path.exists():
            raise FileNotFoundError(f"File disappeared before processing: {path}")  # FIX: descriptive message

        size = path.stat().st_size
        if size == last:
            stable_cycles += 1
            if stable_cycles >= 2:
                return
        else:
            stable_cycles = 0

        last = size
        time.sleep(STABLE_SECONDS)


def run_with_timeout(executor, func, *args):
    future = executor.submit(func, *args)
    try:
        return future.result(timeout=TASK_TIMEOUT)
    except Exception:
        future.cancel()
        raise


def safe_move(src: Path, dest: Path, retries: int = 5, delay: float = 2.0):
    """Move a file with retries to handle Windows file locks (WinError 32)."""
    for attempt in range(retries):
        try:
            shutil.move(str(src), str(dest))
            return True
        except PermissionError as e:
            if attempt < retries - 1:
                log.warning(f"File locked, retry {attempt + 1}/{retries} in {delay}s -> {src.name}: {e}")
                time.sleep(delay)
            else:
                log.error(f"Failed to move after {retries} retries -> {src.name}: {e}")
                raise
    return False

# --------------- worker ---------------- #

class WorkerPool:

    def __init__(self):
        self.q = Queue(maxsize=QUEUE_SIZE)
        self.executor = ThreadPoolExecutor(MAX_WORKERS)
        self.task_executor = ThreadPoolExecutor(MAX_WORKERS)
        self.processing_hashes = set()
        self.enqueued_paths = set()
        self.lock = threading.Lock()

        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def submit(self, path: Path): #put file in queue
        path = path.resolve()  # FIX: always normalize to absolute path
        with self.lock:
            if path in self.enqueued_paths:
                log.debug(f"Already enqueued, ignoring -> {path.name}")
                return
            self.enqueued_paths.add(path)

        self.q.put(path)

    def _loop(self): #get file path from queue
        while True:
            try:
                path = self.q.get(timeout=1)
            except Empty:
                continue

            self.executor.submit(self._process, path)
            self.q.task_done()

    def _process(self, path: Path): #process file
        file_hash = None
        start_time = time.time()

        try:
            wait_until_stable(path)

            # FIX: verify file still exists after stable check
            if not path.exists():
                log.warning(f"File vanished after stable check -> {path.name}")
                return

            file_hash = sha256(path)

            with self.lock:
                # FIX: discard from enqueued only AFTER hash is known and checked
                if file_hash in self.processing_hashes:
                    log.info(f"Duplicate detected by hash, skipping -> {path.name}")
                    self.enqueued_paths.discard(path)
                    return
                self.processing_hashes.add(file_hash)
                self.enqueued_paths.discard(path)

            log.info(f"Processing -> {path.name}")

            result = run_with_timeout(
            self.task_executor,
            PIPELINE.invoke,
            {
                "original_filename": str(path),
                "file_hash": file_hash
            })

            status = result.get("status_comp", "")

            # ── Skipped (duplicate / collision) ──────────────────────────────────────────
            if status == "skipped":
                reason = result.get("skip_reason", "unknown")
                log.info(f"Skipped ({reason}) -> {path.name}")

                # FIX: existence check before move
                if path.exists():
                    dest = PROCESSED_DIR / path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    safe_move(path, dest)
                    log.info(f"Moved to processed (skipped) -> {path.name}")
                else:
                    log.warning(f"File already gone before move -> {path.name}")
                return  # <-- explicit return so finally still runs for hash cleanup

            # ── Incomplete — graph exited too early ──────────────────────────────────────
            if status != "completed":
                raise RuntimeError(
                    f"Pipeline incomplete. Status: '{status}'. "
                    f"Keys returned: {list(result.keys())}"
                )

            # ── Sanity-check chunks ───────────────────────────────────────────────────────
            if not result.get("chunks"):
                raise RuntimeError(
                    f"No chunks produced. Keys returned: {list(result.keys())}"
                )

            # write hash BEFORE move
            hash_file = path.with_suffix(".sha256.tmp")
            with open(hash_file, "w") as f:
                f.write(file_hash)

            dest = PROCESSED_DIR / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            # FIX: existence check before move in completed branch
            if path.exists():
                safe_move(path, dest)
                if hash_file.exists():
                    hash_file.rename(dest.with_suffix(".sha256"))
            else:
                log.warning(f"File already gone before move -> {path.name}")
                if hash_file.exists():
                    hash_file.unlink()

            elapsed = time.time() - start_time
            log.info(f"Done -> {path.name} ({elapsed:.2f}s)")

        except FileNotFoundError as e:
            # FIX: file gone before we could process — not an error worth quarantining
            log.warning(f"File no longer exists, skipping -> {path.name}: {e}")

        except Exception as e:
            log.error(f"Failed {path.name} : {e}", exc_info=True)  # FIX: log full traceback

            # FIX: only quarantine if the file still exists
            if path.exists():
                try:
                    failed_dest = FAILED_DIR / f"{int(time.time())}_{path.name}"
                    safe_move(path, failed_dest)
                    log.info(f"Quarantined -> {failed_dest.name}")
                except Exception as move_err:
                    log.error(f"Failed to quarantine {path.name}: {move_err}")
            else:
                log.warning(f"File already gone, skipping quarantine -> {path.name}")

        finally:
            with self.lock:
                if file_hash:
                    self.processing_hashes.discard(file_hash)
                self.enqueued_paths.discard(path)  # FIX: always clean up


# ------------ watcher ---------------- #

class WatchHandler(FileSystemEventHandler):

    def __init__(self, pool: WorkerPool):
        self.pool = pool
        self._seen = {}          # path -> last_event_time
        self._seen_lock = threading.Lock()

    def _debounced_submit(self, src_path: str):
        """Only submit if this path hasn't been seen within DEBOUNCE_SECONDS."""
        path = Path(src_path).resolve()
        now = time.time()
        with self._seen_lock:
            last = self._seen.get(path, 0)
            if now - last < DEBOUNCE_SECONDS:
                log.debug(f"Debounced duplicate event -> {path.name}")
                return
            self._seen[path] = now
        self.pool.submit(path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._debounced_submit(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._debounced_submit(event.src_path)


# ------------- main ------------------- #

def main():
    init_db_pool()

    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pool = WorkerPool()

    # startup scan
    for f in WATCH_DIR.glob("*.pdf"):
        pool.submit(f)

    handler = WatchHandler(pool)
    obs = Observer()
    obs.schedule(handler, str(WATCH_DIR), recursive=False)

    log.info("Watcher running")
    log.info(f"Watching -> {WATCH_DIR}")

    obs.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
        log.info("Stopping watcher...")

    obs.join()
    pool.executor.shutdown(wait=True)
    pool.task_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()