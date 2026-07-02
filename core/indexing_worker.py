import threading
import time

from .logger import get_logger
from .retriever import HybridRetriever
from .sqlite_store import SQLiteStore


logger = get_logger(__name__)


class IndexingWorker:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        retriever: HybridRetriever,
        poll_interval_seconds: float = 1.0,
        batch_wait_seconds: float = 3.0,
        batch_size: int = 10,
        max_attempts: int = 3,
    ) -> None:
        self.sqlite_store = sqlite_store
        self.retriever = retriever
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_wait_seconds = batch_wait_seconds
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Indexing worker started")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Indexing worker stopped")

    def process_once(self) -> bool:
        jobs = self.sqlite_store.next_indexing_jobs(
            limit=self.batch_size,
            max_attempts=self.max_attempts,
        )
        if not jobs:
            return False
        start = time.perf_counter()
        done_ids: list[int] = []
        try:
            captures = []
            for job in jobs:
                capture = self.sqlite_store.get_capture(job["capture_id"])
                if not capture:
                    self.sqlite_store.mark_indexing_job_failed(
                        job["id"],
                        f"capture not found: {job['capture_id']}",
                        max_attempts=self.max_attempts,
                    )
                    continue
                captures.append(capture)
                done_ids.append(job["id"])
            if captures:
                vectors = self.retriever.index_captures(captures)
                self.sqlite_store.mark_indexing_jobs_done(done_ids)
                logger.info(
                    "Batch indexing completed jobs=%s captures=%s vectors=%s elapsed_ms=%.1f",
                    len(done_ids),
                    len(captures),
                    vectors,
                    (time.perf_counter() - start) * 1000,
                )
            return bool(done_ids)
        except Exception as exc:
            logger.exception("Indexing batch failed")
            for job in jobs:
                self.sqlite_store.mark_indexing_job_failed(
                    job["id"],
                    str(exc),
                    max_attempts=self.max_attempts,
                )
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            did_work = self.process_once()
            if not did_work:
                time.sleep(self.poll_interval_seconds)
            else:
                time.sleep(self.batch_wait_seconds)
