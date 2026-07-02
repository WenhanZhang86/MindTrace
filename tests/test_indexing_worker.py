from core.config import LLMSettings
from core.embedding import HashingEmbeddingModel
from core.indexing_worker import IndexingWorker
from core.retriever import HybridRetriever
from core.session_store import CaptureEntry
from core.sqlite_store import SQLiteStore
from core.vector_store import VectorStore


def test_worker_processes_pending_capture(tmp_path):
    store = SQLiteStore(tmp_path)
    retriever = HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=32),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )
    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "worker indexes FastAPI"),
    )
    store.enqueue_capture(capture_id, "s1")
    worker = IndexingWorker(store, retriever, poll_interval_seconds=0.01)

    assert worker.process_once()

    assert store.queue_counts()["done"] == 1
    assert store.count_vector_chunks() > 0
    assert retriever.retrieve("FastAPI", top_k=2)


def test_worker_failed_job_retries_then_fails(tmp_path):
    class BrokenRetriever:
        def index_captures(self, captures):
            raise RuntimeError("embed failed")

    store = SQLiteStore(tmp_path)
    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "broken job"),
    )
    store.enqueue_capture(capture_id, "s1")
    worker = IndexingWorker(store, BrokenRetriever(), poll_interval_seconds=0.01, max_attempts=2)

    worker.process_once()
    assert store.queue_counts()["pending"] == 1
    worker.process_once()
    assert store.queue_counts()["failed"] == 1


def test_capture_save_enqueue_does_not_embed_inline(tmp_path):
    class CountingEmbedding(HashingEmbeddingModel):
        def __init__(self):
            super().__init__(dimensions=16)
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            return super().embed(text)

    store = SQLiteStore(tmp_path)
    embedding = CountingEmbedding()
    HybridRetriever(
        store,
        embedding,
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )
    embedding.calls = 0

    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "queued only"),
    )
    store.enqueue_capture(capture_id, "s1")

    assert embedding.calls == 0
    assert store.queue_counts()["pending"] == 1


def test_rebuild_leaves_queue_consistent(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_session(
        "s1",
        "2026-07-02T12:00:00",
        "",
        0,
        "test",
        LLMSettings(),
    )
    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "rebuild queue consistency"),
    )
    store.enqueue_capture(capture_id, "s1")
    retriever = HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=24),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )

    retriever.rebuild_vector_index()

    assert store.queue_counts().get("done", 0) == 1
    assert store.count_vector_chunks() > 0


def test_worker_batches_multiple_captures(tmp_path):
    class CountingRetriever:
        def __init__(self):
            self.batch_sizes = []

        def index_captures(self, captures):
            self.batch_sizes.append(len(captures))
            return len(captures)

    store = SQLiteStore(tmp_path)
    for index in range(3):
        capture_id = store.save_capture(
            "s1",
            CaptureEntry("2026-07-02T12:00:00", "screen", f"batch capture {index}"),
        )
        store.enqueue_capture(capture_id, "s1")
    retriever = CountingRetriever()
    worker = IndexingWorker(store, retriever, batch_size=10, poll_interval_seconds=0.01)

    assert worker.process_once()

    assert retriever.batch_sizes == [3]
    assert store.queue_counts()["done"] == 3


def test_worker_saves_embeddings_once_per_batch(tmp_path):
    class CountingVectorStore(VectorStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_calls = 0

        def save(self):
            self.save_calls += 1
            return super().save()

    store = SQLiteStore(tmp_path)
    vector_store = CountingVectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index")
    retriever = HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=vector_store,
        auto_start_background=False,
    )
    vector_store.save_calls = 0
    for index in range(3):
        capture_id = store.save_capture(
            "s1",
            CaptureEntry("2026-07-02T12:00:00", "screen", f"save once capture {index}"),
        )
        store.enqueue_capture(capture_id, "s1")

    worker = IndexingWorker(store, retriever, batch_size=10, poll_interval_seconds=0.01)
    worker.process_once()

    assert vector_store.save_calls == 1
