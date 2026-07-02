from core.config import LLMSettings
from core.embedding import HashingEmbeddingModel
from core.session_store import CaptureEntry
from core.sqlite_store import SQLiteStore
from core.retriever import HybridRetriever
from core.vector_store import VectorStore
import json
import threading
import time


def test_hybrid_retriever_returns_ranked_candidates(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    sqlite_store.save_session(
        session_id="session-1",
        started_at="2026-07-02T12:00:00",
        ended_at="",
        duration_seconds=0,
        app_version="test",
        llm_settings=LLMSettings(),
    )
    sqlite_store.save_capture(
        "session-1",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="FastAPI middleware tutorial and dependency injection notes",
        ),
    )
    sqlite_store.save_capture(
        "session-1",
        CaptureEntry(
            timestamp="2026-07-02T12:00:02",
            source="audio",
            text="Completely unrelated grocery list",
        ),
    )

    retriever = HybridRetriever(sqlite_store, HashingEmbeddingModel(dimensions=64), auto_start_background=False)
    results = retriever.retrieve("FastAPI middleware", top_k=3)

    assert results
    assert "FastAPI" in results[0].chunk.text or "middleware" in results[0].chunk.text
    assert results[0].unified_score > 0


def test_retriever_incremental_add_and_reload(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    model = HashingEmbeddingModel(dimensions=32)
    index_dir = tmp_path / "data" / "vector_index"
    retriever = HybridRetriever(
        sqlite_store,
        model,
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )

    capture_id = sqlite_store.save_capture(
        "session-2",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Persistent vector index remembers Django ORM notes",
        ),
    )
    retriever.index_capture(
        {
            "capture_id": capture_id,
            "session_id": "session-2",
            "timestamp": "2026-07-02T12:00:01",
            "source": "screen",
            "text": "Persistent vector index remembers Django ORM notes",
        }
    )

    assert sqlite_store.count_vector_chunks() > 0
    assert (index_dir / "embeddings.npy").exists()
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["vector_count"] == sqlite_store.count_vector_chunks()

    reloaded = HybridRetriever(
        sqlite_store,
        model,
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )
    reloaded._load_vector_index_background()
    results = reloaded.retrieve("Django ORM", top_k=3)

    assert results
    assert "Django" in results[0].chunk.text or "ORM" in results[0].chunk.text


def test_retriever_rebuild_from_sqlite(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    capture_id = sqlite_store.save_capture(
        "session-3",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Rebuild index from SQLite captures about LangChain",
        ),
    )
    retriever = HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=32),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )
    retriever.vector_store.clear()
    retriever._indexed_capture_ids = set()

    retriever.rebuild_vector_index()

    assert capture_id in retriever._indexed_capture_ids
    assert retriever.retrieve("LangChain", top_k=2)


def test_retriever_embedding_dimension_mismatch_rebuilds(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    sqlite_store.save_capture(
        "session-4",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Dimension mismatch should rebuild index",
        ),
    )
    index_dir = tmp_path / "data" / "vector_index"
    HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )
    HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    ).rebuild_vector_index()

    rebuilt = HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=32),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )
    assert rebuilt.vector_index_status == "stale"
    rebuilt.rebuild_vector_index()

    assert rebuilt.vector_store.embedding_dimension == 32
    assert rebuilt.retrieve("Dimension mismatch", top_k=2)


def test_retriever_corrupt_manifest_triggers_rebuild(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    sqlite_store.save_capture(
        "session-5",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Corrupt manifest should trigger vector rebuild",
        ),
    )
    index_dir = tmp_path / "data" / "vector_index"
    HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=24),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )
    HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=24),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    ).rebuild_vector_index()
    (index_dir / "manifest.json").write_text("{not-json")

    rebuilt = HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=24),
        vector_store=VectorStore(backend="numpy", index_dir=index_dir),
        auto_start_background=False,
    )

    rebuilt.rebuild_vector_index()
    assert rebuilt.retrieve("Corrupt manifest", top_k=2)
    assert json.loads((index_dir / "manifest.json").read_text())["vector_count"] == sqlite_store.count_vector_chunks()


def test_session_delete_removes_vector_chunks(tmp_path):
    sqlite_store = SQLiteStore(tmp_path)
    sqlite_store.save_capture(
        "session-delete",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Delete should remove vectors",
        ),
    )
    retriever = HybridRetriever(
        sqlite_store,
        HashingEmbeddingModel(dimensions=24),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )
    retriever.rebuild_vector_index()
    assert sqlite_store.count_vector_chunks() > 0

    sqlite_store.delete_session("session-delete")
    retriever.rebuild_vector_index()

    assert sqlite_store.count_vector_chunks() == 0
    assert not retriever.retrieve("Delete vectors", top_k=2)


def test_startup_does_not_rebuild_synchronously(tmp_path, monkeypatch):
    calls: list[str] = []
    original = HybridRetriever.rebuild_vector_index

    def tracked_rebuild(self):
        calls.append(threading.current_thread().name)
        time.sleep(0.05)
        return original(self)

    monkeypatch.setattr(HybridRetriever, "rebuild_vector_index", tracked_rebuild)
    store = SQLiteStore(tmp_path)

    HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
    )

    assert "MainThread" not in calls


def test_stale_vector_index_falls_back_to_fts(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "FTS fallback still finds Redis notes"),
    )
    retriever = HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )

    results = retriever.retrieve("Redis", top_k=2)

    assert results
    assert retriever.last_retrieval_status == "using FTS-only fallback"


def test_delete_can_mark_index_dirty_without_blocking_rebuild(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path)
    retriever = HybridRetriever(
        store,
        HashingEmbeddingModel(dimensions=16),
        vector_store=VectorStore(backend="numpy", index_dir=tmp_path / "data" / "vector_index"),
        auto_start_background=False,
    )
    called = []
    monkeypatch.setattr(retriever, "rebuild_vector_index_background", lambda: called.append("background"))

    retriever.mark_vector_index_dirty(rebuild=True)

    assert retriever.vector_index_status == "stale"
    assert called == ["background"]
