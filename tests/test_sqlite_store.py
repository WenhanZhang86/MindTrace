import json
import sqlite3

from core.config import LLMSettings
from core.session_store import CaptureEntry
from core.sqlite_store import SQLiteStore


def test_sqlite_database_initialization(tmp_path):
    store = SQLiteStore(tmp_path)

    assert (tmp_path / "data" / "mindtrace.db").exists()
    with sqlite3.connect(store.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"sessions", "captures", "summaries"}.issubset(tables)
    assert "vector_chunks" in tables
    assert "indexing_queue" in tables
    if store.fts_enabled:
        assert "captures_fts" in tables


def test_sqlite_vector_chunks_metadata(tmp_path):
    store = SQLiteStore(tmp_path)

    store.save_vector_chunk(
        vector_id="v1",
        session_id="s1",
        capture_id=1,
        chunk_id="c1",
        timestamp="2026-07-02T12:00:00",
        source="screen",
        text="chunk text",
        embedding_model_name="hashing-2",
        embedding_dimension=2,
    )

    rows = store.list_vector_chunks()

    assert store.count_vector_chunks() == 1
    assert rows[0]["vector_id"] == "v1"
    assert rows[0]["embedding_model_name"] == "hashing-2"


def test_enqueue_capture_and_status_transitions(tmp_path):
    store = SQLiteStore(tmp_path)
    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "queue text"),
    )

    job_id = store.enqueue_capture(capture_id, "s1")
    job = store.next_indexing_job()
    assert job["id"] == job_id
    assert job["status"] == "processing"

    store.mark_indexing_job_done(job_id)
    assert store.queue_counts()["done"] == 1


def test_failed_job_retry_and_max_attempts(tmp_path):
    store = SQLiteStore(tmp_path)
    capture_id = store.save_capture(
        "s1",
        CaptureEntry("2026-07-02T12:00:00", "screen", "retry text"),
    )
    job_id = store.enqueue_capture(capture_id, "s1")

    job = store.next_indexing_job(max_attempts=3)
    store.mark_indexing_job_failed(job["id"], "boom", max_attempts=3)
    assert store.queue_counts()["pending"] == 1

    for _ in range(2):
        job = store.next_indexing_job(max_attempts=3)
        store.mark_indexing_job_failed(job["id"], "boom", max_attempts=3)

    assert store.queue_counts()["failed"] == 1
    assert store.next_indexing_job(max_attempts=3) is None


def test_sqlite_save_session(tmp_path):
    store = SQLiteStore(tmp_path)

    store.save_session(
        session_id="session-1",
        started_at="2026-07-02T12:00:00",
        ended_at="2026-07-02T12:01:00",
        duration_seconds=60,
        app_version="test-version",
        llm_settings=LLMSettings(provider="deepseek", model="deepseek-v4-flash"),
    )

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT session_id, duration_seconds, llm_provider, llm_model FROM sessions"
        ).fetchone()

    assert row == ("session-1", 60, "deepseek", "deepseek-v4-flash")


def test_sqlite_save_and_search_captures(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_session(
        session_id="session-1",
        started_at="2026-07-02T12:00:00",
        ended_at="",
        duration_seconds=0,
        app_version="test-version",
        llm_settings=LLMSettings(provider="openai", model="gpt-4o-mini"),
    )
    store.save_capture(
        "session-1",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="FastAPI tutorial about middleware and ORM patterns",
        ),
    )
    store.save_capture(
        "session-1",
        CaptureEntry(
            timestamp="2026-07-02T12:00:02",
            source="audio",
            text="Unrelated spoken note",
        ),
    )

    results = store.search_captures("middleware")

    assert len(results) == 1
    assert results[0].session_id == "session-1"
    assert results[0].source == "screen"
    assert "middleware" in results[0].snippet


def test_sqlite_fts_search_returns_expected_capture(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_capture(
        "session-fts",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="LangChain retrieval augmented generation and FastAPI backend",
        ),
    )

    results = store.search_captures("LangChain")

    assert results
    assert results[0].session_id == "session-fts"
    assert "[" in results[0].snippet or "LangChain" in results[0].snippet


def test_sqlite_like_fallback_search(tmp_path):
    store = SQLiteStore(tmp_path)
    store.fts_enabled = False
    store.save_capture(
        "session-like",
        CaptureEntry(
            timestamp="2026-07-02T12:00:01",
            source="screen",
            text="Fallback search should still find middleware",
        ),
    )

    results = store.search_captures("middleware")

    assert len(results) == 1
    assert results[0].rank is None
    assert "middleware" in results[0].snippet


def test_sqlite_save_summary(tmp_path):
    store = SQLiteStore(tmp_path)

    store.save_summary("session-1", "A useful session summary")

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute("SELECT session_id, summary FROM summaries").fetchone()

    assert row == ("session-1", "A useful session summary")


def test_import_json_sessions_imports_captures_and_skips_duplicates(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    payload = {
        "session_id": "json-session-1",
        "started_at": "2026-07-02T12:00:00",
        "ended_at": "2026-07-02T12:05:00",
        "duration_seconds": 300,
        "app_version": "test-version",
        "llm_provider": "deepseek",
        "llm_model": "deepseek-v4-flash",
        "entries": [
            {
                "timestamp": "2026-07-02T12:00:01",
                "source": "screen",
                "text": "Imported FastAPI capture",
            },
            {
                "timestamp": "2026-07-02T12:00:02",
                "source": "audio",
                "text": "Imported audio capture",
            },
        ],
    }
    (sessions_dir / "json-session-1.json").write_text(json.dumps(payload))
    store = SQLiteStore(tmp_path)

    first = store.import_json_sessions(sessions_dir)
    second = store.import_json_sessions(sessions_dir)

    assert first.sessions == 1
    assert first.captures == 2
    assert second.sessions == 0
    assert second.captures == 0
    assert second.skipped_sessions == 1
    assert len(store.search_captures("FastAPI")) == 1
