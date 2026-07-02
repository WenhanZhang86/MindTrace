import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import LLMSettings
from .logger import get_logger
from .session_store import CaptureEntry


logger = get_logger(__name__)


@dataclass
class SearchResult:
    session_id: str
    timestamp: str
    source: str
    snippet: str
    capture_id: int | None = None
    rank: float | None = None


@dataclass
class ImportResult:
    sessions: int = 0
    captures: int = 0
    summaries: int = 0
    skipped_sessions: int = 0


class SQLiteStore:
    def __init__(self, app_dir: Path, db_path: Path | None = None) -> None:
        self.data_dir = app_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = db_path or self.data_dir / "mindtrace.db"
        self.fts_enabled = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    ended_at TEXT,
                    duration_seconds REAL,
                    app_version TEXT,
                    llm_provider TEXT,
                    llm_model TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    source TEXT,
                    text TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    summary TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    vector_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    capture_id INTEGER,
                    chunk_id TEXT,
                    timestamp TEXT,
                    source TEXT,
                    text TEXT,
                    embedding_model_name TEXT,
                    embedding_dimension INTEGER,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexing_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id INTEGER,
                    session_id TEXT,
                    status TEXT,
                    attempts INTEGER,
                    error TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_session_id ON captures(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_text ON captures(text)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_session_id ON vector_chunks(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_capture_id ON vector_chunks(capture_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_indexing_queue_status ON indexing_queue(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_indexing_queue_capture_id ON indexing_queue(capture_id)")
            self.fts_enabled = self._init_fts(conn)
        logger.info("SQLite database initialized at %s", self.db_path)

    def _init_fts(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts
                USING fts5(
                    text,
                    session_id UNINDEXED,
                    timestamp UNINDEXED,
                    source UNINDEXED,
                    capture_id UNINDEXED
                )
                """
            )
            return True
        except sqlite3.OperationalError:
            logger.warning("SQLite FTS5 unavailable; falling back to LIKE search")
            return False

    def _session_exists(self, conn: sqlite3.Connection, session_id: str) -> bool:
        row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row is not None

    def save_session(
        self,
        session_id: str,
        started_at: str,
        ended_at: str,
        duration_seconds: float,
        app_version: str,
        llm_settings: LLMSettings,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, started_at, ended_at, duration_seconds,
                    app_version, llm_provider, llm_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_seconds = excluded.duration_seconds,
                    app_version = excluded.app_version,
                    llm_provider = excluded.llm_provider,
                    llm_model = excluded.llm_model
                """,
                (
                    session_id,
                    started_at,
                    ended_at,
                    duration_seconds,
                    app_version,
                    llm_settings.provider,
                    llm_settings.model,
                ),
            )

    def save_capture(self, session_id: str, entry: CaptureEntry) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO captures (session_id, timestamp, source, text)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, entry.timestamp, entry.source, entry.text),
            )
            self._save_capture_fts(conn, cursor.lastrowid, session_id, entry)
            return int(cursor.lastrowid)

    def _save_capture_fts(
        self,
        conn: sqlite3.Connection,
        capture_id: int,
        session_id: str,
        entry: CaptureEntry,
    ) -> None:
        if not self.fts_enabled:
            return
        try:
            conn.execute(
                """
                INSERT INTO captures_fts (rowid, text, session_id, timestamp, source, capture_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (capture_id, entry.text, session_id, entry.timestamp, entry.source, capture_id),
            )
        except sqlite3.OperationalError:
            logger.exception("FTS insert failed; future searches will use LIKE fallback")
            self.fts_enabled = False

    def save_summary(self, session_id: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (session_id, timestamp, summary)
                VALUES (?, ?, ?)
                """,
                (session_id, datetime.now().isoformat(timespec="seconds"), summary),
            )

    def delete_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM vector_chunks WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM indexing_queue WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM captures WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            if self.fts_enabled:
                try:
                    conn.execute("DELETE FROM captures_fts WHERE session_id = ?", (session_id,))
                except sqlite3.OperationalError:
                    logger.exception("FTS delete failed")
                    self.fts_enabled = False

    def search_captures(self, query: str, limit: int = 20) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        if self.fts_enabled:
            try:
                return self._search_captures_fts(query, limit)
            except sqlite3.OperationalError:
                logger.exception("FTS search failed; falling back to LIKE search")
                self.fts_enabled = False
        return self._search_captures_like(query, limit)

    def _search_captures_fts(self, query: str, limit: int) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    capture_id,
                    session_id,
                    timestamp,
                    source,
                    snippet(captures_fts, 0, '[', ']', '...', 24) AS snippet_text,
                    bm25(captures_fts) AS rank
                FROM captures_fts
                WHERE captures_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [
            SearchResult(
                capture_id=row[0],
                session_id=row[1],
                timestamp=row[2],
                source=row[3],
                snippet=row[4],
                rank=row[5],
            )
            for row in rows
        ]

    def _search_captures_like(self, query: str, limit: int) -> list[SearchResult]:
        pattern = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, timestamp, source, text
                FROM captures
                WHERE text LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
        return [
            SearchResult(
                capture_id=row[0],
                session_id=row[1],
                timestamp=row[2],
                source=row[3],
                snippet=self._snippet(row[4], query),
                rank=None,
            )
            for row in rows
        ]

    def list_captures(self, limit: int = 1000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, timestamp, source, text
                FROM captures
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "capture_id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "source": row[3],
                "text": row[4],
            }
            for row in rows
        ]

    def clear_vector_chunks(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vector_chunks")

    def save_vector_chunk(
        self,
        vector_id: str,
        session_id: str,
        capture_id: int | None,
        chunk_id: str,
        timestamp: str,
        source: str,
        text: str,
        embedding_model_name: str,
        embedding_dimension: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vector_chunks (
                    vector_id, session_id, capture_id, chunk_id, timestamp, source, text,
                    embedding_model_name, embedding_dimension, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vector_id,
                    session_id,
                    capture_id,
                    chunk_id,
                    timestamp,
                    source,
                    text,
                    embedding_model_name,
                    embedding_dimension,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def save_vector_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            (
                item["vector_id"],
                item["session_id"],
                item.get("capture_id"),
                item["chunk_id"],
                item["timestamp"],
                item["source"],
                item["text"],
                item["embedding_model_name"],
                item["embedding_dimension"],
                now,
            )
            for item in chunks
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO vector_chunks (
                    vector_id, session_id, capture_id, chunk_id, timestamp, source, text,
                    embedding_model_name, embedding_dimension, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def list_vector_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT vector_id, session_id, capture_id, chunk_id, timestamp, source, text,
                       embedding_model_name, embedding_dimension, created_at
                FROM vector_chunks
                ORDER BY rowid
                """
            ).fetchall()
        return [
            {
                "vector_id": row[0],
                "session_id": row[1],
                "capture_id": row[2],
                "chunk_id": row[3],
                "timestamp": row[4],
                "source": row[5],
                "text": row[6],
                "embedding_model_name": row[7],
                "embedding_dimension": row[8],
                "created_at": row[9],
            }
            for row in rows
        ]

    def count_vector_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM vector_chunks").fetchone()
        return int(row[0])

    def enqueue_capture(self, capture_id: int, session_id: str) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM indexing_queue
                WHERE capture_id = ? AND status IN ('pending', 'processing', 'done')
                """,
                (capture_id,),
            ).fetchone()
            if existing:
                return int(existing[0])
            cursor = conn.execute(
                """
                INSERT INTO indexing_queue (capture_id, session_id, status, attempts, error, created_at, updated_at)
                VALUES (?, ?, 'pending', 0, '', ?, ?)
                """,
                (capture_id, session_id, now, now),
            )
            return int(cursor.lastrowid)

    def next_indexing_jobs(self, limit: int = 10, max_attempts: int = 3) -> list[dict]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, capture_id, session_id, status, attempts, error, created_at, updated_at
                FROM indexing_queue
                WHERE status = 'pending' OR (status = 'failed' AND attempts < ?)
                ORDER BY created_at
                LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [row[0] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE indexing_queue
                SET status = 'processing', updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (now, *ids),
            )
        return [
            {
                "id": row[0],
                "capture_id": row[1],
                "session_id": row[2],
                "status": "processing",
                "attempts": row[4],
                "error": row[5],
                "created_at": row[6],
                "updated_at": now,
            }
            for row in rows
        ]

    def next_indexing_job(self, max_attempts: int = 3) -> dict | None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, capture_id, session_id, status, attempts, error, created_at, updated_at
                FROM indexing_queue
                WHERE status = 'pending' OR (status = 'failed' AND attempts < ?)
                ORDER BY created_at
                LIMIT 1
                """,
                (max_attempts,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE indexing_queue
                SET status = 'processing', updated_at = ?
                WHERE id = ?
                """,
                (now, row[0]),
            )
        return {
            "id": row[0],
            "capture_id": row[1],
            "session_id": row[2],
            "status": "processing",
            "attempts": row[4],
            "error": row[5],
            "created_at": row[6],
            "updated_at": now,
        }

    def mark_indexing_job_done(self, job_id: int) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE indexing_queue
                SET status = 'done', error = '', updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )

    def mark_indexing_jobs_done(self, job_ids: list[int]) -> None:
        if not job_ids:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE indexing_queue
                SET status = 'done', error = '', updated_at = ?
                WHERE id = ?
                """,
                [(now, job_id) for job_id in job_ids],
            )

    def mark_indexing_job_failed(self, job_id: int, error: str, max_attempts: int = 3) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute("SELECT attempts FROM indexing_queue WHERE id = ?", (job_id,)).fetchone()
            attempts = int(row[0]) + 1 if row else 1
            status = "failed" if attempts >= max_attempts else "pending"
            conn.execute(
                """
                UPDATE indexing_queue
                SET status = ?, attempts = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, attempts, error, now, job_id),
            )

    def get_capture(self, capture_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, timestamp, source, text
                FROM captures
                WHERE id = ?
                """,
                (capture_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "capture_id": row[0],
            "session_id": row[1],
            "timestamp": row[2],
            "source": row[3],
            "text": row[4],
        }

    def queue_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*)
                FROM indexing_queue
                GROUP BY status
                """
            ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def clear_indexing_queue(self, status: str = "done") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE indexing_queue
                SET status = ?, error = '', updated_at = ?
                WHERE status IN ('pending', 'processing', 'failed')
                """,
                (status, now),
            )

    def import_json_sessions(self, sessions_dir: str | Path = "sessions") -> ImportResult:
        source_dir = Path(sessions_dir)
        if not source_dir.is_absolute():
            source_dir = self.db_path.parent.parent / source_dir
        result = ImportResult()
        if not source_dir.exists():
            return result

        for path in sorted(source_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                logger.exception("Failed to read session JSON: %s", path)
                continue

            session_id = payload.get("session_id")
            if not session_id:
                continue

            with self._connect() as conn:
                if self._session_exists(conn, session_id):
                    result.skipped_sessions += 1
                    continue

            llm_settings = LLMSettings(
                provider=payload.get("llm_provider", ""),
                model=payload.get("llm_model", ""),
            )
            self.save_session(
                session_id=session_id,
                started_at=payload.get("started_at", ""),
                ended_at=payload.get("ended_at", ""),
                duration_seconds=float(payload.get("duration_seconds", 0) or 0),
                app_version=payload.get("app_version", ""),
                llm_settings=llm_settings,
            )
            result.sessions += 1

            for item in payload.get("entries", []):
                entry = CaptureEntry(
                    timestamp=item.get("timestamp", ""),
                    source=item.get("source", ""),
                    text=item.get("text", ""),
                )
                self.save_capture(session_id, entry)
                result.captures += 1

            summary = payload.get("summary")
            if summary:
                self.save_summary(session_id, summary)
                result.summaries += 1

        return result

    def _snippet(self, text: str, query: str, radius: int = 80) -> str:
        lower_text = text.lower()
        lower_query = query.lower()
        index = lower_text.find(lower_query)
        if index < 0:
            return text[: radius * 2]
        start = max(0, index - radius)
        end = min(len(text), index + len(query) + radius)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"
