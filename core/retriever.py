import threading
import time

from .chunker import Chunker, TextChunk
from .embedding import EmbeddingModel
from .logger import get_logger
from .reranker import RetrievalCandidate, SimpleReranker
from .sqlite_store import SQLiteStore
from .vector_store import VectorRecord, VectorStore


logger = get_logger(__name__)


class QueryPlanner:
    def plan(self, question: str) -> str:
        return question.strip()


class HybridRetriever:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore | None = None,
        chunker: Chunker | None = None,
        reranker: SimpleReranker | None = None,
        auto_start_background: bool = True,
    ) -> None:
        self.sqlite_store = sqlite_store
        self.embedding_model = embedding_model
        self.vector_store = vector_store or VectorStore()
        self.chunker = chunker or Chunker()
        self.reranker = reranker or SimpleReranker()
        self.query_planner = QueryPlanner()
        self._indexed_capture_ids: set[int] = set()
        self._lock = threading.RLock()
        self.vector_index_status = "unknown"
        self.last_retrieval_status = ""
        self.auto_start_background = auto_start_background
        self._load_or_schedule_vector_index()

    def retrieve(self, question: str, top_k: int = 8) -> list[RetrievalCandidate]:
        start = time.perf_counter()
        planned_query = self.query_planner.plan(question)

        candidates: dict[str, RetrievalCandidate] = {}
        fts_results = self.sqlite_store.search_captures(planned_query, limit=top_k * 4)
        self._add_fts_candidates(candidates, fts_results)

        if self.vector_index_status != "ready":
            self.last_retrieval_status = (
                "vector index rebuilding" if self.vector_index_status == "rebuilding" else "using FTS-only fallback"
            )
            logger.info(
                "Retrieval completed status=%s elapsed_ms=%.1f",
                self.last_retrieval_status,
                (time.perf_counter() - start) * 1000,
            )
            return self.reranker.rerank(list(candidates.values()), top_k=top_k)

        try:
            query_embedding = self.embedding_model.embed(planned_query)
            with self._lock:
                vector_results = self.vector_store.search(query_embedding, limit=top_k * 4)
            for result in vector_results:
                chunk = self._chunk_from_record(result.record)
                candidate = candidates.get(chunk.chunk_id)
                if candidate:
                    candidate.vector_score = max(candidate.vector_score, result.similarity)
                else:
                    candidates[chunk.chunk_id] = RetrievalCandidate(
                        chunk=chunk,
                        vector_score=result.similarity,
                    )
            self.last_retrieval_status = "using hybrid retrieval"
        except Exception:
            logger.exception("Vector retrieval failed; using FTS-only fallback")
            self.last_retrieval_status = "using FTS-only fallback"

        logger.info(
            "Retrieval completed status=%s elapsed_ms=%.1f",
            self.last_retrieval_status,
            (time.perf_counter() - start) * 1000,
        )
        return self.reranker.rerank(list(candidates.values()), top_k=top_k)

    def index_capture(self, capture: dict) -> None:
        self.index_captures([capture])

    def index_captures(self, captures: list[dict]) -> int:
        records: list[VectorRecord] = []
        indexed_ids: set[int] = set()
        with self._lock:
            for capture in captures:
                capture_id = capture["capture_id"]
                if capture_id in self._indexed_capture_ids:
                    continue
                capture_records = self._records_for_capture(capture)
                records.extend(capture_records)
                indexed_ids.add(capture_id)
            if not records:
                return 0
            self.vector_store.add_many(records, persist=False)
            self.sqlite_store.save_vector_chunks(
                [
                    {
                        "vector_id": record.id,
                        "session_id": record.session_id,
                        "capture_id": record.capture_id,
                        "chunk_id": record.metadata.get("chunk_id", record.id),
                        "timestamp": record.timestamp,
                        "source": record.metadata.get("source", ""),
                        "text": record.metadata.get("text", ""),
                        "embedding_model_name": self.embedding_model.model_name,
                        "embedding_dimension": len(record.embedding),
                    }
                    for record in records
                ]
            )
            self._indexed_capture_ids.update(indexed_ids)
            self.vector_store.save()
            if self.vector_index_status != "rebuilding":
                self.vector_index_status = "ready"
        return len(records)

    def rebuild_vector_index(self) -> None:
        start = time.perf_counter()
        logger.info("Rebuilding vector index")
        self.vector_index_status = "rebuilding"
        with self._lock:
            self.sqlite_store.clear_vector_chunks()
            self.vector_store.clear(persist=False)
            self.vector_store.configure_embedding(
                self.embedding_model.model_name,
                self.embedding_model.dimensions,
            )
            self._indexed_capture_ids = set()
            captures = self.sqlite_store.list_captures(limit=100000)
            for offset in range(0, len(captures), 50):
                self.index_captures(captures[offset : offset + 50])
            self.vector_store.save()
            self.vector_index_status = "ready"
        self.sqlite_store.clear_indexing_queue(status="done")
        logger.info("Vector index rebuild completed elapsed_ms=%.1f", (time.perf_counter() - start) * 1000)

    def mark_vector_index_dirty(self, rebuild: bool = True) -> None:
        self.vector_index_status = "stale"
        if rebuild:
            self.rebuild_vector_index_background()

    def rebuild_vector_index_background(self) -> None:
        if self.vector_index_status == "rebuilding":
            return

        def worker():
            try:
                self.rebuild_vector_index()
            except Exception:
                logger.exception("Background vector rebuild failed")
                self.vector_index_status = "stale"

        threading.Thread(target=worker, daemon=True).start()

    def _load_or_schedule_vector_index(self) -> None:
        self.vector_store.configure_embedding(
            self.embedding_model.model_name,
            self.embedding_model.dimensions,
        )
        metadata_count = self.sqlite_store.count_vector_chunks()
        healthy = self.vector_store.health_check(
            self.embedding_model.model_name,
            self.embedding_model.dimensions,
            metadata_count,
        )
        if healthy:
            self.vector_index_status = "loading"
            if self.auto_start_background:
                threading.Thread(target=self._load_vector_index_background, daemon=True).start()
            return
        self.vector_index_status = "stale"
        if self.auto_start_background:
            self.rebuild_vector_index_background()

    def _load_vector_index_background(self) -> None:
        try:
            metadata_records = self.sqlite_store.list_vector_chunks()
            loaded = self.vector_store.load(
                expected_model_name=self.embedding_model.model_name,
                expected_dimension=self.embedding_model.dimensions,
                metadata_records=metadata_records,
            )
            if not loaded:
                self.vector_index_status = "stale"
                self.rebuild_vector_index_background()
                return
            self._indexed_capture_ids = {
                record.capture_id
                for record in self.vector_store.records.values()
                if record.capture_id is not None
            }
            self.vector_index_status = "ready"
            logger.info("Vector index loaded in background records=%s", len(self.vector_store.records))
        except Exception:
            logger.exception("Vector index background load failed")
            self.vector_index_status = "stale"
            self.rebuild_vector_index_background()

    def _add_fts_candidates(self, candidates: dict[str, RetrievalCandidate], fts_results) -> None:
        for result in fts_results:
            chunk = TextChunk(
                chunk_id=f"fts:{result.session_id}:{result.capture_id}",
                capture_id=result.capture_id,
                session_id=result.session_id,
                timestamp=result.timestamp,
                source=result.source,
                text=result.snippet,
                metadata={"retrieval": "fts"},
            )
            fts_score = self._normalize_fts_score(result.rank)
            candidates[chunk.chunk_id] = RetrievalCandidate(chunk=chunk, fts_score=fts_score)

    def _records_for_capture(self, capture: dict) -> list[VectorRecord]:
        chunks = self.chunker.chunk_capture(
            text=capture["text"],
            session_id=capture["session_id"],
            timestamp=capture["timestamp"],
            source=capture["source"],
            capture_id=capture["capture_id"],
        )
        records = []
        for chunk in chunks:
            embedding = self.embedding_model.embed(chunk.text)
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    embedding=embedding,
                    session_id=chunk.session_id,
                    capture_id=chunk.capture_id,
                    timestamp=chunk.timestamp,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )
        return records

    def _chunk_from_record(self, record: VectorRecord) -> TextChunk:
        return TextChunk(
            chunk_id=record.metadata.get("chunk_id", record.id),
            capture_id=record.capture_id,
            session_id=record.session_id,
            timestamp=record.timestamp,
            source=record.metadata.get("source", ""),
            text=record.metadata.get("text", ""),
            metadata=record.metadata,
        )

    def _normalize_fts_score(self, rank: float | None) -> float:
        if rank is None:
            return 0.6
        return 1.0 / (1.0 + abs(rank))
