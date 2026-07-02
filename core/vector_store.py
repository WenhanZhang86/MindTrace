import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VectorRecord:
    id: str
    embedding: list[float]
    session_id: str
    capture_id: int | None
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    record: VectorRecord
    similarity: float


class VectorStore:
    def __init__(self, backend: str = "auto", index_dir: Path | None = None) -> None:
        self.backend_name = "numpy"
        self.records: dict[str, VectorRecord] = {}
        self.embedding_model_name = ""
        self.embedding_dimension: int | None = None
        self.index_dir = index_dir
        self.embeddings_path = index_dir / "embeddings.npy" if index_dir else None
        self.manifest_path = index_dir / "manifest.json" if index_dir else None
        self._faiss = None
        self._faiss_index = None
        self._faiss_ids: list[str] = []
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        if backend in {"auto", "faiss"}:
            try:
                import faiss  # type: ignore

                self._faiss = faiss
                self.backend_name = "faiss"
            except Exception:
                if backend == "faiss":
                    raise

    def add(self, record: VectorRecord, persist: bool = True) -> None:
        self.add_many([record], persist=False)
        if persist:
            self.save()

    def add_many(self, records: list[VectorRecord], persist: bool = True) -> None:
        if not records:
            return
        for record in records:
            self.records[record.id] = self._normalized_record(record)
            self.embedding_dimension = len(record.embedding)
        self._rebuild_faiss()
        if persist:
            self.save()

    def search(self, embedding: list[float], limit: int = 10) -> list[VectorSearchResult]:
        if not self.records:
            return []
        query = self._normalize(np.array(embedding, dtype="float32"))
        if self.backend_name == "faiss" and self._faiss_index is not None:
            scores, indexes = self._faiss_index.search(query.reshape(1, -1), min(limit, len(self._faiss_ids)))
            results = []
            for score, index in zip(scores[0], indexes[0]):
                if index < 0:
                    continue
                record = self.records[self._faiss_ids[index]]
                results.append(VectorSearchResult(record=record, similarity=float(score)))
            return results

        matrix = np.array([record.embedding for record in self.records.values()], dtype="float32")
        scores = matrix @ query
        ordered = np.argsort(scores)[::-1][:limit]
        records = list(self.records.values())
        return [
            VectorSearchResult(record=records[index], similarity=float(scores[index]))
            for index in ordered
        ]

    def delete(self, record_id: str) -> None:
        self.records.pop(record_id, None)
        self._rebuild_faiss()
        self.save()

    def clear(self, persist: bool = True) -> None:
        self.records.clear()
        self._faiss_index = None
        self._faiss_ids = []
        if persist:
            self.save()

    def configure_embedding(self, model_name: str, dimension: int | None) -> None:
        self.embedding_model_name = model_name
        self.embedding_dimension = dimension

    def health_check(
        self,
        expected_model_name: str,
        expected_dimension: int | None,
        metadata_count: int,
    ) -> bool:
        if not self.embeddings_path or not self.manifest_path:
            return False
        if not self.embeddings_path.exists() or not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("embedding_model_name", "") != expected_model_name:
                return False
            dimension = manifest.get("embedding_dimension")
            if expected_dimension is not None and dimension != expected_dimension:
                return False
            if int(manifest.get("vector_count", -1)) != metadata_count:
                return False
            return True
        except Exception:
            return False

    def load(
        self,
        expected_model_name: str,
        expected_dimension: int | None,
        metadata_records: list[dict],
    ) -> bool:
        if not self.embeddings_path or not self.manifest_path:
            return False
        if not self.embeddings_path.exists() or not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text())
            model_name = manifest.get("embedding_model_name", "")
            dimension = manifest.get("embedding_dimension")
            vector_count = int(manifest.get("vector_count", -1))
            if model_name != expected_model_name:
                return False
            if expected_dimension is not None and dimension != expected_dimension:
                return False
            if vector_count != len(metadata_records):
                return False
            matrix = np.load(self.embeddings_path)
            if matrix.ndim != 2:
                return False
            if matrix.shape[0] != len(metadata_records):
                return False
            if dimension is not None and matrix.shape[1] != dimension:
                return False
            records = {}
            for row, item in zip(matrix, metadata_records):
                record = VectorRecord(
                    id=item["vector_id"],
                    embedding=row.astype("float32").tolist(),
                    session_id=item["session_id"],
                    capture_id=item.get("capture_id"),
                    timestamp=item["timestamp"],
                    metadata={
                        "chunk_id": item["chunk_id"],
                        "source": item["source"],
                        "text": item["text"],
                    },
                )
                records[record.id] = self._normalized_record(record)
            self.records = records
            self.embedding_model_name = model_name
            self.embedding_dimension = dimension
            self._rebuild_faiss()
            return True
        except Exception:
            self.records = {}
            self._faiss_index = None
            self._faiss_ids = []
            return False

    def save(self) -> None:
        if not self.embeddings_path or not self.manifest_path:
            return
        records = list(self.records.values())
        if records:
            matrix = np.array([record.embedding for record in records], dtype="float32")
            self.embedding_dimension = matrix.shape[1]
        else:
            dimension = self.embedding_dimension or 0
            matrix = np.empty((0, dimension), dtype="float32")
        np.save(self.embeddings_path, matrix)
        now = datetime.now().isoformat(timespec="seconds")
        existing = {}
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text())
            except Exception:
                existing = {}
        manifest = {
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "vector_count": len(records),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def _normalized_record(self, record: VectorRecord) -> VectorRecord:
        embedding = self._normalize(np.array(record.embedding, dtype="float32")).tolist()
        return VectorRecord(
            id=record.id,
            embedding=embedding,
            session_id=record.session_id,
            capture_id=record.capture_id,
            timestamp=record.timestamp,
            metadata=record.metadata,
        )

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _rebuild_faiss(self) -> None:
        if self.backend_name != "faiss" or not self.records:
            return
        dimension = len(next(iter(self.records.values())).embedding)
        self._faiss_index = self._faiss.IndexFlatIP(dimension)
        self._faiss_ids = list(self.records.keys())
        matrix = np.array([self.records[id_].embedding for id_ in self._faiss_ids], dtype="float32")
        self._faiss_index.add(matrix)
