from dataclasses import dataclass
from datetime import datetime

from .chunker import TextChunk


@dataclass
class RetrievalCandidate:
    chunk: TextChunk
    fts_score: float = 0.0
    vector_score: float = 0.0
    unified_score: float = 0.0


class SimpleReranker:
    def __init__(
        self,
        fts_weight: float = 0.35,
        vector_weight: float = 0.45,
        recency_weight: float = 0.15,
        source_weight: float = 0.05,
    ) -> None:
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight
        self.recency_weight = recency_weight
        self.source_weight = source_weight

    def rerank(self, candidates: list[RetrievalCandidate], top_k: int = 8) -> list[RetrievalCandidate]:
        for candidate in candidates:
            recency = self._recency_bonus(candidate.chunk.timestamp)
            source = self._source_bonus(candidate.chunk.source)
            candidate.unified_score = (
                self.fts_weight * candidate.fts_score
                + self.vector_weight * candidate.vector_score
                + self.recency_weight * recency
                + self.source_weight * source
            )
        return sorted(candidates, key=lambda item: item.unified_score, reverse=True)[:top_k]

    def _recency_bonus(self, timestamp: str) -> float:
        try:
            then = datetime.fromisoformat(timestamp).timestamp()
        except Exception:
            return 0.0
        age_days = max(0.0, (datetime.now().timestamp() - then) / 86400)
        return 1.0 / (1.0 + age_days)

    def _source_bonus(self, source: str) -> float:
        if source == "screen":
            return 1.0
        if source == "audio":
            return 0.8
        return 0.5
