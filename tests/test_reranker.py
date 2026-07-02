from core.chunker import TextChunk
from core.reranker import RetrievalCandidate, SimpleReranker


def test_reranker_combines_scores():
    old = RetrievalCandidate(
        chunk=TextChunk("old", 1, "s1", "2020-01-01T00:00:00", "audio", "old text"),
        fts_score=0.1,
        vector_score=0.1,
    )
    strong = RetrievalCandidate(
        chunk=TextChunk("strong", 2, "s2", "2026-07-02T12:00:00", "screen", "strong text"),
        fts_score=0.9,
        vector_score=0.9,
    )

    ranked = SimpleReranker().rerank([old, strong], top_k=2)

    assert ranked[0].chunk.chunk_id == "strong"
    assert ranked[0].unified_score > ranked[1].unified_score
