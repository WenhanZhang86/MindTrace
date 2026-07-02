from core.chunker import TextChunk
from core.context_builder import ContextBuilder
from core.reranker import RetrievalCandidate


def test_context_builder_respects_budget_and_dedupes():
    chunk = TextChunk("c1", 1, "s1", "2026-07-02T12:00:00", "screen", "FastAPI middleware context")
    duplicate = TextChunk("c2", 2, "s1", "2026-07-02T12:00:01", "screen", "FastAPI middleware context")
    long_chunk = TextChunk("c3", 3, "s1", "2026-07-02T12:00:02", "screen", " ".join(["extra"] * 100))

    context = ContextBuilder(token_budget=20).build(
        [
            RetrievalCandidate(chunk=chunk, unified_score=1.0),
            RetrievalCandidate(chunk=duplicate, unified_score=0.9),
            RetrievalCandidate(chunk=long_chunk, unified_score=0.8),
        ]
    )

    assert context.count("FastAPI middleware context") == 1
    assert "extra" not in context
