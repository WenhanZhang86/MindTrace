from core.chunker import Chunker


def test_chunker_splits_and_preserves_metadata():
    chunker = Chunker(target_tokens=10, overlap_tokens=2)
    text = " ".join(f"word{i}" for i in range(25))

    chunks = chunker.chunk_capture(
        text=text,
        session_id="session-1",
        timestamp="2026-07-02T12:00:00",
        source="screen",
        capture_id=7,
        metadata={"title": "demo"},
    )

    assert len(chunks) > 1
    assert chunks[0].session_id == "session-1"
    assert chunks[0].capture_id == 7
    assert chunks[0].metadata["title"] == "demo"
    assert chunks[1].metadata["token_start"] < chunks[0].metadata["token_end"]
