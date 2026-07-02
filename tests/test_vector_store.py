from core.vector_store import VectorRecord, VectorStore


def test_vector_store_add_search_delete_numpy():
    store = VectorStore(backend="numpy")
    store.add(
        VectorRecord(
            id="a",
            embedding=[1.0, 0.0],
            session_id="s1",
            capture_id=1,
            timestamp="2026-07-02T12:00:00",
            metadata={"text": "alpha"},
        )
    )
    store.add(
        VectorRecord(
            id="b",
            embedding=[0.0, 1.0],
            session_id="s2",
            capture_id=2,
            timestamp="2026-07-02T12:00:01",
            metadata={"text": "beta"},
        )
    )

    results = store.search([1.0, 0.0], limit=1)

    assert results[0].record.id == "a"
    assert results[0].similarity > 0.9

    store.delete("a")
    assert store.search([1.0, 0.0], limit=1)[0].record.id == "b"


def test_vector_store_save_and_load(tmp_path):
    index_dir = tmp_path / "data" / "vector_index"
    store = VectorStore(backend="numpy", index_dir=index_dir)
    store.configure_embedding("hashing-2", 2)
    store.add(
        VectorRecord(
            id="a",
            embedding=[1.0, 0.0],
            session_id="s1",
            capture_id=1,
            timestamp="2026-07-02T12:00:00",
            metadata={"chunk_id": "a", "source": "screen", "text": "alpha"},
        )
    )

    loaded = VectorStore(backend="numpy", index_dir=index_dir)
    metadata = [
        {
            "vector_id": "a",
            "session_id": "s1",
            "capture_id": 1,
            "chunk_id": "a",
            "timestamp": "2026-07-02T12:00:00",
            "source": "screen",
            "text": "alpha",
        }
    ]

    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "manifest.json").exists()
    assert loaded.load("hashing-2", 2, metadata)
    assert loaded.search([1.0, 0.0], limit=1)[0].record.id == "a"


def test_vector_store_dimension_mismatch_rejects_load(tmp_path):
    index_dir = tmp_path / "data" / "vector_index"
    store = VectorStore(backend="numpy", index_dir=index_dir)
    store.configure_embedding("hashing-2", 2)
    store.add(
        VectorRecord(
            id="a",
            embedding=[1.0, 0.0],
            session_id="s1",
            capture_id=1,
            timestamp="2026-07-02T12:00:00",
        )
    )

    loaded = VectorStore(backend="numpy", index_dir=index_dir)

    assert not loaded.load("hashing-3", 3, [])
