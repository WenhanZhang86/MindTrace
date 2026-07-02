from core.embedding import EmbeddingModel, HashingEmbeddingModel, create_embedding_model


def test_hashing_embedding_model_implements_interface():
    model = HashingEmbeddingModel(dimensions=32)

    vector = model.embed("FastAPI middleware memory")

    assert isinstance(model, EmbeddingModel)
    assert len(vector) == 32
    assert any(value != 0 for value in vector)


def test_create_embedding_model_defaults_to_hashing(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    model = create_embedding_model(dimensions=16)

    assert isinstance(model, HashingEmbeddingModel)
    assert len(model.embed("hello")) == 16
