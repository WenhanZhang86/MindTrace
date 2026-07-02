import hashlib
import math
import os
import re
from abc import ABC, abstractmethod

from openai import OpenAI


class EmbeddingModel(ABC):
    model_name = "embedding-model"
    dimensions: int | None = None

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashingEmbeddingModel(EmbeddingModel):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.model_name = f"hashing-{dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        key = os.getenv(api_key_env, "")
        if not key:
            raise RuntimeError(f"{api_key_env} missing")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.model_name = f"openai:{model}"
        self.dimensions = None

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class SentenceTransformersEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        raise RuntimeError(
            "sentence-transformers embeddings are not bundled yet. "
            f"Requested model: {model}"
        )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


def create_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    dimensions: int = 384,
) -> EmbeddingModel:
    provider = (provider or os.getenv("EMBEDDING_PROVIDER") or "hashing").lower()
    model = model or os.getenv("EMBEDDING_MODEL")
    api_key_env = api_key_env or os.getenv("EMBEDDING_API_KEY_ENV")
    base_url = base_url or os.getenv("EMBEDDING_BASE_URL")

    if provider == "openai":
        return OpenAIEmbeddingModel(
            model=model or "text-embedding-3-small",
            api_key_env=api_key_env or "OPENAI_API_KEY",
            base_url=base_url,
        )
    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        return SentenceTransformersEmbeddingModel(model=model or "all-MiniLM-L6-v2")
    return HashingEmbeddingModel(dimensions=dimensions)
