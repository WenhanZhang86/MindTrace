from .reranker import RetrievalCandidate


class ContextBuilder:
    def __init__(self, token_budget: int = 1800) -> None:
        self.token_budget = token_budget

    def build(self, candidates: list[RetrievalCandidate]) -> str:
        chunks: list[str] = []
        seen: set[str] = set()
        used_tokens = 0
        for candidate in candidates:
            text = candidate.chunk.text.strip()
            if not text:
                continue
            fingerprint = " ".join(text.lower().split()[:40])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            token_count = self._estimate_tokens(text)
            if used_tokens + token_count > self.token_budget:
                break
            used_tokens += token_count
            chunks.append(
                "\n".join(
                    [
                        f"[source={candidate.chunk.source}]",
                        f"score={candidate.unified_score:.3f} fts={candidate.fts_score:.3f} vector={candidate.vector_score:.3f}",
                        text,
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))
