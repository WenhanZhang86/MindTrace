import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    capture_id: int | None
    session_id: str
    timestamp: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    def __init__(self, target_tokens: int = 450, overlap_tokens: int = 64) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_capture(
        self,
        text: str,
        session_id: str,
        timestamp: str,
        source: str,
        capture_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        tokens = self._tokens(text)
        if not tokens:
            return []

        chunks: list[TextChunk] = []
        step = max(1, self.target_tokens - self.overlap_tokens)
        for index, start in enumerate(range(0, len(tokens), step)):
            end = min(len(tokens), start + self.target_tokens)
            chunk_text = " ".join(tokens[start:end]).strip()
            if not chunk_text:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=self._chunk_id(session_id, capture_id, index, chunk_text),
                    capture_id=capture_id,
                    session_id=session_id,
                    timestamp=timestamp,
                    source=source,
                    text=chunk_text,
                    metadata={
                        **(metadata or {}),
                        "chunk_index": index,
                        "token_start": start,
                        "token_end": end,
                    },
                )
            )
            if end >= len(tokens):
                break
        return chunks

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[\w\u4e00-\u9fff]+|[^\s]", text)

    def _chunk_id(self, session_id: str, capture_id: int | None, index: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"{session_id}:{capture_id or 'capture'}:{index}:{digest}"
