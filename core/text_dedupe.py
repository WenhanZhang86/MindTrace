import difflib
import re
from collections import deque


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return re.sub(r"\s+", " ", text).strip()


def is_useful_text(text: str, min_length: int = 18) -> bool:
    if not text or len(text) < min_length:
        return False
    signal_chars = sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text)
    return signal_chars / max(len(text), 1) >= 0.25


class TextDeduper:
    def __init__(self, threshold: float = 0.90, max_recent: int = 5) -> None:
        self.threshold = threshold
        self.recent: deque[str] = deque(maxlen=max_recent)

    def is_duplicate(self, text: str) -> bool:
        for previous in self.recent:
            if difflib.SequenceMatcher(None, previous, text).ratio() >= self.threshold:
                return True
        return False

    def remember(self, text: str) -> None:
        self.recent.append(text)

    def should_store(self, text: str) -> bool:
        if self.is_duplicate(text):
            return False
        self.remember(text)
        return True
