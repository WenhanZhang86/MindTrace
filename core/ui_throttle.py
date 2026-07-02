import time


class CaptureLogThrottler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.last_flush = 0.0
        self.counts: dict[str, int] = {}
        self.last_preview = ""

    def add(self, source: str, preview: str) -> None:
        self.counts[source] = self.counts.get(source, 0) + 1
        self.last_preview = f"[{source}] {preview}"

    def should_flush(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return bool(self.counts) and (now - self.last_flush) >= self.interval_seconds

    def flush(self, now: float | None = None) -> str:
        now = now if now is not None else time.time()
        self.last_flush = now
        if not self.counts:
            return ""
        summary = ", ".join(f"{source}: {count}" for source, count in sorted(self.counts.items()))
        message = f"Captured {summary}. Latest {self.last_preview}"
        self.counts = {}
        self.last_preview = ""
        return message
