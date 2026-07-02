import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LLMSettings
from .logger import get_logger


logger = get_logger(__name__)


@dataclass
class CaptureEntry:
    timestamp: str
    source: str
    text: str


class SessionStore:
    def __init__(self, app_dir: Path) -> None:
        self.sessions_dir = app_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.session_id = ""
        self.session_path: Path | None = None
        self.started_at = ""
        self.ended_at = ""
        self.start_time = 0.0
        self.end_time = 0.0
        self.entries: list[CaptureEntry] = []

    def start(self, start_time: float) -> str:
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.session_path = self.sessions_dir / f"{self.session_id}.json"
        self.start_time = start_time
        self.end_time = 0.0
        self.started_at = datetime.fromtimestamp(start_time).isoformat(timespec="seconds")
        self.ended_at = ""
        self.entries = []
        return self.session_id

    def add_entry(self, source: str, text: str) -> CaptureEntry:
        entry = CaptureEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source=source,
            text=text,
        )
        self.entries.append(entry)
        return entry

    def duration_seconds(self, running: bool) -> int:
        if not self.start_time:
            return 0
        end_time = datetime.now().timestamp() if running else self.end_time or datetime.now().timestamp()
        return max(0, int(end_time - self.start_time))

    def save(
        self,
        running: bool,
        max_duration_seconds: int,
        app_version: str,
        llm_settings: LLMSettings,
    ) -> None:
        if not self.session_path:
            return
        now = datetime.now().timestamp()
        if running:
            ended_at = ""
            duration_seconds = self.duration_seconds(running=True)
        else:
            if not self.end_time:
                self.end_time = now
                self.ended_at = datetime.fromtimestamp(self.end_time).isoformat(timespec="seconds")
            ended_at = self.ended_at
            duration_seconds = self.duration_seconds(running=False)

        capture_counts = {
            "audio": sum(1 for entry in self.entries if entry.source == "audio"),
            "screen": sum(1 for entry in self.entries if entry.source == "screen"),
        }
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "app_version": app_version,
            "llm_provider": llm_settings.provider,
            "llm_model": llm_settings.model,
            "capture_counts": capture_counts,
            "running": running,
            "max_duration_seconds": max_duration_seconds,
            "entries": [asdict(entry) for entry in self.entries],
        }
        self.session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info(
            "Session saved",
            extra={
                "session_id": self.session_id,
                "running": running,
                "entries": len(self.entries),
            },
        )

    def load(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def delete_current(self) -> None:
        if self.session_path and self.session_path.exists():
            self.session_path.unlink()
        self.session_id = ""
        self.session_path = None
        self.started_at = ""
        self.ended_at = ""
        self.start_time = 0.0
        self.end_time = 0.0
        self.entries = []

    def joined_context(self, limit_chars: int = 12000) -> str:
        chunks = [f"[{e.timestamp}][{e.source}] {e.text}" for e in self.entries]
        return "\n".join(chunks)[-limit_chars:]
