from core.config import LLMSettings
from core.session_store import SessionStore


def test_session_store_save_and_load(tmp_path):
    store = SessionStore(tmp_path)
    session_id = store.start(start_time=1000.0)
    store.add_entry("audio", "hello from audio")
    store.add_entry("screen", "hello from screen")

    store.save(
        running=False,
        max_duration_seconds=7200,
        app_version="test-version",
        llm_settings=LLMSettings(provider="deepseek", model="deepseek-v4-flash"),
    )

    assert store.session_path is not None
    data = store.load(store.session_path)

    assert data["session_id"] == session_id
    assert data["app_version"] == "test-version"
    assert data["llm_provider"] == "deepseek"
    assert data["llm_model"] == "deepseek-v4-flash"
    assert data["capture_counts"] == {"audio": 1, "screen": 1}
    assert data["ended_at"]
    assert data["duration_seconds"] >= 0
    assert len(data["entries"]) == 2
