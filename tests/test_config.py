import json

from core.config import load_config


def test_load_config_reads_json(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "max_duration_seconds": 123,
                "audio_chunk_seconds": 3,
                "screenshot_interval_seconds": 0.5,
                "frame_diff_threshold": 0.03,
                "llm": {
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                },
            }
        )
    )

    config, messages = load_config(tmp_path)

    assert config.max_duration_seconds == 123
    assert config.audio_chunk_seconds == 3
    assert config.screenshot_interval_seconds == 0.5
    assert config.frame_diff_threshold == 0.03
    assert config.llm.provider == "deepseek"
    assert config.llm.model == "deepseek-v4-flash"
    assert "Loaded config.json" in messages


def test_env_var_override_wins(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                }
            }
        )
    )
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setenv("LLM_MODEL", "grok-test-model")

    config, _ = load_config(tmp_path)

    assert config.llm.provider == "grok"
    assert config.llm.api_key_env == "XAI_API_KEY"
    assert config.llm.model == "grok-test-model"
