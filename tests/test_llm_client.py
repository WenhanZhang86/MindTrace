from core.config import LLMSettings
from core.llm_client import LLMClient


def test_llm_client_without_key_does_not_call_api(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    client = LLMClient(LLMSettings(api_key_env="MISSING_TEST_API_KEY"))

    assert not client.is_available


def test_llm_client_initializes_with_mocked_openai(monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("TEST_API_KEY", "test-key")
    monkeypatch.setattr("core.llm_client.OpenAI", FakeOpenAI)

    client = LLMClient(
        LLMSettings(
            provider="openai_compatible",
            api_key_env="TEST_API_KEY",
            base_url="https://example.test/v1",
            model="test-model",
            endpoint="chat",
        )
    )

    assert client.is_available
    assert calls == [{"api_key": "test-key", "base_url": "https://example.test/v1", "timeout": 30.0}]
