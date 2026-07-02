import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROVIDER_DEFAULTS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "",
        "model": "gpt-4o-mini",
        "endpoint": "responses",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "endpoint": "chat",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash",
        "endpoint": "chat",
    },
    "grok": {
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4.3",
        "endpoint": "chat",
    },
    "openai_compatible": {
        "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
        "base_url": "",
        "model": "",
        "endpoint": "chat",
    },
}

APP_VERSION = "0.2.0"


@dataclass
class LLMSettings:
    provider: str = "openai"
    api_key_env: str = PROVIDER_DEFAULTS["openai"]["api_key_env"]
    base_url: str = PROVIDER_DEFAULTS["openai"]["base_url"]
    model: str = PROVIDER_DEFAULTS["openai"]["model"]
    endpoint: str = PROVIDER_DEFAULTS["openai"]["endpoint"]
    timeout_seconds: float = 30.0


@dataclass
class AppConfig:
    app_version: str = APP_VERSION
    max_duration_seconds: int = 60 * 60 * 2
    audio_chunk_seconds: int = 7
    ocr_interval_seconds: int = 8
    screenshot_interval_seconds: float = 0.5
    sample_rate: int = 16000
    input_device: Any = None
    min_capture_text_len: int = 18
    audio_language: str | None = None
    ocr_language: str = "eng+chi_sim"
    ocr_tesseract_config: str = "--psm 11"
    ocr_scale_factor: float = 1.0
    frame_diff_threshold: float = 0.02
    subtitle_crop_top_ratio: float = 0.55
    subtitle_crop_bottom_ratio: float = 0.92
    max_entry_chars: int = 3000
    context_limit_chars: int = 12000
    excluded_apps: list[str] = field(default_factory=lambda: ["1Password", "Keychain Access"])
    llm: LLMSettings = field(default_factory=LLMSettings)


def configure_llm(
    settings: LLMSettings,
    provider: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
) -> None:
    if provider:
        provider = provider.lower()
        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai_compatible"])
        settings.provider = provider
        settings.api_key_env = defaults["api_key_env"]
        settings.base_url = defaults["base_url"]
        settings.model = defaults["model"]
        settings.endpoint = defaults["endpoint"]
    if api_key_env:
        settings.api_key_env = api_key_env
    if base_url is not None:
        settings.base_url = base_url
    if model:
        settings.model = model
    if endpoint:
        settings.endpoint = endpoint.lower()
    if timeout_seconds is not None:
        settings.timeout_seconds = float(timeout_seconds)


def load_config(app_dir: Path) -> tuple[AppConfig, list[str]]:
    config = AppConfig()
    messages: list[str] = []
    cfg_path = app_dir / "config.json"

    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text())
            config.max_duration_seconds = int(raw.get("max_duration_seconds", config.max_duration_seconds))
            config.audio_chunk_seconds = int(raw.get("audio_chunk_seconds", config.audio_chunk_seconds))
            config.ocr_interval_seconds = int(raw.get("ocr_interval_seconds", config.ocr_interval_seconds))
            config.screenshot_interval_seconds = float(raw.get("screenshot_interval_seconds", config.screenshot_interval_seconds))
            config.sample_rate = int(raw.get("sample_rate", config.sample_rate))
            config.input_device = raw.get("input_device", config.input_device)
            config.min_capture_text_len = int(raw.get("min_capture_text_len", config.min_capture_text_len))
            config.audio_language = raw.get("audio_language", config.audio_language)
            config.ocr_language = raw.get("ocr_language", config.ocr_language)
            config.ocr_tesseract_config = raw.get("ocr_tesseract_config", config.ocr_tesseract_config)
            config.ocr_scale_factor = float(raw.get("ocr_scale_factor", config.ocr_scale_factor))
            config.frame_diff_threshold = float(raw.get("frame_diff_threshold", config.frame_diff_threshold))
            config.subtitle_crop_top_ratio = float(raw.get("subtitle_crop_top_ratio", config.subtitle_crop_top_ratio))
            config.subtitle_crop_bottom_ratio = float(raw.get("subtitle_crop_bottom_ratio", config.subtitle_crop_bottom_ratio))
            config.max_entry_chars = int(raw.get("max_entry_chars", config.max_entry_chars))
            config.context_limit_chars = int(raw.get("context_limit_chars", config.context_limit_chars))
            excluded_apps = raw.get("excluded_apps", config.excluded_apps)
            if isinstance(excluded_apps, list):
                config.excluded_apps = [str(app) for app in excluded_apps if str(app).strip()]

            llm_cfg = raw.get("llm", {})
            if isinstance(llm_cfg, dict):
                configure_llm(
                    config.llm,
                    provider=llm_cfg.get("provider"),
                    api_key_env=llm_cfg.get("api_key_env"),
                    base_url=llm_cfg.get("base_url"),
                    model=llm_cfg.get("model"),
                    endpoint=llm_cfg.get("endpoint"),
                    timeout_seconds=llm_cfg.get("timeout_seconds"),
                )
            messages.append("Loaded config.json")
        except Exception as exc:
            messages.append(f"Config load error: {exc}")

    configure_llm(
        config.llm,
        provider=os.getenv("LLM_PROVIDER"),
        api_key_env=os.getenv("LLM_API_KEY_ENV"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        endpoint=os.getenv("LLM_ENDPOINT"),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS")) if os.getenv("LLM_TIMEOUT_SECONDS") else None,
    )
    return config, messages
