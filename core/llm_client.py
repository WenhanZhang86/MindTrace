import os
import time

from openai import OpenAI

from .config import LLMSettings
from .logger import get_logger


logger = get_logger(__name__)
SYSTEM_PROMPT = (
    "You are a careful context analyst. The input is noisy OCR and speech "
    "transcription. Preserve uncertainty, extract useful evidence, and answer "
    "in the user's language when possible."
)


class LLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.client = self._build_client()

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def _build_client(self):
        key = os.getenv(self.settings.api_key_env, "")
        if not key:
            return None
        kwargs = {"api_key": key}
        if self.settings.base_url:
            kwargs["base_url"] = self.settings.base_url
        if self.settings.timeout_seconds:
            kwargs["timeout"] = self.settings.timeout_seconds
        return OpenAI(**kwargs)

    def summarize(self, text: str) -> str:
        return self._call(text)

    def answer(self, question: str, context: str) -> str:
        return self._call(f"Question: {question}\n\nCaptured context:\n{context}")

    def run_prompt(self, prompt: str) -> str:
        return self._call(prompt)

    def _call(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError(f"{self.settings.api_key_env} missing")

        logger.info(
            "LLM request start provider=%s model=%s endpoint=%s",
            self.settings.provider,
            self.settings.model,
            self.settings.endpoint,
        )
        start = time.perf_counter()
        if self.settings.endpoint == "responses":
            try:
                response = self.client.responses.create(
                    model=self.settings.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.output_text.strip()
                logger.info(
                    "LLM request end provider=%s model=%s elapsed_ms=%.1f",
                    self.settings.provider,
                    self.settings.model,
                    (time.perf_counter() - start) * 1000,
                )
                return text
            except Exception:
                logger.exception("LLM request error")
                raise

        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            logger.info(
                "LLM request end provider=%s model=%s elapsed_ms=%.1f",
                self.settings.provider,
                self.settings.model,
                (time.perf_counter() - start) * 1000,
            )
            return text
        except Exception:
            logger.exception("LLM request error")
            raise
