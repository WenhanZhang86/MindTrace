import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from .config import AppConfig
from .logger import get_logger
from .text_dedupe import clean_text, is_useful_text


logger = get_logger(__name__)
TextCallback = Callable[[str, str], None]
ErrorCallback = Callable[[str], None]
RunningCallback = Callable[[], bool]
PausedCallback = Callable[[], bool]


class AudioCapture:
    def __init__(
        self,
        config: AppConfig,
        on_text: TextCallback,
        on_error: ErrorCallback,
        is_running: RunningCallback,
        is_paused: PausedCallback | None = None,
    ) -> None:
        self.config = config
        self.on_text = on_text
        self.on_error = on_error
        self.is_running = is_running
        self.is_paused = is_paused or (lambda: False)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.whisper_model = None
        self._last_audio_text = ""

    def start(self) -> None:
        logger.info("Audio capture start")
        threading.Thread(target=self._audio_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()

    def _audio_loop(self) -> None:
        while self.is_running():
            if self.is_paused():
                time.sleep(0.5)
                continue
            try:
                frames = int(self.config.sample_rate * self.config.audio_chunk_seconds)
                audio = sd.rec(
                    frames,
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.config.input_device,
                )
                sd.wait()
                if self.is_running():
                    try:
                        self.audio_queue.put_nowait(audio.copy())
                    except queue.Full:
                        logger.warning("Audio queue full; dropping oldest audio chunk")
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.audio_queue.put_nowait(audio.copy())
            except Exception as exc:
                logger.exception("Audio capture error")
                self.on_error(f"Audio capture failed: {exc}")
                time.sleep(2)

    def _transcribe_loop(self) -> None:
        while self.is_running():
            if self.is_paused():
                time.sleep(0.5)
                continue
            try:
                audio = self.audio_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                model = self._get_whisper_model()
                samples = np.squeeze(audio).astype(np.float32)
                start = time.perf_counter()
                segments, _ = model.transcribe(
                    samples,
                    language=self.config.audio_language,
                )
                logger.info("Audio transcription completed elapsed_ms=%.1f", (time.perf_counter() - start) * 1000)
                text = clean_text(" ".join(seg.text.strip() for seg in segments).strip())
                if (
                    is_useful_text(text, self.config.min_capture_text_len)
                    and text != self._last_audio_text
                ):
                    self._last_audio_text = text
                    self.on_text("audio", text)
            except Exception as exc:
                logger.exception("Audio transcription error")
                self.on_error(f"Audio transcription failed: {exc}")

    def _get_whisper_model(self):
        if self.whisper_model is None:
            start = time.perf_counter()
            logger.info("Loading Whisper model in background")
            self.whisper_model = WhisperModel("base", compute_type="int8")
            logger.info("Whisper model loaded elapsed_ms=%.1f", (time.perf_counter() - start) * 1000)
        return self.whisper_model
