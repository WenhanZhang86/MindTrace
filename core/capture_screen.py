import os
import subprocess
import threading
import time
from collections.abc import Callable

import mss
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from .config import AppConfig
from .image_dedupe import FrameDeduper
from .logger import get_logger
from .text_dedupe import TextDeduper, clean_text, is_useful_text


logger = get_logger(__name__)
TextCallback = Callable[[str, str], None]
ErrorCallback = Callable[[str], None]
RunningCallback = Callable[[], bool]
PausedCallback = Callable[[], bool]


class ScreenCapture:
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
        self.deduper = TextDeduper(threshold=0.90)
        self.frame_deduper = FrameDeduper(
            threshold=config.frame_diff_threshold,
            crop_top_ratio=config.subtitle_crop_top_ratio,
            crop_bottom_ratio=config.subtitle_crop_bottom_ratio,
        )

    def start(self) -> None:
        logger.info("Screen OCR capture start")
        threading.Thread(target=self._ocr_loop, daemon=True).start()

    def _ocr_loop(self) -> None:
        if os.path.exists("/opt/homebrew/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

        ocr_language = self._resolve_ocr_language()
        with mss.mss() as sct:
            while self.is_running():
                if self.is_paused():
                    time.sleep(0.5)
                    continue
                if self._is_sensitive_app_active():
                    time.sleep(self.config.screenshot_interval_seconds)
                    continue
                try:
                    cycle_start = time.perf_counter()
                    monitor = sct.monitors[0]
                    shot = sct.grab(monitor)
                    image = Image.frombytes("RGB", shot.size, shot.rgb)
                    if not self.frame_deduper.should_process(image):
                        logger.info(
                            "Screen frame skipped as near-duplicate elapsed_ms=%.1f",
                            (time.perf_counter() - cycle_start) * 1000,
                        )
                        time.sleep(self.config.screenshot_interval_seconds)
                        continue
                    image = self.frame_deduper.crop_region(image)
                    image = self._prepare_ocr_image(image)
                    text = pytesseract.image_to_string(
                        image,
                        lang=ocr_language,
                        config=self.config.ocr_tesseract_config,
                    ).strip()
                    cleaned = clean_text(text)[: self.config.max_entry_chars]
                    if (
                        is_useful_text(cleaned, self.config.min_capture_text_len)
                        and self.deduper.should_store(cleaned)
                    ):
                        self.on_text("screen", cleaned)
                    logger.info("OCR capture cycle elapsed_ms=%.1f", (time.perf_counter() - cycle_start) * 1000)
                except Exception as exc:
                    logger.exception("OCR error")
                    self.on_error(f"OCR failed: {exc}")
                time.sleep(self.config.screenshot_interval_seconds)

    def _prepare_ocr_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.grayscale(image)
        image = ImageEnhance.Contrast(image).enhance(1.8)
        scale = max(0.5, float(getattr(self.config, "ocr_scale_factor", 1.0)))
        if scale == 1.0:
            return image
        width, height = image.size
        return image.resize((int(width * scale), int(height * scale)))

    def _resolve_ocr_language(self) -> str:
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            available = set()
        requested = [lang for lang in self.config.ocr_language.split("+") if lang]
        if requested and all(lang in available for lang in requested):
            return self.config.ocr_language
        if "eng" in available:
            missing = ", ".join(lang for lang in requested if lang not in available)
            if missing:
                logger.warning("OCR language missing (%s); falling back to eng", missing)
                self.on_error(f"OCR language missing ({missing}); falling back to eng.")
            return "eng"
        return self.config.ocr_language

    def _is_sensitive_app_active(self) -> bool:
        active_app = self._frontmost_app_name()
        if not active_app:
            return False
        return any(active_app.lower() == app.lower() for app in self.config.excluded_apps)

    def _frontmost_app_name(self) -> str:
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first application process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
        except Exception:
            return ""
        return result.stdout.strip()
