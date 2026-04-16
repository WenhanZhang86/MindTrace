import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, scrolledtext

import mss
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from openai import OpenAI


APP_DIR = Path(__file__).parent
SESSIONS_DIR = APP_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


@dataclass
class CaptureEntry:
    timestamp: str
    source: str
    text: str


class ContextAssistant:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Context Assistant")
        self.root.geometry("980x720")

        self.running = False
        self.start_time = 0.0
        self.max_duration_seconds = 60 * 60 * 2
        self.audio_chunk_seconds = 7
        self.ocr_interval_seconds = 8
        self.sample_rate = 16000
        self.input_device = None
        self.min_capture_text_len = 18

        self.audio_queue = queue.Queue()
        self.capture_entries: List[CaptureEntry] = []
        self.session_id = ""
        self.session_path: Path | None = None
        self._last_screen_text = ""
        self._last_audio_text = ""

        self._build_ui()
        self.whisper_model = WhisperModel("base", compute_type="int8")
        self.openai_client = self._init_openai()
        self._load_config()
        self._ui_log("Ready. Configure permissions and click Start.")

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=12, pady=10)

        self.start_btn = tk.Button(top, text="Start to Work", width=18, command=self.start_work)
        self.start_btn.pack(side=tk.LEFT, padx=6)

        self.end_btn = tk.Button(top, text="End to Work", width=18, command=self.end_work, state=tk.DISABLED)
        self.end_btn.pack(side=tk.LEFT, padx=6)

        self.summary_btn = tk.Button(top, text="Summarize", width=12, command=self.summarize_session)
        self.summary_btn.pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(top, textvariable=self.status_var, fg="#1a5fb4").pack(side=tk.RIGHT, padx=8)

        mid = tk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.log_area = scrolledtext.ScrolledText(mid, wrap=tk.WORD, height=18)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        ask_frame = tk.Frame(self.root)
        ask_frame.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(ask_frame, text="Ask about this session:").pack(anchor="w")
        self.ask_input = tk.Entry(ask_frame)
        self.ask_input.pack(fill=tk.X, pady=6)
        self.ask_input.bind("<Return>", lambda _: self.answer_question())
        tk.Button(ask_frame, text="Ask", command=self.answer_question).pack(anchor="e")

        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        tk.Label(bottom, text="Assistant output:").pack(anchor="w")
        self.answer_area = scrolledtext.ScrolledText(bottom, wrap=tk.WORD, height=12)
        self.answer_area.pack(fill=tk.BOTH, expand=True)

    def _init_openai(self):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            self._ui_log("OPENAI_API_KEY not found. Summarize/Q&A needs an API key.")
            return None
        try:
            return OpenAI(api_key=key)
        except Exception as exc:
            self._ui_log(f"OpenAI init failed: {exc}")
            return None

    def _load_config(self) -> None:
        cfg_path = APP_DIR / "config.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text())
            self.max_duration_seconds = int(cfg.get("max_duration_seconds", self.max_duration_seconds))
            self.audio_chunk_seconds = int(cfg.get("audio_chunk_seconds", self.audio_chunk_seconds))
            self.ocr_interval_seconds = int(cfg.get("ocr_interval_seconds", self.ocr_interval_seconds))
            self.sample_rate = int(cfg.get("sample_rate", self.sample_rate))
            self.input_device = cfg.get("input_device", self.input_device)
            self.min_capture_text_len = int(cfg.get("min_capture_text_len", self.min_capture_text_len))
            self._ui_log("Loaded config.json")
        except Exception as exc:
            self._ui_log(f"Config load error: {exc}")

    def start_work(self) -> None:
        if self.running:
            return

        self.running = True
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.session_path = SESSIONS_DIR / f"{self.session_id}.json"
        self.capture_entries = []
        self._save_session()

        self.start_btn.config(state=tk.DISABLED)
        self.end_btn.config(state=tk.NORMAL)
        self.status_var.set("Running")
        self._ui_log(f"Session started: {self.session_id}")

        threading.Thread(target=self._audio_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        threading.Thread(target=self._ocr_loop, daemon=True).start()
        threading.Thread(target=self._duration_guard_loop, daemon=True).start()

    def end_work(self) -> None:
        if not self.running:
            return
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.end_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self._ui_log("Session ended and saved.")
        self._save_session()

    def _duration_guard_loop(self) -> None:
        while self.running:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_duration_seconds:
                self.root.after(0, self._stop_due_to_timeout)
                return
            time.sleep(1)

    def _stop_due_to_timeout(self) -> None:
        if not self.running:
            return
        self.end_work()
        messagebox.showwarning(
            "Session ended",
            "This session reached the maximum duration, so it was ended and saved.",
        )

    def _audio_loop(self) -> None:
        while self.running:
            try:
                frames = int(self.sample_rate * self.audio_chunk_seconds)
                audio = sd.rec(
                    frames,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.input_device,
                )
                sd.wait()
                if self.running:
                    self.audio_queue.put(audio.copy())
            except Exception as exc:
                self._ui_log(f"Audio capture error: {exc}")
                time.sleep(2)

    def _transcribe_loop(self) -> None:
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                samples = np.squeeze(audio).astype(np.float32)
                segments, _ = self.whisper_model.transcribe(samples, language="en")
                text = " ".join(seg.text.strip() for seg in segments).strip()
                text = self._clean_text(text)
                if self._is_useful_text(text, self._last_audio_text):
                    self._last_audio_text = text
                    self._add_entry("audio", text)
            except Exception as exc:
                self._ui_log(f"Transcription error: {exc}")

    def _ocr_loop(self) -> None:
        # Homebrew installs tesseract to /opt/homebrew/bin on Apple Silicon.
        if os.path.exists("/opt/homebrew/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

        with mss.mss() as sct:
            while self.running:
                try:
                    monitor = sct.monitors[0]
                    shot = sct.grab(monitor)
                    image = Image.frombytes("RGB", shot.size, shot.rgb)
                    text = pytesseract.image_to_string(image).strip()
                    cleaned = self._clean_text(text)[:1000]
                    if self._is_useful_text(cleaned, self._last_screen_text):
                        self._last_screen_text = cleaned
                        self._add_entry("screen", cleaned)
                except Exception as exc:
                    self._ui_log(f"OCR error: {exc}")
                time.sleep(self.ocr_interval_seconds)

    def _clean_text(self, text: str) -> str:
        text = " ".join(text.split())
        text = re.sub(r"[^\x20-\x7E]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_useful_text(self, text: str, previous_text: str) -> bool:
        if not text:
            return False
        if len(text) < self.min_capture_text_len:
            return False
        if text == previous_text:
            return False
        # Filter out obvious OCR gibberish blocks with very low alphabetic content.
        alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)
        if alpha_ratio < 0.35:
            return False
        return True

    def _add_entry(self, source: str, text: str) -> None:
        entry = CaptureEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source=source,
            text=text,
        )
        self.capture_entries.append(entry)
        self._save_session()
        preview = text if len(text) < 140 else text[:140] + "..."
        self._ui_log(f"[{source}] {preview}")

    def _save_session(self) -> None:
        if not self.session_path:
            return
        payload = {
            "session_id": self.session_id,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat(timespec="seconds")
            if self.start_time
            else "",
            "running": self.running,
            "max_duration_seconds": self.max_duration_seconds,
            "entries": [asdict(x) for x in self.capture_entries],
        }
        self.session_path.write_text(json.dumps(payload, indent=2))

    def summarize_session(self) -> None:
        if not self.capture_entries:
            self._write_answer("No captured data yet.")
            return
        if not self.openai_client:
            self._write_answer("OPENAI_API_KEY missing. Cannot summarize right now.")
            return

        joined = self._joined_context(limit_chars=18000)
        prompt = (
            "Summarize this session in concise bullets for the user. "
            "Include key topics, action items, and unresolved questions.\n\n"
            f"{joined}"
        )
        self._run_llm(prompt, "Session Summary")

    def answer_question(self) -> None:
        question = self.ask_input.get().strip()
        if not question:
            return
        if not self.capture_entries:
            self._write_answer("No captured data yet.")
            return
        if not self.openai_client:
            self._write_answer("OPENAI_API_KEY missing. Cannot answer questions right now.")
            return

        context = self._joined_context(limit_chars=18000)
        prompt = (
            "Answer the user's question only using the captured session context. "
            "If context is insufficient, say what is missing.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        self._run_llm(prompt, "Answer")

    def _run_llm(self, prompt: str, title: str) -> None:
        def worker():
            try:
                response = self.openai_client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                )
                text = response.output_text.strip()
                self.root.after(0, lambda: self._write_answer(f"{title}\n\n{text}"))
            except Exception as exc:
                err_msg = f"LLM error: {exc}"
                self.root.after(0, lambda msg=err_msg: self._write_answer(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _joined_context(self, limit_chars: int = 12000) -> str:
        chunks = [f"[{e.timestamp}][{e.source}] {e.text}" for e in self.capture_entries]
        content = "\n".join(chunks)
        return content[-limit_chars:]

    def _ui_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "log_area"):
            self.log_area.insert(tk.END, f"[{stamp}] {text}\n")
            self.log_area.see(tk.END)
        else:
            print(f"[{stamp}] {text}")

    def _write_answer(self, text: str) -> None:
        self.answer_area.delete("1.0", tk.END)
        self.answer_area.insert(tk.END, text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ContextAssistant(root)

    def on_close():
        app.end_work()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
