import re
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, scrolledtext

from core.capture_audio import AudioCapture
from core.capture_screen import ScreenCapture
from core.config import load_config
from core.context_builder import ContextBuilder
from core.embedding import HashingEmbeddingModel, create_embedding_model
from core.exporter import Exporter
from core.indexing_worker import IndexingWorker
from core.llm_client import LLMClient
from core.logger import get_logger, setup_logging
from core.retriever import HybridRetriever
from core.session_store import SessionStore
from core.sqlite_store import SQLiteStore
from core.summarizer import Summarizer
from core.ui_throttle import CaptureLogThrottler
from core.vector_store import VectorStore


APP_DIR = Path(__file__).parent
setup_logging(APP_DIR)
logger = get_logger(__name__)


class ActionButton(tk.Label):
    def __init__(
        self,
        parent,
        text: str,
        command,
        bg: str = "#0b1220",
        fg: str = "#f8fafc",
        hover_bg: str = "#2563eb",
        disabled_bg: str = "#1e293b",
        disabled_fg: str = "#94a3b8",
        state=tk.NORMAL,
        tooltip: str | None = None,
    ) -> None:
        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            padx=10,
            pady=7,
            font=("Helvetica", 11, "bold"),
            cursor="hand2",
            highlightthickness=1,
            highlightbackground="#334155",
            highlightcolor="#60a5fa",
        )
        self.command = command
        self._state = state
        self._normal_bg = bg
        self._normal_fg = fg
        self._hover_bg = hover_bg
        self._disabled_bg = disabled_bg
        self._disabled_fg = disabled_fg
        self._tooltip_text = tooltip or text
        self._tooltip_window = None
        self._tooltip_job = None
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self._apply_state()

    def config(self, cnf=None, **kwargs):
        options = {}
        if cnf:
            options.update(cnf)
        options.update(kwargs)
        if "state" in options:
            self._state = options.pop("state")
        if "bg" in options:
            self._normal_bg = options.pop("bg")
        if "fg" in options:
            self._normal_fg = options.pop("fg")
        tk.Label.config(self, **options)
        self._apply_state()

    configure = config

    def __getitem__(self, key):
        if key == "state":
            return self._state
        return super().__getitem__(key)

    def _apply_state(self) -> None:
        if self._state == tk.DISABLED:
            tk.Label.config(self, bg=self._disabled_bg, fg=self._disabled_fg, cursor="arrow")
        else:
            tk.Label.config(self, bg=self._normal_bg, fg=self._normal_fg, cursor="hand2")

    def _on_enter(self, _event) -> None:
        if self._state != tk.DISABLED:
            tk.Label.config(self, bg=self._hover_bg, fg="#ffffff")
        self._tooltip_job = self.after(350, self._show_tooltip)

    def _on_leave(self, _event) -> None:
        if self._tooltip_job:
            self.after_cancel(self._tooltip_job)
            self._tooltip_job = None
        self._hide_tooltip()
        self._apply_state()

    def _on_click(self, _event) -> None:
        if self._state != tk.DISABLED and self.command:
            self.command()

    def _show_tooltip(self) -> None:
        if self._tooltip_window or not self._tooltip_text:
            return
        x = self.winfo_rootx() + self.winfo_width() // 2
        y = self.winfo_rooty() - 36
        window = tk.Toplevel(self)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            window,
            text=self._tooltip_text,
            bg="#27272a",
            fg="#ffffff",
            padx=12,
            pady=7,
            font=("Helvetica", 12),
        )
        label.pack()
        self._tooltip_window = window

    def _hide_tooltip(self) -> None:
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None


class ContextAssistant:
    def __init__(self, root: tk.Tk) -> None:
        startup_start = time.perf_counter()
        self.root = root
        self.root.title("Context Assistant")
        self.root.geometry("980x720")
        logger.info("App start")

        self.running = False
        self.paused = False
        self.start_time = 0.0
        self.last_summary = ""
        self.last_answer = ""
        self.busy = False
        self.busy_message = ""
        self.busy_pulse_on = False
        self.answer_mode = "detailed"
        self.last_json_save_time = 0.0
        self.capture_log_throttler = CaptureLogThrottler(interval_seconds=1.0)
        self.capture_log_flush_scheduled = False
        self.audio_capture: AudioCapture | None = None
        self.screen_capture: ScreenCapture | None = None
        self.session_lock = threading.Lock()

        self._build_ui()
        self.config, config_messages = load_config(APP_DIR)
        self.excluded_apps_var.set(", ".join(self.config.excluded_apps))
        for message in config_messages:
            self._ui_log(message)

        self.session_store = SessionStore(APP_DIR)
        self.sqlite_store = SQLiteStore(APP_DIR)
        self.exporter = Exporter(APP_DIR)
        self.embedding_model = self._init_embedding_model()
        self.retriever = HybridRetriever(
            self.sqlite_store,
            self.embedding_model,
            vector_store=VectorStore(index_dir=self.sqlite_store.data_dir / "vector_index"),
        )
        self.indexing_worker = IndexingWorker(self.sqlite_store, self.retriever)
        self.indexing_worker.start()
        self.context_builder = ContextBuilder(token_budget=900)
        self.llm_client = self._init_llm_client()
        self.summarizer = Summarizer(self.llm_client) if self.llm_client else None
        self._ui_log("Ready. Configure permissions and click Start.")
        logger.info("App startup completed elapsed_ms=%.1f", (time.perf_counter() - startup_start) * 1000)

    def _build_ui(self) -> None:
        app_bg = "#0f172a"
        panel_bg = "#111827"
        surface_bg = "#020617"
        text_fg = "#f9fafb"
        muted_fg = "#cbd5e1"

        self.root.configure(bg=app_bg)
        self.root.geometry("1180x820")

        shell = tk.Frame(self.root, bg=app_bg)
        shell.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(shell, bg=surface_bg, width=280, highlightbackground="#1f2937", highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        main = tk.Frame(shell, bg=app_bg)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            sidebar,
            text="Mindtrace",
            bg=surface_bg,
            fg=text_fg,
            font=("Helvetica", 24, "bold"),
        ).pack(anchor="w", padx=20, pady=(22, 4))
        tk.Label(
            sidebar,
            text="AI memory assistant",
            bg=surface_bg,
            fg=muted_fg,
            font=("Helvetica", 12),
        ).pack(anchor="w", padx=20, pady=(0, 16))

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(
            sidebar,
            textvariable=self.status_var,
            bg="#111827",
            fg=text_fg,
            font=("Helvetica", 11, "bold"),
            padx=12,
            pady=6,
        )
        self.status_label.pack(anchor="w", padx=20, pady=(0, 18))

        controls = tk.Frame(sidebar, bg=surface_bg)
        controls.pack(fill=tk.X, padx=18)
        self.start_btn = self._button(controls, "Start capture", self.start_work, fill=tk.X)
        self.end_btn = self._button(controls, "Stop capture", self.end_work, state=tk.DISABLED, fill=tk.X)
        self.pause_btn = self._button(controls, "Pause", self.toggle_pause, state=tk.DISABLED, fill=tk.X)
        self.summary_btn = self._button(controls, "Summarize", self.summarize_session, fill=tk.X)
        self.delete_btn = self._button(controls, "Delete session", self.delete_session, fill=tk.X)
        self.export_md_btn = self._button(controls, "Export Markdown", self.export_summary_markdown, fill=tk.X)
        self.export_pdf_btn = self._button(controls, "Export PDF", self.export_summary_pdf, fill=tk.X)

        privacy = tk.Frame(sidebar, bg=surface_bg)
        privacy.pack(fill=tk.X, padx=18, pady=(16, 0))
        tk.Label(privacy, text="Excluded apps", bg=surface_bg, fg=muted_fg, font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.excluded_apps_var = tk.StringVar(value=", ".join(self.config.excluded_apps) if hasattr(self, "config") else "")
        self.excluded_apps_entry = tk.Entry(
            privacy,
            textvariable=self.excluded_apps_var,
            relief=tk.FLAT,
            bg=panel_bg,
            fg=text_fg,
            insertbackground=text_fg,
            insertontime=650,
            insertofftime=350,
            highlightthickness=1,
            highlightbackground="#334155",
            highlightcolor="#ffffff",
            font=("Helvetica", 11),
        )
        self.excluded_apps_entry.pack(fill=tk.X, pady=(8, 8), ipady=7)
        self.excluded_apps_entry.bind("<FocusIn>", lambda event: event.widget.config(highlightbackground="#ffffff"))
        self.excluded_apps_entry.bind("<FocusOut>", lambda event: event.widget.config(highlightbackground="#334155"))
        self._button(privacy, "Apply privacy", self.apply_privacy_settings, fill=tk.X)

        search = tk.Frame(sidebar, bg=surface_bg)
        search.pack(fill=tk.X, padx=18, pady=(16, 0))
        tk.Label(search, text="Search memory", bg=surface_bg, fg=muted_fg, font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.search_input = tk.Entry(
            search,
            relief=tk.FLAT,
            bg=panel_bg,
            fg=text_fg,
            insertbackground=text_fg,
            insertontime=650,
            insertofftime=350,
            highlightthickness=1,
            highlightbackground="#334155",
            highlightcolor="#ffffff",
            font=("Helvetica", 12),
        )
        self.search_input.pack(fill=tk.X, pady=(8, 8), ipady=8)
        self.search_input.bind("<Return>", lambda _: self.search_sessions())
        self.search_input.bind("<FocusIn>", lambda event: event.widget.config(highlightbackground="#ffffff"))
        self.search_input.bind("<FocusOut>", lambda event: event.widget.config(highlightbackground="#334155"))
        self.search_btn = self._button(search, "Search", self.search_sessions, fill=tk.X)
        self.import_btn = self._button(search, "Import old sessions", self.import_existing_sessions, fill=tk.X)

        event_panel = tk.Frame(sidebar, bg=surface_bg)
        event_panel.pack(fill=tk.BOTH, expand=True, padx=18, pady=(16, 18))
        tk.Label(event_panel, text="Activity", bg=surface_bg, fg=muted_fg, font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(event_panel, wrap=tk.WORD, height=8, relief=tk.FLAT, bg="#0b1120", fg="#e5e7eb", font=("Helvetica", 11))
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        chat_header = tk.Frame(main, bg=app_bg)
        chat_header.pack(fill=tk.X, padx=34, pady=(24, 8))
        tk.Label(chat_header, text="Ask Across Memory", bg=app_bg, fg=text_fg, font=("Helvetica", 24, "bold")).pack(anchor="w")
        tk.Label(chat_header, text="Ask about this session or anything Mindtrace has indexed.", bg=app_bg, fg=muted_fg, font=("Helvetica", 13)).pack(anchor="w", pady=(4, 0))

        self.chat_area = scrolledtext.ScrolledText(main, wrap=tk.WORD, relief=tk.FLAT, bg=app_bg, fg=text_fg, insertbackground=text_fg, font=("Helvetica", 16), padx=18, pady=18)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 12))
        self.chat_area.tag_configure("user", foreground=text_fg, spacing3=8, lmargin1=180, lmargin2=180, rmargin=18)
        self.chat_area.tag_configure("assistant", foreground=text_fg, spacing3=14, lmargin1=18, lmargin2=18, rmargin=160)
        self.chat_area.tag_configure("system", foreground="#eab308", spacing3=10, lmargin1=18, lmargin2=18)
        self.chat_area.insert(tk.END, "Mindtrace\nAsk a question below, or start capture from the sidebar.\n", "assistant")
        self.chat_area.config(state=tk.DISABLED)

        self.chat_status_var = tk.StringVar(value="")
        self.chat_status_label = tk.Label(
            main,
            textvariable=self.chat_status_var,
            bg=app_bg,
            fg=text_fg,
            font=("Helvetica", 11, "bold"),
            padx=10,
            pady=4,
        )
        self.chat_status_label.pack(anchor="w", padx=52, pady=(0, 6))

        composer = tk.Frame(main, bg=surface_bg, highlightbackground="#334155", highlightthickness=1)
        composer.pack(fill=tk.X, padx=34, pady=(0, 24))
        self.ask_input = tk.Entry(
            composer,
            relief=tk.FLAT,
            bg=surface_bg,
            fg=text_fg,
            insertbackground=text_fg,
            insertontime=650,
            insertofftime=350,
            font=("Helvetica", 15),
        )
        self.ask_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=16, pady=14, ipady=6)
        self.ask_input.bind("<Return>", lambda _: self.answer_question())
        self.ask_input.bind("<FocusIn>", lambda _: composer.config(highlightbackground="#ffffff"))
        self.ask_input.bind("<FocusOut>", lambda _: composer.config(highlightbackground="#334155"))
        self.ask_btn = ActionButton(composer, "Ask", self.answer_question, tooltip="Send")
        self.ask_btn.pack(side=tk.RIGHT, padx=(8, 12))
        self.detailed_mode_btn = ActionButton(
            composer,
            "Detailed",
            lambda: None,
            bg="#2563eb",
            hover_bg="#3b82f6",
            tooltip="Detailed answer",
        )
        self.detailed_mode_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self.answer_area = self.chat_area

    def _panel(self, parent, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg="white", highlightbackground="#e5e7eb", highlightthickness=1)
        tk.Label(frame, text=title, bg="white", fg="#111827", font=("Helvetica", 13, "bold")).pack(
            anchor="w",
            padx=12,
            pady=10,
        )
        return frame

    def _button(self, parent, text: str, command, bg: str = "#0b1220", fg: str = "#f8fafc", state=tk.NORMAL, fill=None) -> ActionButton:
        button = ActionButton(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            state=state,
            tooltip=text,
        )
        if fill:
            button.pack(fill=fill, pady=(0, 6))
        else:
            button.pack(side=tk.LEFT, padx=(0, 8))
        return button

    def _init_llm_client(self) -> LLMClient | None:
        try:
            client = LLMClient(self.config.llm)
            if not client.is_available:
                self._ui_log(f"{self.config.llm.api_key_env} not found. Summarize/Q&A needs an API key.")
                return None
            self._ui_log(f"Loaded LLM provider: {self.config.llm.provider} ({self.config.llm.model})")
            return client
        except Exception as exc:
            logger.exception("LLM init failed")
            self._ui_log(f"LLM init failed: {exc}")
            return None

    def _init_embedding_model(self):
        try:
            return create_embedding_model()
        except Exception as exc:
            logger.exception("Embedding model init failed; falling back to local hashing embeddings")
            self._ui_log(f"Embedding init failed: {exc}. Using local hashing embeddings.")
            return HashingEmbeddingModel()

    def start_work(self) -> None:
        if self.running:
            return

        self.running = True
        self.paused = False
        self.start_time = time.time()
        session_id = self.session_store.start(self.start_time)
        self._save_session(force=True)
        logger.info("Capture start session_id=%s", session_id)

        self.start_btn.config(state=tk.DISABLED)
        self.end_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self._update_recording_timer()
        self._ui_log(f"Session started: {session_id}")

        self.audio_capture = AudioCapture(
            self.config,
            on_text=self._add_entry,
            on_error=self._thread_log,
            is_running=lambda: self.running,
            is_paused=lambda: self.paused,
        )
        self.screen_capture = ScreenCapture(
            self.config,
            on_text=self._add_entry,
            on_error=self._thread_log,
            is_running=lambda: self.running,
            is_paused=lambda: self.paused,
        )
        self.audio_capture.start()
        self.screen_capture.start()
        self.root.update_idletasks()
        threading.Thread(target=self._duration_guard_loop, daemon=True).start()

    def end_work(self) -> None:
        if not self.running:
            return
        self.running = False
        self.paused = False
        logger.info("Capture stop session_id=%s", self.session_store.session_id)
        self.start_btn.config(state=tk.NORMAL)
        self.end_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.status_var.set("Stopped")
        self._ui_log("Session ended and saved.")
        self._save_session(force=True)

    def _duration_guard_loop(self) -> None:
        while self.running:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.max_duration_seconds:
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

    def toggle_pause(self) -> None:
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        self.status_var.set(self._recording_status_text())
        self._ui_log("Capture paused." if self.paused else "Capture resumed.")

    def _update_recording_timer(self) -> None:
        if not self.running:
            return
        self.status_var.set(self._recording_status_text())
        self.root.after(1000, self._update_recording_timer)

    def _recording_status_text(self) -> str:
        elapsed = max(0, int(time.time() - self.start_time)) if self.start_time else 0
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        label = "Paused" if self.paused else "Recording"
        return f"{label} {hours:02d}:{minutes:02d}:{seconds:02d}"

    def apply_privacy_settings(self) -> None:
        apps = [app.strip() for app in self.excluded_apps_var.get().split(",") if app.strip()]
        self.config.excluded_apps = apps
        self._ui_log(f"Excluded apps updated: {', '.join(apps) if apps else 'none'}")

    def delete_session(self) -> None:
        session_id = self.session_store.session_id
        if not session_id:
            self._write_answer("No active session to delete.")
            return
        if self.running:
            self.end_work()

        self._set_busy("Deleting session...")

        def worker():
            try:
                self.sqlite_store.delete_session(session_id)
                with self.session_lock:
                    self.session_store.delete_current()
                self.retriever.mark_vector_index_dirty(rebuild=True)
                self.last_summary = ""
                self.last_answer = ""
                self.root.after(0, lambda: self._ui_log(f"Deleted session: {session_id}"))
                self.root.after(0, lambda: self._write_answer("Deleted the current session."))
            except Exception as exc:
                logger.exception("Delete session failed")
                self.root.after(0, lambda msg=f"Delete session failed: {exc}": self._write_answer(msg))
            finally:
                self.root.after(0, self._clear_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _add_entry(self, source: str, text: str) -> None:
        with self.session_lock:
            entry = self.session_store.add_entry(source, text)
            self._save_session(force=False)
            self._save_capture_to_sqlite(entry)
        preview = entry.text if len(entry.text) < 140 else entry.text[:140] + "..."
        self._queue_capture_log(entry.source, preview)

    def _save_session(self, force: bool = False) -> None:
        try:
            now = time.time()
            should_write_json = force or not self.running or (now - self.last_json_save_time) >= 30
            if should_write_json:
                self.session_store.save(
                    running=self.running,
                    max_duration_seconds=self.config.max_duration_seconds,
                    app_version=self.config.app_version,
                    llm_settings=self.config.llm,
                )
                self.last_json_save_time = now
            if force:
                self.sqlite_store.save_session(
                    session_id=self.session_store.session_id,
                    started_at=self.session_store.started_at,
                    ended_at=self.session_store.ended_at if not self.running else "",
                    duration_seconds=self.session_store.duration_seconds(self.running),
                    app_version=self.config.app_version,
                    llm_settings=self.config.llm,
                )
        except Exception as exc:
            logger.exception("Session save error")
            self._ui_log(f"Session save failed: {exc}")

    def _save_capture_to_sqlite(self, entry) -> None:
        try:
            start = time.perf_counter()
            capture_id = self.sqlite_store.save_capture(self.session_store.session_id, entry)
            sqlite_elapsed_ms = (time.perf_counter() - start) * 1000
            enqueue_start = time.perf_counter()
            self.sqlite_store.enqueue_capture(capture_id, self.session_store.session_id)
            enqueue_elapsed_ms = (time.perf_counter() - enqueue_start) * 1000
            pending = self.sqlite_store.queue_counts().get("pending", 0)
            logger.info(
                "Capture persisted source=%s sqlite_ms=%.1f enqueue_ms=%.1f pending=%s",
                entry.source,
                sqlite_elapsed_ms,
                enqueue_elapsed_ms,
                pending,
            )
        except Exception as exc:
            logger.exception("SQLite capture save error")
            self._thread_log(f"Search index save failed: {exc}")

    def summarize_session(self) -> None:
        with self.session_lock:
            has_entries = bool(self.session_store.entries)
            context = self.session_store.joined_context(limit_chars=self.config.context_limit_chars)
        if not has_entries:
            self._write_answer("No captured data yet.")
            return
        if not self.summarizer:
            self._write_answer(f"{self.config.llm.api_key_env} missing. Cannot summarize right now.")
            return

        self._run_llm(lambda: self.summarizer.summarize(context), "Session Summary", save_summary=True)

    def answer_question(self) -> None:
        question = self.ask_input.get().strip()
        if not question:
            return
        if not self.summarizer:
            self._write_answer(f"{self.config.llm.api_key_env} missing. Cannot answer questions right now.")
            return
        self.ask_input.delete(0, tk.END)
        self._append_chat("user", question)
        response_mode = self.answer_mode

        def task():
            context = self._rag_context(question)
            if not context:
                return "No captured or indexed context found for that question."
            return self.summarizer.answer(question, context, response_mode=response_mode)

        self._run_llm(task, "Answer", store_answer=True, loading_text="Searching memory and preparing an answer...")

    def _rag_context(self, question: str) -> str:
        parts: list[str] = []
        with self.session_lock:
            current_limit = min(4000, max(1000, self.config.context_limit_chars // 3))
            current = self.session_store.joined_context(limit_chars=current_limit)
        if current:
            parts.append("Current session context:\n" + current)
        try:
            candidates = self.retriever.retrieve(question, top_k=8)
            self._thread_log(self.retriever.last_retrieval_status)
            retrieval_context = self.context_builder.build(candidates)
        except Exception:
            logger.exception("Hybrid retrieval failed")
            retrieval_context = ""
        if retrieval_context:
            parts.append("Relevant retrieved memory:\n" + retrieval_context)
        return "\n\n".join(parts)

    def search_sessions(self) -> None:
        query = self.search_input.get().strip()
        if not query:
            self._write_answer("Enter a search query.")
            return
        self._append_chat("user", f"Search memory: {query}")
        self._set_busy("Searching memory...")

        def worker():
            try:
                results = self.sqlite_store.search_captures(query)
                if not results:
                    text = "No matching captures found."
                else:
                    lines = ["Search results", ""]
                    for result in results:
                        rank = f" relevance {abs(result.rank):.4f}" if result.rank is not None else ""
                        lines.append(f"{result.source}{rank}")
                        lines.append(result.snippet)
                        lines.append("")
                    text = "\n".join(lines).strip()
                self.root.after(0, lambda value=text: self._write_answer(value))
            except Exception as exc:
                logger.exception("Search failed")
                self.root.after(0, lambda msg=f"Search failed: {exc}": self._write_answer(msg))
            finally:
                self.root.after(0, self._clear_busy)

        threading.Thread(target=worker, daemon=True).start()

    def import_existing_sessions(self) -> None:
        self._set_busy("Importing existing sessions...")

        def worker():
            try:
                result = self.sqlite_store.import_json_sessions(APP_DIR / "sessions")
                text = (
                    "Import existing sessions\n\n"
                    f"Imported sessions: {result.sessions}\n"
                    f"Imported captures: {result.captures}\n"
                    f"Imported summaries: {result.summaries}\n"
                    f"Skipped existing sessions: {result.skipped_sessions}"
                )
                self.root.after(0, lambda value=text: self._write_answer(value))
            except Exception as exc:
                logger.exception("Import existing sessions failed")
                self.root.after(0, lambda msg=f"Import failed: {exc}": self._write_answer(msg))
            finally:
                self.root.after(0, self._clear_busy)

        threading.Thread(target=worker, daemon=True).start()

    def export_summary_markdown(self) -> None:
        self._export_summary("markdown")

    def export_summary_pdf(self) -> None:
        self._export_summary("pdf")

    def _export_summary(self, file_type: str) -> None:
        if not self.last_summary:
            self._write_answer("No summary to export yet. Click Summarize first.")
            return
        session_id = self.session_store.session_id or "session"
        try:
            if file_type == "markdown":
                path = self.exporter.export_markdown(session_id, "Session Summary", self.last_summary)
            else:
                path = self.exporter.export_pdf(session_id, "Session Summary", self.last_summary)
            self._write_answer(f"Exported {file_type.upper()}:\n{path}")
        except Exception as exc:
            logger.exception("Export failed")
            self._write_answer(f"Export failed: {exc}")

    def _run_llm(
        self,
        task,
        title: str,
        save_summary: bool = False,
        store_answer: bool = False,
        loading_text: str | None = None,
    ) -> None:
        self._set_busy(loading_text or f"{title} in progress...")
        self._append_chat("system", loading_text or "Mindtrace is thinking...")

        def worker():
            try:
                text = task()
                if save_summary:
                    self.last_summary = text
                if store_answer:
                    self.last_answer = text
                if save_summary:
                    try:
                        self.sqlite_store.save_summary(self.session_store.session_id, text)
                    except Exception:
                        logger.exception("SQLite summary save error")
                self.root.after(0, lambda: self._write_answer(f"{title}\n\n{text}"))
            except Exception as exc:
                logger.exception("LLM task failed")
                err_msg = f"LLM request failed: {exc}"
                self.root.after(0, lambda msg=err_msg: self._write_answer(msg))
            finally:
                self.root.after(0, self._clear_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, text: str) -> None:
        self.busy = True
        self.busy_message = text
        self.busy_pulse_on = True
        self.chat_status_var.set(f"● {text}")
        self.status_var.set(text)
        self.status_label.config(bg="#ffffff", fg="#020617")
        self.chat_status_label.config(bg="#020617", fg="#ffffff")
        for button in (self.ask_btn, self.detailed_mode_btn, self.summary_btn, self.search_btn, self.import_btn, self.delete_btn):
            button.config(state=tk.DISABLED)
        self.root.update_idletasks()
        self._pulse_busy()

    def _clear_busy(self) -> None:
        self.busy = False
        self.busy_message = ""
        self.chat_status_var.set("")
        self.status_var.set(self._recording_status_text() if self.running else "Stopped")
        self.status_label.config(bg="#111827", fg="#f9fafb")
        self.chat_status_label.config(bg="#0f172a", fg="#f9fafb")
        self.ask_btn.config(state=tk.NORMAL)
        self.detailed_mode_btn.config(state=tk.NORMAL)
        self.summary_btn.config(state=tk.NORMAL)
        self.search_btn.config(state=tk.NORMAL)
        self.import_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)

    def _pulse_busy(self) -> None:
        if not self.busy or not self.busy_message:
            return
        self.busy_pulse_on = not self.busy_pulse_on
        marker = "●" if self.busy_pulse_on else "○"
        self.chat_status_var.set(f"{marker} {self.busy_message}")
        self.root.after(450, self._pulse_busy)

    def _ui_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        if hasattr(self, "log_area"):
            self.log_area.insert(tk.END, f"[{stamp}] {text}\n")
            self.log_area.see(tk.END)
        else:
            print(f"[{stamp}] {text}")

    def _thread_log(self, text: str) -> None:
        self.root.after(0, lambda: self._ui_log(text))

    def _queue_capture_log(self, source: str, preview: str) -> None:
        self.capture_log_throttler.add(source, preview)
        if not self.capture_log_flush_scheduled:
            self.capture_log_flush_scheduled = True
            self.root.after(1000, self._flush_capture_log)

    def _flush_capture_log(self) -> None:
        self.capture_log_flush_scheduled = False
        if not self.capture_log_throttler.should_flush():
            return
        message = self.capture_log_throttler.flush()
        if message:
            self._ui_log(message)

    def _write_answer(self, text: str) -> None:
        self._append_chat("assistant", self._clean_user_facing_output(text))

    def _append_chat(self, role: str, text: str) -> None:
        if not hasattr(self, "chat_area"):
            return
        label = {"user": "You", "assistant": "Mindtrace", "system": "Status"}.get(role, "Mindtrace")
        tag = role if role in {"user", "assistant", "system"} else "assistant"
        self.chat_area.config(state=tk.NORMAL)
        if self.chat_area.index("end-1c") != "1.0":
            self.chat_area.insert(tk.END, "\n\n")
        self.chat_area.insert(tk.END, f"{label}\n", tag)
        self.chat_area.insert(tk.END, text.strip(), tag)
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def _clean_user_facing_output(self, text: str) -> str:
        text = re.sub(r"^\[[0-2]\d:[0-5]\d:[0-5]\d\]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"`?\[[0-2]\d:[0-5]\d:[0-5]\d\]`?", "the captured moment", text)
        text = re.sub(r"Session started:\s*\S+", "Session started", text)
        text = re.sub(r"session_id\s*=\s*\S+", "session", text, flags=re.IGNORECASE)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = text.replace("*", "")
        return text.strip()


if __name__ == "__main__":
    root = tk.Tk()
    app = ContextAssistant(root)

    def on_close():
        app.end_work()
        app.indexing_worker.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
