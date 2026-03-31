"""
RAG (Retrieval-Augmented Generation) App using Tkinter + Ollama
Requirements: pip install ollama chromadb pypdf2 python-docx sentence-transformers
Ollama must be running locally: https://ollama.com
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import hashlib
import re

# ── Lazy imports (installed at runtime) ──────────────────────────────────────
def _lazy_import():
    global ollama, chromadb, SentenceTransformer, PyPDF2, docx
    try:
        import ollama
    except ImportError:
        ollama = None
    try:
        import chromadb
    except ImportError:
        chromadb = None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
    try:
        import PyPDF2
    except ImportError:
        PyPDF2 = None
    try:
        import docx
    except ImportError:
        docx = None

_lazy_import()

# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 50
TOP_K         = 4     # retrieved chunks per query
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_PATH   = "./rag_chroma_db"

# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    elif ext in (".docx", ".doc"):
        return _read_docx(path)
    elif ext in (".txt", ".md", ".csv", ".py", ".js", ".json", ".html", ".xml"):
        with open(path, "r", errors="replace") as f:
            return f.read()
    else:
        try:
            with open(path, "r", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Unsupported file type: {ext}") from e


def _read_pdf(path: str) -> str:
    if PyPDF2 is None:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _read_docx(path: str) -> str:
    if docx is None:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    import docx as _docx
    doc = _docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def chunk_text(text: str) -> list:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self._embedder   = None
        self._client     = None
        self._collection = None
        self._current_file = None

    def _ensure_embedder(self):
        if self._embedder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed.\nRun: pip install sentence-transformers")
            self._embedder = SentenceTransformer(EMBED_MODEL)

    def _ensure_chroma(self):
        if self._client is None:
            if chromadb is None:
                raise ImportError("chromadb not installed.\nRun: pip install chromadb")
            self._client = chromadb.PersistentClient(path=CHROMA_PATH)

    def _get_or_create_collection(self, name: str):
        self._ensure_chroma()
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60] or "rag_collection"
        self._collection = self._client.get_or_create_collection(safe)
        return self._collection

    def ingest_file(self, path: str, progress_cb=None) -> int:
        self._ensure_embedder()
        raw    = extract_text_from_file(path)
        chunks = chunk_text(raw)
        if not chunks:
            raise ValueError("No text could be extracted from the file.")

        col_name = hashlib.md5(path.encode()).hexdigest()[:16]
        col = self._get_or_create_collection(col_name)
        self._current_file = path

        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])

        batch_size = 32
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch  = chunks[i : i + batch_size]
            ids    = [f"chunk_{i+j}" for j in range(len(batch))]
            embeds = self._embedder.encode(batch).tolist()
            col.add(documents=batch, embeddings=embeds, ids=ids)
            if progress_cb:
                progress_cb(min(i + batch_size, total), total)

        return total

    def retrieve(self, query: str) -> list:
        if self._collection is None:
            raise RuntimeError("No document loaded. Please load a file first.")
        self._ensure_embedder()
        q_embed = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=q_embed,
            n_results=min(TOP_K, self._collection.count())
        )
        return results["documents"][0] if results["documents"] else []

    def query(self, question: str, model: str, stream_cb=None) -> str:
        if ollama is None:
            raise ImportError("ollama not installed.\nRun: pip install ollama")
        chunks  = self.retrieve(question)
        context = "\n\n---\n\n".join(chunks)
        prompt  = (
            f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
            f"If the answer is not in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        full_response = ""
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            if stream_cb:
                stream_cb(token)
        return full_response

    def list_local_models(self) -> tuple:
        """Returns (model_names, error_message). error_message is '' on success."""
        if ollama is None:
            return [], "ollama package not installed.\nRun: pip install ollama"
        try:
            resp = ollama.list()
            if hasattr(resp, "models"):
                models = resp.models
                names = []
                for m in models:
                    name = getattr(m, "model", None) or getattr(m, "name", None)
                    if name:
                        names.append(name)
                return names, ""
            if isinstance(resp, dict):
                return [m.get("name", m.get("model", "")) for m in resp.get("models", [])], ""
            return [], "Unexpected response format from ollama.list()"
        except Exception as e:
            err = str(e)
            if "Connection refused" in err or "ConnectError" in err or "connect" in err.lower():
                return [], "Ollama is not running.\nStart it from the system tray or run: ollama serve"
            return [], f"Ollama error: {err}"


# ── GUI ───────────────────────────────────────────────────────────────────────

class RAGApp(tk.Tk):
    DARK_BG  = "#1e1e2e"
    PANEL_BG = "#2a2a3e"
    ACCENT   = "#7c6af7"
    ACCENT2  = "#a78bfa"
    TEXT     = "#e2e2f0"
    SUBTEXT  = "#a0a0b8"
    SUCCESS  = "#4ade80"
    ERROR    = "#f87171"
    WARN     = "#fbbf24"
    ENTRY_BG = "#16162a"

    def __init__(self):
        super().__init__()
        self.title("RAG with Ollama")
        self.geometry("1000x720")
        self.minsize(820, 580)
        self.configure(bg=self.DARK_BG)

        self.engine = RAGEngine()
        self._ingest_thread = None
        self._query_thread  = None
        self._file_path     = None

        # ── FIX 1: Create tk variables AFTER super().__init__() (already correct)
        # but store the after-ID so we can cancel on close ──────────────────────
        self._retry_after_id = None
        self._alive          = True          # gate for all .after() callbacks

        self._loaded_file = tk.StringVar(value="No file loaded")
        self._model_var   = tk.StringVar()
        self._status_var  = tk.StringVar(value="Ready")
        self._chunk_size_var    = tk.IntVar(value=CHUNK_SIZE)
        self._chunk_overlap_var = tk.IntVar(value=CHUNK_OVERLAP)
        self._top_k_var         = tk.IntVar(value=TOP_K)

        self._ollama_ok = False
        self._build_ui()
        self._refresh_models()

        # ── FIX 2: Clean up properly on window close ──────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Cancel pending .after() callbacks before destroying the window."""
        self._alive = False
        if self._retry_after_id is not None:
            self.after_cancel(self._retry_after_id)
            self._retry_after_id = None
        self.destroy()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        top = tk.Frame(self, bg=self.DARK_BG, pady=10)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="🔍 RAG + Ollama", bg=self.DARK_BG,
                 fg=self.ACCENT2, font=("Segoe UI", 18, "bold")).pack(side="left")

        model_frame = tk.Frame(top, bg=self.DARK_BG)
        model_frame.pack(side="right")
        tk.Label(model_frame, text="Model:", bg=self.DARK_BG,
                 fg=self.SUBTEXT, font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))
        self._model_combo = ttk.Combobox(model_frame, textvariable=self._model_var,
                                          width=24, state="readonly")
        self._model_combo.pack(side="left")
        tk.Button(model_frame, text="⟳", command=self._refresh_models,
                  bg=self.PANEL_BG, fg=self.TEXT, relief="flat",
                  font=("Segoe UI", 11), cursor="hand2",
                  activebackground=self.ACCENT).pack(side="left", padx=4)

        paned = tk.PanedWindow(self, orient="horizontal", bg=self.DARK_BG,
                                sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        left = tk.Frame(paned, bg=self.PANEL_BG, bd=0)
        paned.add(left, minsize=260)
        self._build_left(left)

        right = tk.Frame(paned, bg=self.PANEL_BG, bd=0)
        paned.add(right, minsize=480)
        self._build_right(right)

        status_bar = tk.Frame(self, bg=self.ENTRY_BG, pady=4)
        status_bar.pack(fill="x", padx=16, pady=(0, 8))
        self._progress = ttk.Progressbar(status_bar, mode="determinate", length=120)
        self._progress.pack(side="left", padx=(8, 12))
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=self.ENTRY_BG, fg=self.SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left")

        self._style_ttk()

    def _build_left(self, parent):
        tk.Label(parent, text="📂  Document", bg=self.PANEL_BG,
                 fg=self.TEXT, font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(12, 4))

        file_row = tk.Frame(parent, bg=self.PANEL_BG)
        file_row.pack(fill="x", padx=12, pady=4)
        tk.Button(file_row, text="Browse…", command=self._browse_file,
                  bg=self.ACCENT, fg="white", relief="flat",
                  font=("Segoe UI", 10, "bold"), cursor="hand2",
                  activebackground=self.ACCENT2, padx=10).pack(side="left")

        tk.Label(parent, textvariable=self._loaded_file, bg=self.PANEL_BG,
                 fg=self.SUBTEXT, font=("Segoe UI", 8),
                 wraplength=220, justify="left").pack(fill="x", padx=12, pady=2)

        self._ingest_btn = tk.Button(parent, text="⚙  Index Document",
                                      command=self._start_ingest,
                                      bg=self.PANEL_BG, fg=self.ACCENT2,
                                      relief="flat", font=("Segoe UI", 10),
                                      cursor="hand2", padx=10, pady=4,
                                      activebackground=self.ENTRY_BG)
        self._ingest_btn.pack(fill="x", padx=12, pady=(4, 8))

        sep = tk.Frame(parent, bg=self.ACCENT, height=1)
        sep.pack(fill="x", padx=12, pady=4)

        tk.Label(parent, text="Chunk Settings", bg=self.PANEL_BG,
                 fg=self.SUBTEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 2))

        for label, var, lo, hi in [
            ("Chunk size", self._chunk_size_var,    100, 2000),
            ("Overlap",    self._chunk_overlap_var,   0,  200),
            ("Top-K",      self._top_k_var,            1,   10),
        ]:
            row = tk.Frame(parent, bg=self.PANEL_BG)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, bg=self.PANEL_BG,
                     fg=self.TEXT, font=("Segoe UI", 9), width=10,
                     anchor="w").pack(side="left")
            tk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=7,
                       bg=self.ENTRY_BG, fg=self.TEXT, relief="flat",
                       insertbackground=self.TEXT,
                       buttonbackground=self.PANEL_BG).pack(side="left", padx=4)

        sep2 = tk.Frame(parent, bg=self.ACCENT, height=1)
        sep2.pack(fill="x", padx=12, pady=(12, 4))

        tk.Label(parent, text="Chunk Preview", bg=self.PANEL_BG,
                 fg=self.SUBTEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12)
        self._chunk_preview = scrolledtext.ScrolledText(
            parent, bg=self.ENTRY_BG, fg=self.SUBTEXT,
            font=("Consolas", 8), relief="flat", wrap="word",
            state="disabled", height=8)
        self._chunk_preview.pack(fill="both", expand=True, padx=12, pady=8)

    def _build_right(self, parent):
        tk.Label(parent, text="💬  Chat", bg=self.PANEL_BG,
                 fg=self.TEXT, font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(12, 4))

        self._chat_area = scrolledtext.ScrolledText(
            parent, bg=self.ENTRY_BG, fg=self.TEXT,
            font=("Segoe UI", 10), relief="flat", wrap="word",
            state="disabled", spacing1=4, spacing3=4)
        self._chat_area.pack(fill="both", expand=True, padx=12, pady=4)

        self._chat_area.tag_config("user",      foreground=self.ACCENT2, font=("Segoe UI", 10, "bold"))
        self._chat_area.tag_config("assistant", foreground=self.SUCCESS,  font=("Segoe UI", 10, "bold"))
        self._chat_area.tag_config("error",     foreground=self.ERROR,    font=("Segoe UI", 10, "bold"))
        self._chat_area.tag_config("info",      foreground=self.WARN,     font=("Segoe UI", 9, "italic"))
        self._chat_area.tag_config("body",      foreground=self.TEXT,     font=("Segoe UI", 10))

        input_row = tk.Frame(parent, bg=self.PANEL_BG)
        input_row.pack(fill="x", padx=12, pady=(4, 12))

        self._query_entry = tk.Text(input_row, bg=self.ENTRY_BG, fg=self.TEXT,
                                     font=("Segoe UI", 10), relief="flat",
                                     insertbackground=self.TEXT, height=3, wrap="word")
        self._query_entry.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._query_entry.bind("<Return>",       self._on_enter)
        self._query_entry.bind("<Shift-Return>", lambda e: None)

        btn_col = tk.Frame(input_row, bg=self.PANEL_BG)
        btn_col.pack(side="right")
        self._send_btn = tk.Button(btn_col, text="Send\n▶", command=self._start_query,
                                    bg=self.ACCENT, fg="white", relief="flat",
                                    font=("Segoe UI", 10, "bold"), cursor="hand2",
                                    activebackground=self.ACCENT2, width=6, pady=6)
        self._send_btn.pack(pady=(0, 4))
        tk.Button(btn_col, text="Clear", command=self._clear_chat,
                  bg=self.PANEL_BG, fg=self.SUBTEXT, relief="flat",
                  font=("Segoe UI", 9), cursor="hand2",
                  activebackground=self.ENTRY_BG, width=6).pack()

    def _style_ttk(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground=self.ENTRY_BG,
                         background=self.PANEL_BG,
                         foreground=self.TEXT,
                         selectbackground=self.ACCENT,
                         selectforeground="white")
        style.configure("Horizontal.TProgressbar",
                         troughcolor=self.ENTRY_BG,
                         background=self.ACCENT)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _browse_file(self):
        filetypes = [
            ("Supported files", "*.txt *.pdf *.docx *.md *.csv *.py *.js *.json *.html *.xml"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select a document", filetypes=filetypes)
        if path:
            self._file_path = path
            self._loaded_file.set(os.path.basename(path))
            self._append_chat("info", f"📄 Loaded: {os.path.basename(path)}\nClick 'Index Document' to process it.\n")

    def _start_ingest(self):
        if not self._file_path:
            messagebox.showwarning("No file", "Please select a file first.")
            return
        if self._ingest_thread and self._ingest_thread.is_alive():
            return
        self._ingest_btn.config(state="disabled")
        self._set_status("Indexing…", busy=True)
        self._ingest_thread = threading.Thread(target=self._ingest_worker, daemon=True)
        self._ingest_thread.start()

    def _ingest_worker(self):
        global CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
        CHUNK_SIZE    = self._chunk_size_var.get()
        CHUNK_OVERLAP = self._chunk_overlap_var.get()
        TOP_K         = self._top_k_var.get()
        try:
            def progress(done, total):
                if not self._alive:
                    return
                pct = int(done / total * 100)
                self.after(0, lambda: self._progress.config(value=pct))
                self.after(0, lambda: self._set_status(f"Indexed {done}/{total} chunks…"))

            total = self.engine.ingest_file(self._file_path, progress_cb=progress)
            if self._alive:
                self.after(0, lambda: self._on_ingest_done(total))
        except Exception as e:
            msg = str(e)
            if self._alive:
                self.after(0, lambda: self._on_ingest_error(msg))

    def _on_ingest_done(self, total):
        if not self._alive:
            return
        self._set_status(f"Ready — {total} chunks indexed", busy=False)
        self._progress.config(value=100)
        self._ingest_btn.config(state="normal")
        self._append_chat("info", f"✅ Indexed {total} chunks. You can now ask questions!\n")
        self._show_chunk_preview()

    def _on_ingest_error(self, msg):
        if not self._alive:
            return
        self._set_status("Indexing failed", busy=False)
        self._ingest_btn.config(state="normal")
        self._append_chat("error", f"❌ Indexing error:\n{msg}\n")
        messagebox.showerror("Indexing Error", msg)

    def _show_chunk_preview(self):
        try:
            col = self.engine._collection
            if col is None:
                return
            sample  = col.get(limit=3)
            preview = ""
            for i, doc in enumerate(sample["documents"]):
                preview += f"── Chunk {i+1} ──\n{doc[:200]}…\n\n"
            self._chunk_preview.config(state="normal")
            self._chunk_preview.delete("1.0", "end")
            self._chunk_preview.insert("end", preview)
            self._chunk_preview.config(state="disabled")
        except Exception:
            pass

    def _on_enter(self, event):
        if not (event.state & 0x1):   # Shift not held
            self._start_query()
            return "break"

    def _start_query(self):
        question = self._query_entry.get("1.0", "end").strip()
        if not question:
            return
        if not self._file_path or self.engine._collection is None:
            messagebox.showwarning("No document", "Please load and index a document first.")
            return
        if not self._model_var.get():
            messagebox.showwarning("No model", "Please select an Ollama model.")
            return
        if self._query_thread and self._query_thread.is_alive():
            return

        self._query_entry.delete("1.0", "end")
        self._append_chat("user",      f"You: {question}\n")
        self._append_chat("assistant", "Assistant: ")
        self._send_btn.config(state="disabled")
        self._set_status("Generating…", busy=True)

        self._query_thread = threading.Thread(
            target=self._query_worker, args=(question,), daemon=True)
        self._query_thread.start()

    def _query_worker(self, question: str):
        try:
            def stream_cb(token):
                if self._alive:
                    self.after(0, lambda: self._append_stream(token))

            self.engine.query(question, self._model_var.get(), stream_cb=stream_cb)
            if self._alive:
                self.after(0, self._on_query_done)
        except Exception as e:
            msg = str(e)
            if self._alive:
                self.after(0, lambda: self._on_query_error(msg))

    def _on_query_done(self):
        if not self._alive:
            return
        self._append_stream("\n")
        self._send_btn.config(state="normal")
        self._set_status("Ready", busy=False)

    def _on_query_error(self, msg):
        if not self._alive:
            return
        self._append_chat("error", f"\n❌ Query error:\n{msg}\n")
        self._send_btn.config(state="normal")
        self._set_status("Error", busy=False)

    def _refresh_models(self):
        # ── FIX 3: Cancel any pending retry before scheduling a new one ────────
        if self._retry_after_id is not None:
            self.after_cancel(self._retry_after_id)
            self._retry_after_id = None

        models, err = self.engine.list_local_models()
        if models:
            self._model_combo["values"] = models
            if not self._model_var.get() or self._model_var.get() not in models:
                self._model_combo.current(0)
            self._set_status(f"{len(models)} model(s) found")
            self._ollama_ok = True
        else:
            self._model_combo["values"] = ["(no models found)"]
            self._model_combo.current(0)
            self._set_status(f"⚠ {err.splitlines()[0]}", busy=False)
            self._append_chat("error", f"⚠ Ollama: {err}\n")
            self._ollama_ok = False
            # Schedule retry, saving the ID so we can cancel on close
            if self._alive:
                self._retry_after_id = self.after(5000, self._retry_models)

    def _retry_models(self):
        """Silent background retry — only fires if previous attempt failed."""
        self._retry_after_id = None
        if not self._alive or self._ollama_ok:
            return
        models, err = self.engine.list_local_models()
        if models:
            self._model_combo["values"] = models
            self._model_combo.current(0)
            self._set_status(f"{len(models)} model(s) found")
            self._append_chat("info", f"✅ Ollama connected — {len(models)} model(s) available.\n")
            self._ollama_ok = True
        else:
            self._set_status(f"⚠ {err.splitlines()[0]}", busy=False)
            if self._alive:
                self._retry_after_id = self.after(5000, self._retry_models)

    def _append_chat(self, tag: str, text: str):
        if not self._alive:
            return
        self._chat_area.config(state="normal")
        self._chat_area.insert("end", text, tag)
        self._chat_area.see("end")
        self._chat_area.config(state="disabled")

    def _append_stream(self, token: str):
        if not self._alive:
            return
        self._chat_area.config(state="normal")
        self._chat_area.insert("end", token, "body")
        self._chat_area.see("end")
        self._chat_area.config(state="disabled")

    def _clear_chat(self):
        self._chat_area.config(state="normal")
        self._chat_area.delete("1.0", "end")
        self._chat_area.config(state="disabled")

    def _set_status(self, msg: str, busy: bool = False):
        if not self._alive:
            return
        self._status_var.set(msg)
        if busy:
            self._progress.config(mode="indeterminate")
            self._progress.start(12)
        else:
            self._progress.stop()
            self._progress.config(mode="determinate")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    DEPS = {
        "ollama":                "ollama",
        "chromadb":              "chromadb",
        "sentence_transformers": "sentence-transformers",
        "PyPDF2":                "PyPDF2",
        "docx":                  "python-docx",
    }
    missing = []
    for mod, pkg in DEPS.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--break-system-packages", *missing])
        _lazy_import()
        print("Dependencies installed. Starting app…\n")

    app = RAGApp()
    app.mainloop()