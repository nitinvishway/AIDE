"""
semantic_index.py

High-performance SQLite semantic and full-text index with incremental sync and
real-time filesystem change tracking (via watchdog).
"""

import os
import time
import json
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from .config import DB_PATH, MAX_FILE_SIZE_BYTES
from .embeddings import get_embedder, TextChunker, TextChunk


class SemanticFileIndex:
    """
    Manages the hybrid SQLite database storing:
    1. File metadata (path, size, mtime, sha256 hash)
    2. Document text chunks with L2-normalized vector embeddings (BLOB)
    3. SQLite FTS5 full-text search index for BM25 keyword matching
    4. Tags and access history (for personalized retrieval)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.chunker = TextChunker()
        self.embedder = get_embedder()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Files table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
            """)

            # Chunks table with embedding vector BLOB
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                )
            """)

            # Check if chunks_fts needs schema update for file_name column
            try:
                conn.execute("SELECT file_name FROM chunks_fts LIMIT 0")
            except sqlite3.OperationalError:
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.execute("""
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        file_id UNINDEXED,
                        file_name,
                        text,
                        tokenize='porter unicode61'
                    )
                """)

            # Tags and categorization
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(file_id, tag),
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                )
            """)

            # Access logs for personalization (recency & user affinity)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                )
            """)

            # Indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON file_chunks(file_id)")
            conn.commit()

    def _detect_category(self, ext: str) -> str:
        ext = ext.lower().lstrip(".")
        if ext in ["pdf", "doc", "docx", "txt", "rtf", "odt"]:
            return "document"
        elif ext in ["py", "rs", "ts", "tsx", "js", "jsx", "html", "css", "json", "toml", "yaml", "yml", "sql"]:
            return "code"
        elif ext in ["png", "jpg", "jpeg", "webp", "gif", "bmp", "svg"]:
            return "image"
        elif ext in ["csv", "xlsx", "xls", "tsv"]:
            return "spreadsheet"
        elif ext in ["mp3", "wav", "flac", "m4a", "ogg"]:
            return "audio"
        elif ext in ["mp4", "mkv", "mov", "avi"]:
            return "video"
        return "other"

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from text, code, markdown, PDF, or multimodal image files."""
        ext = file_path.suffix.lower().lstrip(".")
        if ext in ["py", "rs", "ts", "tsx", "js", "jsx", "html", "css", "json", "toml", "yaml", "yml", "sql", "txt", "md", "csv", "log"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text_parts = []
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        text_parts.append(txt)
                return "\n".join(text_parts)
            except Exception:
                return ""
        elif ext in ["png", "jpg", "jpeg", "webp", "gif", "bmp"]:
            try:
                if not hasattr(self, "_multimodal_extractor") or self._multimodal_extractor is None:
                    from file_intel.multimodal_extractor import MultimodalExtractor
                    self._multimodal_extractor = MultimodalExtractor()
                return self._multimodal_extractor.extract(file_path)
            except Exception as e:
                logger.warning(f"Multimodal extraction fallback for {file_path}: {e}")
                return ""
        return ""

    def index_file(self, file_path: str, content: Optional[str] = None, force: bool = False) -> Optional[str]:
        """
        Incrementally index a single file. If unchanged, skip.
        """
        p = Path(file_path).resolve()
        if not p.exists():
            return None

        stat = p.stat()
        mtime = stat.st_mtime
        size_bytes = stat.st_size

        if size_bytes > MAX_FILE_SIZE_BYTES:
            return None

        # Check existing record for incremental sync
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT file_id, mtime, sha256 FROM files WHERE path = ?", (str(p),)
            ).fetchone()

        # If file text wasn't provided, read it
        raw_text = content if content is not None else self._extract_text(p)
        if not raw_text.strip():
            raw_text = f"File: {p.name}\nPath: {str(p)}\nCategory: {self._detect_category(p.suffix)}"

        sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # Incremental skip if unmodified
        if existing and not force:
            if existing["mtime"] == mtime and existing["sha256"] == sha256:
                return existing["file_id"]

        # Chunk and embed
        chunks = self.chunker.chunk_text(raw_text)
        if not chunks:
            return None

        file_id = hashlib.md5(str(p).encode("utf-8")).hexdigest()
        category = self._detect_category(p.suffix)
        now_iso = datetime.now().isoformat()

        with self._get_connection() as conn:
            # Upsert file metadata
            conn.execute("""
                INSERT INTO files (file_id, path, name, category, extension, size_bytes, mtime, sha256, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes = excluded.size_bytes,
                    mtime = excluded.mtime,
                    sha256 = excluded.sha256,
                    indexed_at = excluded.indexed_at
            """, (file_id, str(p), p.name, category, p.suffix.lower(), size_bytes, mtime, sha256, now_iso))

            # Delete old chunks
            conn.execute("DELETE FROM file_chunks WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM chunks_fts WHERE file_id = ?", (file_id,))

            # Insert chunks with embeddings and FTS
            for c in chunks:
                chunk_id = f"{file_id}_{c.chunk_index}"
                emb = self.embedder.embed_text(c.text)
                emb_blob = emb.astype(np.float32).tobytes()

                conn.execute("""
                    INSERT INTO file_chunks (chunk_id, file_id, chunk_index, start_char, end_char, start_line, end_line, text, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, file_id, c.chunk_index, c.start_char, c.end_char, c.start_line, c.end_line, c.text, emb_blob))

                conn.execute("""
                    INSERT INTO chunks_fts (chunk_id, file_id, file_name, text)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, file_id, p.name, c.text))

            conn.commit()

        return file_id

    def delete_file(self, file_path: str):
        """Remove file from index."""
        p = str(Path(file_path).resolve())
        with self._get_connection() as conn:
            row = conn.execute("SELECT file_id FROM files WHERE path = ?", (p,)).fetchone()
            if row:
                file_id = row["file_id"]
                conn.execute("DELETE FROM file_chunks WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM chunks_fts WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM file_tags WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM access_logs WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
                conn.commit()

    def index_directory(self, dir_path: str, recursive: bool = True, extensions: Optional[List[str]] = None, force: bool = False) -> int:
        """Scan and index all matching files in a directory."""
        root = Path(dir_path).resolve()
        if not root.is_dir():
            return 0

        target_exts = set(e.lower().lstrip(".") for e in extensions) if extensions else {
            "txt", "md", "py", "rs", "ts", "tsx", "js", "jsx", "json", "html", "css", "pdf", "toml", "yaml",
            "png", "jpg", "jpeg", "webp", "gif", "bmp"
        }

        # Exclude directories
        exclude_dirs = {".git", "node_modules", "target", ".venv", "__pycache__", "dist"}

        count = 0
        pattern = "**/*" if recursive else "*"
        for path in root.glob(pattern):
            if path.is_file():
                # Check path excludes
                if any(part in exclude_dirs for part in path.parts):
                    continue
                ext = path.suffix.lower().lstrip(".")
                if ext in target_exts:
                    try:
                        res = self.index_file(str(path), force=force)
                        if res:
                            count += 1
                    except Exception as e:
                        print(f"Skipping {path}: {e}")
        return count

    def index_results_log(self, jsonl_path: str, max_records: int = 1000) -> int:
        """
        Ingest multimodal records from dispatcher.py's results_log.jsonl into the semantic index.
        """
        p = Path(jsonl_path)
        if not p.exists():
            return 0

        indexed_count = 0
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    status = record.get("status", "")
                    if status not in ("ok", "success"):
                        continue
                    file_path = record.get("path")
                    if not file_path or not os.path.exists(file_path):
                        continue

                    # Compose full searchable multimodal text representation
                    parts = [
                        f"File: {record.get('name', '')}",
                        f"Category: {record.get('category', '')}",
                    ]
                    desc = record.get("description", "")
                    if desc:
                        parts.append(f"Visual Description: {desc}")
                    extra = record.get("extra", {})
                    if extra and isinstance(extra, dict):
                        for k, v in extra.items():
                            if v:
                                parts.append(f"{k.capitalize()}: {v}")

                    content = "\n".join(parts)
                    if self.index_file(file_path, content=content):
                        indexed_count += 1
                        if indexed_count >= max_records:
                            break
                except Exception:
                    continue

        return indexed_count

    def index_memory_log(self, jsonl_path: Optional[str] = None) -> int:
        """
        Ingest records from AgentSix desktop's memory_log.jsonl.
        Each entry contains user captured screenshots, summaries, AI captions, and keywords.
        """
        if not jsonl_path:
            import os
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata:
                jsonl_path = os.path.join(local_appdata, "agentsix", "memory_log.jsonl")
            else:
                jsonl_path = str(Path.home() / ".local" / "share" / "agentsix" / "memory_log.jsonl")

        p = Path(jsonl_path)
        if not p.exists():
            return 0

        indexed_count = 0
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    try:
                        record = json.loads(line)
                    except Exception:
                        import re
                        fixed_line = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', line)
                        record = json.loads(fixed_line)

                    file_path = record.get("path")
                    if not file_path:
                        continue

                    fp = Path(file_path)
                    if not fp.exists():
                        continue

                    name = record.get("name") or fp.name
                    summary = record.get("summary") or ""
                    keywords = record.get("keywords") or []
                    text_preview = record.get("text_preview") or ""

                    ext = fp.suffix.lower().lstrip(".")
                    cat = self._detect_category(ext)

                    parts = [
                        f"File Name: {name}",
                        f"Category: {cat}",
                    ]
                    if summary:
                        parts.append(f"Visual Description / Summary: {summary}")
                    if keywords:
                        kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                        parts.append(f"Key Concepts & Tags: {kw_str}")
                    if text_preview:
                        parts.append(f"Content Analysis: {text_preview}")

                    content = "\n".join(parts)
                    fid = self.index_file(str(fp), content=content, force=True)
                    if fid:
                        if isinstance(keywords, list):
                            for kw in keywords:
                                if kw.strip():
                                    self.tag_file(str(fp), kw.strip())
                        indexed_count += 1
                except Exception as e:
                    print(f"Skipping memory record: {e}")
                    continue

        return indexed_count

    def remove_paths_containing(self, substring: str) -> int:
        """Remove indexed files containing a specific substring (e.g. workspace src code)."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT file_id, path FROM files WHERE path LIKE ?", (f"%{substring}%",)).fetchall()
            removed = 0
            for r in rows:
                conn.execute("DELETE FROM files WHERE file_id = ?", (r["file_id"],))
                removed += 1
            conn.commit()
            return removed

    def log_file_access(self, file_path: str, event_type: str = "open"):
        """Record file interaction for personalization scoring."""
        p = str(Path(file_path).resolve())
        with self._get_connection() as conn:
            row = conn.execute("SELECT file_id FROM files WHERE path = ?", (p,)).fetchone()
            if row:
                conn.execute(
                    "INSERT INTO access_logs (file_id, event_type, timestamp) VALUES (?, ?, ?)",
                    (row["file_id"], event_type, time.time())
                )
                conn.commit()

    def tag_file(self, file_path: str, tag: str):
        """Add organizational tag to a file."""
        p = str(Path(file_path).resolve())
        with self._get_connection() as conn:
            row = conn.execute("SELECT file_id FROM files WHERE path = ?", (p,)).fetchone()
            if row:
                conn.execute("""
                    INSERT OR IGNORE INTO file_tags (file_id, tag, created_at)
                    VALUES (?, ?, ?)
                """, (row["file_id"], tag.strip().lower(), datetime.now().isoformat()))
                conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM file_chunks").fetchone()[0]
            categories = conn.execute(
                "SELECT category, COUNT(*) as c FROM files GROUP BY category"
            ).fetchall()
            cat_dict = {row["category"]: row["c"] for row in categories}
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "categories": cat_dict,
            "db_path": str(self.db_path)
        }


class LiveDirectoryWatcher:
    """
    Watches directories using `watchdog` to dynamically update the semantic index
    when files are modified, created, or deleted.
    """
    def __init__(self, index: SemanticFileIndex, watch_dir: str):
        self.index = index
        self.watch_dir = watch_dir
        self.observer = None
        self._last_events: Dict[str, float] = {}

    def start(self):
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            index_ref = self.index
            last_events = self._last_events

            class Handler(FileSystemEventHandler):
                def _debounce(self, path: str, delay: float = 1.0) -> bool:
                    now = time.time()
                    if path in last_events and (now - last_events[path]) < delay:
                        return False
                    last_events[path] = now
                    return True

                def on_modified(self, event):
                    if not event.is_directory and self._debounce(event.src_path):
                        index_ref.index_file(event.src_path)

                def on_created(self, event):
                    if not event.is_directory and self._debounce(event.src_path):
                        index_ref.index_file(event.src_path)

                def on_deleted(self, event):
                    if not event.is_directory:
                        index_ref.delete_file(event.src_path)

            self.observer = Observer()
            self.observer.schedule(Handler(), self.watch_dir, recursive=True)
            self.observer.start()
        except ImportError:
            print("watchdog package not installed or failed to initialize.")

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
