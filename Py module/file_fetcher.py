"""
file_fetcher.py

A general-purpose file fetcher that scans a directory (or the whole computer),
logs metadata for each file to disk AS IT GOES (so memory stays flat no matter
how many files exist), then sorts that log by file size once the scan is done.

A separate program (dispatcher.py) reads the sorted log afterwards and routes
each file to a type-specific handler.

Usage:
    from file_fetcher import FileFetcher

    fetcher = FileFetcher(whole_computer=True, recursive=True)
    log_path = fetcher.fetch_to_log("scan_log.jsonl")   # streams metadata, one file per line
    fetcher.sort_log_by_size(log_path, descending=True) # rewrites the log sorted by size

    # log is now ready for dispatcher.py to consume
"""

import os
import json
import platform
import string
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Iterable, Callable


# ---------------------------------------------------------------------------
# Extension -> category mapping. Extend this as needed for your dispatcher.
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "document": {".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt"},
    "spreadsheet": {".xlsx", ".xls", ".csv", ".tsv", ".ods"},
    "presentation": {".pptx", ".ppt", ".odp"},
    "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".tiff"},
    "audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"},
    "archive": {".zip", ".tar", ".gz", ".rar", ".7z", ".bz2"},
    "code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".html", ".css", ".json", ".xml", ".yaml", ".yml"},
    "ebook": {".epub", ".mobi", ".azw3"},
}

# Reverse lookup: extension -> category
_EXT_TO_CATEGORY = {
    ext: category
    for category, exts in CATEGORY_MAP.items()
    for ext in exts
}

# Directories that are noisy, huge, or system-critical — skipped by default
# when scanning the whole computer. Matched by directory name, not full path.
DEFAULT_SYSTEM_EXCLUDES = {
    ".git", "__pycache__", "node_modules", ".DS_Store",
    "$Recycle.Bin", "System Volume Information", "Windows", "WinSxS",
    "Program Files", "Program Files (x86)", "ProgramData",
    "proc", "sys", "dev", "run", "snap",
    "Library", "private",  # macOS system-heavy dirs (still lets user Library through if scanned explicitly)
    ".Trash", ".cache", "AppData",
}


def get_system_roots() -> List[str]:
    """
    Return sensible top-level roots to scan for a whole-computer search,
    depending on OS. On Windows this means every mounted drive letter;
    on macOS/Linux it means '/'.
    """
    system = platform.system()

    if system == "Windows":
        roots = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                roots.append(drive)
        return roots or ["C:\\"]

    # macOS / Linux
    return ["/"]


@dataclass
class FileInfo:
    """Metadata about a single fetched file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    modified_at: datetime
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "category": self.category,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FileFetcher:
    def __init__(
        self,
        directories: Optional[Iterable[str]] = None,
        whole_computer: bool = False,
        recursive: bool = True,
        extensions: Optional[Iterable[str]] = None,
        exclude_dirs: Optional[Iterable[str]] = None,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        follow_symlinks: bool = False,
        on_error=None,
    ):
        """
        :param directories: one or more root directories to scan. Ignored if
            whole_computer=True. Defaults to current directory if neither is given.
        :param whole_computer: if True, scan every drive/root on the machine
            (uses get_system_roots()) instead of `directories`.
        :param recursive: search subdirectories too
        :param extensions: only include these extensions, e.g. {'.pdf', '.jpg'} (None = all)
        :param exclude_dirs: directory names to skip entirely. Defaults to
            DEFAULT_SYSTEM_EXCLUDES when whole_computer=True, otherwise a small
            common set (.git, __pycache__, etc).
        :param min_size_bytes: skip files smaller than this
        :param max_size_bytes: skip files larger than this
        :param follow_symlinks: whether to follow symlinked directories (off by
            default to avoid infinite loops / escaping intended scan area)
        :param on_error: optional callback(OSError) invoked when a directory
            can't be read (e.g. permission denied). Defaults to silently skipping.
        """
        self.whole_computer = whole_computer

        if whole_computer:
            self.directories = [Path(r) for r in get_system_roots()]
            default_excludes = DEFAULT_SYSTEM_EXCLUDES
        else:
            dirs = directories if directories is not None else (".",)
            self.directories = [Path(os.path.expanduser(d)) for d in dirs]
            default_excludes = {".git", "__pycache__", "node_modules", ".DS_Store"}

        self.recursive = recursive
        self.extensions = {e.lower() for e in extensions} if extensions else None
        self.exclude_dirs = set(exclude_dirs) if exclude_dirs is not None else default_excludes
        self.min_size_bytes = min_size_bytes
        self.max_size_bytes = max_size_bytes
        self.follow_symlinks = follow_symlinks
        self.on_error = on_error or (lambda e: None)

    def _classify(self, extension: str) -> str:
        return _EXT_TO_CATEGORY.get(extension.lower(), "other")

    def _walk(self, root: Path):
        if not root.exists():
            return
        if self.recursive:
            for dirpath, dirnames, filenames in os.walk(
                root, onerror=self.on_error, followlinks=self.follow_symlinks
            ):
                # prune excluded directories in-place (by name, and hidden dirs
                # like .config, .local when whole_computer scanning)
                dirnames[:] = [
                    d for d in dirnames
                    if d not in self.exclude_dirs
                    and not (self.whole_computer and d.startswith("."))
                ]
                for fname in filenames:
                    yield Path(dirpath) / fname
        else:
            try:
                for entry in root.iterdir():
                    if entry.is_file():
                        yield entry
            except (OSError, PermissionError) as e:
                self.on_error(e)

    def fetch(self) -> List[FileInfo]:
        """Scan configured directories and return a list of FileInfo objects."""
        results: List[FileInfo] = []

        for root in self.directories:
            for path in self._walk(root):
                if not path.is_file():
                    continue

                ext = path.suffix.lower()
                if self.extensions and ext not in self.extensions:
                    continue

                try:
                    stat = path.stat()
                except (OSError, PermissionError):
                    continue

                size = stat.st_size
                if self.min_size_bytes is not None and size < self.min_size_bytes:
                    continue
                if self.max_size_bytes is not None and size > self.max_size_bytes:
                    continue

                results.append(
                    FileInfo(
                        path=str(path.resolve()),
                        name=path.name,
                        extension=ext,
                        category=self._classify(ext),
                        size_bytes=size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime),
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                    )
                )

        return results

    @staticmethod
    def group_by_category(files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        """Group a list of FileInfo objects by their category."""
        grouped: Dict[str, List[FileInfo]] = {}
        for f in files:
            grouped.setdefault(f.category, []).append(f)
        return grouped

    # -----------------------------------------------------------------
    # Streaming log mode — for whole-computer scans where holding every
    # FileInfo in memory at once isn't desirable. Each file is written to
    # the log the moment it's discovered (one JSON object per line, i.e.
    # JSONL), so memory usage stays flat regardless of scan size.
    # -----------------------------------------------------------------
    def fetch_to_log(
        self,
        log_path: str,
        progress_every: int = 5000,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """
        Scan and write one JSON line per file to `log_path` as files are found.

        :param log_path: where to write the JSONL metadata log
        :param progress_every: call progress_callback every N files (0 = never)
        :param progress_callback: optional fn(count_so_far, last_path) for
            progress reporting during a long whole-computer scan
        :return: the log_path (for chaining into sort_log_by_size)
        """
        count = 0
        with open(log_path, "w", encoding="utf-8") as log_file:
            for root in self.directories:
                for path in self._walk(root):
                    if not path.is_file():
                        continue

                    ext = path.suffix.lower()
                    if self.extensions and ext not in self.extensions:
                        continue

                    try:
                        stat = path.stat()
                    except (OSError, PermissionError) as e:
                        self.on_error(e)
                        continue

                    size = stat.st_size
                    if self.min_size_bytes is not None and size < self.min_size_bytes:
                        continue
                    if self.max_size_bytes is not None and size > self.max_size_bytes:
                        continue

                    record = {
                        "path": str(path.resolve()),
                        "name": path.name,
                        "extension": ext,
                        "category": self._classify(ext),
                        "size_bytes": size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    }
                    log_file.write(json.dumps(record) + "\n")

                    count += 1
                    if progress_callback and progress_every and count % progress_every == 0:
                        progress_callback(count, record["path"])

        return log_path

    @staticmethod
    def sort_log_by_size(log_path: str, descending: bool = True, output_path: Optional[str] = None) -> str:
        """
        Read back a JSONL log written by fetch_to_log, sort entries by
        size_bytes, and write the sorted result (overwriting log_path unless
        output_path is given).

        Note: this loads all *metadata* (not file contents) into memory to
        sort — for tens of millions of files this could get sizeable, but for
        a typical single-machine scan (hundreds of thousands of files) it's
        fine, since each record is just a small dict.
        """
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        records.sort(key=lambda r: r["size_bytes"], reverse=descending)

        target = output_path or log_path
        with open(target, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return target


# ---------------------------------------------------------------------------
# Example standalone run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    skipped = []

    fetcher = FileFetcher(
        whole_computer=True,   # scan every drive/root on this machine
        recursive=True,
        on_error=lambda e: skipped.append(str(e)),  # log permission errors instead of crashing
    )

    def report_progress(count, last_path):
        print(f"  ...{count} files logged so far (last: {last_path})")

    log_path = fetcher.fetch_to_log(
        "scan_log.jsonl",
        progress_every=5000,
        progress_callback=report_progress,
    )
    print(f"Scan complete. Metadata log written to: {log_path}")
    print(f"Skipped {len(skipped)} unreadable directories (permissions, etc.)")

    # Ascending = smallest files first. The dispatcher processes small,
    # cheap files first and works its way up to the big/slow ones
    # (large PDFs, videos) last.
    sorted_path = fetcher.sort_log_by_size(log_path, descending=False)
    print(f"Log sorted by size (smallest first): {sorted_path}")
    print("Ready for dispatcher.py to consume.")
