"""
file_intel configuration and path management.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(PROJECT_DIR))) / "agentsix"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database and Log Paths
DB_PATH = DATA_DIR / "file_index.db"
MEMORY_LOG_PATH = DATA_DIR / "memory_log.jsonl"
RESULTS_LOG_PATH = PROJECT_DIR / "Py module" / "results_log.jsonl"
SCAN_LOG_PATH = PROJECT_DIR / "Py module" / "scan_log.jsonl"

# Models and Binaries
TAURI_DIR = PROJECT_DIR / "src-tauri"
LLAMA_CLI_PATH = TAURI_DIR / "binaries" / "llama-cli.exe"
GEMMA_MODEL_PATH = TAURI_DIR / "models" / "gemma-2-2b-it-q4_k_m.gguf"
SMOLVLM_MODEL_PATH = TAURI_DIR / "models" / "smolvlm-256m-q4_k_m.gguf"

# Indexing Defaults
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 100
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit for full text indexing
EMBEDDING_DIM = 384
