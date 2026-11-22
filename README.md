# AIDE: Autonomous Intelligent Desktop Environment

A private, on-device multimodal AI desktop copilot with two-stage hybrid search, behavioral personalization, and offline local LLM reasoning.

[Releases](https://github.com/nitinvishway/AIDE/releases) | [Architecture](#architecture) | [Features](#key-features) | [Quick Start](#quick-start) | [Evaluation](#systematic-evaluation--benchmarks) | [API Reference](#api-reference)

---

## Overview

AIDE is a lightweight, non-intrusive floating desktop copilot built with Tauri v2, Rust, React, and Python. It functions as a private, on-device second brain that interfaces directly with local documents, screenshots, and media.

Unlike traditional desktop search tools that rely exclusively on filename matching or cloud AI services that upload private documents to third-party infrastructure, AIDE runs entirely locally:
1. **Semantic Understanding**: Parses and semantically indexes text, PDFs, code, screenshots, and visual media.
2. **Behavioral Personalization**: Re-ranks search results using interaction frequency, recency decay, and active workspace proximity.
3. **On-Device LLM Dialogue**: Executes local inference with Google Gemma 2, supporting continuous multi-turn conversations and deep chain-of-thought ("Think Mode") reasoning grounded in local file context.

---

## Key Features

### 1. On-Device LLM Dialogue and Chain-of-Thought Reasoning
* Powered by Google Gemma 2 (2B IT, Q4 quantized) running locally via GGUF runtime.
* **Continuous Multi-Turn Chat**: Contextual question answering, document summarization, and multi-turn message history tracking.
* **Think Mode**: Explicit chain-of-thought step-by-step reasoning executed under a dedicated thinking breakdown before synthesizing final conclusions.
* **Complete Privacy**: Zero telemetry, tokens, or document content transmitted over the network.

### 2. Two-Stage Hybrid Multimodal Search
* **Stage 1 (Hybrid Candidate Retrieval)**: Combines dense vector semantic similarity (`all-MiniLM-L6-v2`) and sparse keyword search (SQLite FTS5 BM25) fused via Reciprocal Rank Fusion (RRF).
* **Stage 2 (Cross-Factor Re-Ranking)**: Evaluates candidate chunks against personalized user factors:
  * **User Affinity Score**: Logarithmic frequency scoring based on files opened and queried most often.
  * **Exponential Recency Decay**: Prioritizes fresh items using an exponential decay curve (30-day half-life).
  * **Workspace Proximity Boost**: Grants priority to files located within the user's active working directory.
* **Intent Detection**: Automatically infers search target categories (images, screenshots, documents, code) from query syntax and applies targeted filtering.

### 3. Multimodal Vision Extraction
* Integrated on-device vision model (SmolVLM 256M) extracts natural language scene descriptions, interface components, and OCR text from screenshots, dropped images, and clipboard captures.
* Enables semantic queries over visual media, such as locating screenshots based on depicted software, interface elements, or visual content.

### 4. Event-Driven Dynamic Indexing
* Uses an event-driven `watchdog` file observer to monitor filesystem additions, edits, and deletions in real-time.
* Incremental sync ensures new files, screenshots, and memory entries are immediately queryable without manual database rebuilds.

### 5. Floating Desktop Interface
* **Fixed Width (175px)**: Compact, non-resizable floating widget designed to remain accessible without obstructing active workspace windows.
* **Glassmorphic UI**: Semi-transparent design with configurable themes (Glass Mode / Dark Mode).
* **Native OS Execution**: Direct execution hooks open retrieved documents, images, and search hits in default Windows host applications.

### 6. Automated Evaluation Framework
* Standalone evaluation harness (`file_intel/evals/eval_runner.py`) running benchmark datasets to compute standard Information Retrieval (IR) metrics:
  * Mean Reciprocal Rank (MRR)
  * Normalized Discounted Cumulative Gain (NDCG@k)
  * Precision@k and Recall@k
  * Retrieval Latency and Context Grounding

---

## Architecture

```
 +--------------------------------------------------------+
 |            AIDE Floating Widget (React 19)             |
 |   - Fixed 175px Canvas       - HTML5 Drag-and-Drop     |
 |   - Think Mode Toggle        - Multi-turn Chat Stream  |
 +---------------------------+----------------------------+
                             |
             +---------------+---------------+
             |                               |
      Tauri IPC (Rust)                   REST API (HTTP)
             |                               |
             v                               v
 +-----------------------+      +-------------------------+
 │   Tauri Rust Core     │      │   FastAPI Python Engine │
 │ - Native Shell Launch │      │ - Semantic Index (SQLite)│
 │ - Clipboard Monitor   │      │ - Hybrid Retriever      │
 │ - Local Gemma 2 Host  │      │ - Dynamic File Watcher  │
 +-----------------------+      +------------+------------+
                                             |
                       +---------------------+---------------------+
                       |                                           |
                       v                                           v
         +---------------------------+               +---------------------------+
         |     SQLite FTS5 + BLOB    |               |    Vision / SmolVLM       |
         | - Porter Stemming BM25    |               | - Scene Analysis          |
         | - Dense Vector Matrix     |               | - Local Image OCR         |
         | - User Affinity Logs      |               | - Metadata Extraction     |
         +---------------------------+               +---------------------------+
```

---

## Quick Start

### Option A: Pre-Compiled Release (Recommended)

1. Navigate to the [Releases](https://github.com/nitinvishway/AIDE/releases) page.
2. Select the distribution package:
   * **Full Standalone Bundle (`AIDE-v1.0.0-Full-Windows.zip`)**: Includes `agentsix.exe`, pre-installed Gemma 2 2B weights, SmolVLM vision models, and runtime DLLs. Extract and launch with zero manual setup.
   * **Lightweight Executable (`agentsix.exe`)**: 17 MB application binary for environments with existing model weights.
3. Launch the executable.

---

### Option B: Build from Source

#### Prerequisites
* Node.js (v18+) and npm
* Rust (stable toolchain with cargo)
* Python (v3.10+)
* WebView2 Runtime (default on Windows 10/11)

#### 1. Clone and Install Dependencies
```bash
git clone https://github.com/nitinvishway/AIDE.git
cd AIDE

# Install frontend dependencies
npm install

# Build static assets
npm run build
```

#### 2. Start the Backend Intelligence Service
```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic numpy sentence-transformers watchdog

# Run the local REST API server
python -m uvicorn file_intel.api:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Run the Desktop Application
```bash
# Development mode
npm run tauri dev

# Production release compilation
npm run tauri build
```

---

## API Reference

AIDE provides a local REST API via FastAPI running on `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/api/health` | Health check and system status |
| GET | `/api/stats` | Database statistics (file counts, chunk count, categories) |
| GET | `/api/search?q={query}&category={cat}&top_k={k}` | Execute two-stage hybrid retrieval and personalized re-ranking |
| POST | `/api/ask` | Query the autonomous agent with grounded document context |
| POST | `/api/index/file` | Embed and index a specified file path |
| POST | `/api/index/directory` | Recursively index a directory |
| POST | `/api/index/sync_memories` | Ingest and sync desktop memory logs and captures |

---

## Systematic Evaluation & Benchmarks

The retrieval engine includes an automated evaluation suite (`tests/test_file_intel.py` and `file_intel/evals/eval_runner.py`):

```bash
python -m unittest tests/test_file_intel.py
python -m unittest tests/test_multimodal_search.py
```

### Benchmark Results

| Metric | Baseline (Dense Vector Only) | AIDE Two-Stage Hybrid + Personalization |
| :--- | :--- | :--- |
| Mean Reciprocal Rank (MRR) | 0.68 | **0.89** (+30.8%) |
| NDCG@4 | 0.71 | **0.91** (+28.1%) |
| Precision@4 | 0.62 | **0.84** (+35.4%) |
| Query Latency | ~65 ms | **< 35 ms** |
| Test Suite Pass Rate | -- | **100% (8/8 Unit and Multimodal Tests Passing)** |

---

## Technology Stack

* **Desktop Framework**: Tauri v2, Rust 2021, React 19, TypeScript, Vite
* **Local LLM**: Google Gemma 2 (2B IT), GGUF quantization, llama.cpp runtime
* **Backend**: Python, FastAPI, Uvicorn, Pydantic
* **Search & Indexing**: SQLite3 with FTS5, Sentence-Transformers (`all-MiniLM-L6-v2`), NumPy
* **Filesystem Monitoring**: watchdog event-driven file listener
* **Vision Model**: SmolVLM 256M multimodal projector

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
