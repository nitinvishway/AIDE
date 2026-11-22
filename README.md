# AIDE: Autonomous Intelligent Desktop Environment

<div align="center">

[![Release](https://img.shields.io/github/v/release/nitinvishway/AIDE?style=flat-square&color=blue)](https://github.com/nitinvishway/AIDE/releases)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-informational?style=flat-square&logo=windows)](https://github.com/nitinvishway/AIDE)
[![Tauri](https://img.shields.io/badge/Tauri-v2-24C8D8?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Rust](https://img.shields.io/badge/Rust-2021-black?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A private, on-device multimodal AI desktop copilot with two-stage hybrid search, behavioral personalization, and offline local LLM reasoning.**

[Download Release](https://github.com/nitinvishway/AIDE/releases) • [Architecture](#-architecture) • [Features](#-key-features) • [Quick Start](#-quick-start) • [Evaluation](#-systematic-evaluation--benchmarks)

</div>

---

## 📌 Overview

**AIDE** is a non-intrusive, floating desktop copilot built with **Tauri v2**, **Rust**, **React**, and **Python**. It functions as a private, on-device second brain that connects directly to your local documents, screenshots, and media.

Unlike traditional desktop search tools that rely solely on filename matching or cloud AI services that upload private files, AIDE processes everything locally:
1. **Understands files semantically** across text, PDFs, code, screenshots, and visual media.
2. **Ranks files personalized to your behavior** using interaction frequency, recency decay, and active workspace proximity.
3. **Converses using local LLMs** (Google Gemma 2) with deep chain-of-thought ("Think Mode") reasoning grounded in your local documents.

---

## 🚀 Key Features

### 1. 🧠 On-Device LLM Dialogue & "Think Mode"
* Powered by **Google Gemma 2 (2B IT Q4)** running 100% locally via quantized GGUF runtime.
* **Continuous Multi-Turn Chat:** Ask questions, summarize documents, or trace ideas across long conversation threads.
* **Think Mode (Chain-of-Thought):** Enables deep step-by-step reasoning under a dedicated `### 🧠 Thinking Process` breakdown before synthesizing structured conclusions.
* **100% Private & Offline:** Zero tokens or data sent to external cloud servers.

### 2. 🔍 Two-Stage Hybrid Multimodal Search
* **Stage 1 (Hybrid Candidate Retrieval):** Combines dense vector semantic similarity (`all-MiniLM-L6-v2`) and sparse full-text keyword search (**SQLite FTS5 BM25**) fused via **Reciprocal Rank Fusion (RRF)**.
* **Stage 2 (Cross-Factor Re-Ranking):** Evaluates candidate chunks against personalized user factors:
  * **User Affinity Score:** Logarithmic boost based on files you interact with and open most.
  * **Exponential Recency Decay:** Prioritizes fresh items (30-day half-life decay).
  * **Workspace Proximity Boost:** Prioritizes files in your currently active project directory.
* **Smart Intent Auto-Filtering:** Automatically detects queries for screenshots, media, documents, or code, and routes them to the appropriate category filter.

### 3. 🖼️ Multimodal Vision Extraction
* Bundles an on-device vision model (**SmolVLM 256M**) to extract visual scene descriptions, UI elements, and OCR text from dropped images, screenshots, and screen captures.
* Enables natural queries like *"find the screenshot of a video"* or *"locate the diagram of neural network layers"*.

### 4. ⚡ Zero-Downtime Dynamic Indexing
* Uses an event-driven `watchdog` file observer to track filesystem modifications, additions, and deletions in real-time.
* Incremental sync ensures new files, screenshots, and clipboard grabs are instantly searchable without manual database rebuilds.

### 5. 🪟 Minimalist Floating Desktop Interface
* **Hardcoded 175px Fixed Width:** Compact, non-resizable floating widget that never clutters your screen.
* **Glassmorphic Aesthetic:** Semi-transparent glass styling with custom theme switching (Glass Mode / Dark Mode).
* **Native Windows Integration:** One-click `[Open]` button launches files directly in your native default Windows applications (Photos, Acrobat, Notepad, Office).

### 6. 📊 Automated LLM Evaluation Framework
* Offline evaluation harness (`file_intel/evals/eval_runner.py`) running benchmark datasets to compute standard Information Retrieval (IR) metrics:
  * **MRR (Mean Reciprocal Rank)**
  * **NDCG@k (Normalized Discounted Cumulative Gain)**
  * **Precision@k & Recall@k**
  * **Retrieval Latency & Context Relevance**

---

## 🏗️ Architecture

```
 ┌────────────────────────────────────────────────────────┐
 │            AIDE Floating Widget (React 19)             │
 │   - Fixed 175px Canvas       - HTML5 Drag-and-Drop     │
 │   - Think Mode Toggle        - Multi-turn Chat Stream  │
 └───────────┬────────────────────────────────┬───────────┘
             │                                │
      Tauri IPC (Rust)                   REST API (HTTP)
             │                                │
             ▼                                ▼
 ┌───────────────────────┐      ┌─────────────────────────┐
 │   Tauri Rust Core     │      │   FastAPI Python Engine │
 │ - Native Shell Launch │      │ - Semantic Index (SQLite)│
 │ - Clipboard Monitor   │      │ - Hybrid Retriever      │
 │ - Local Gemma 2 Host  │      │ - Dynamic File Watcher  │
 └───────────────────────┘      └────────────┬────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │     SQLite FTS5 + BLOB    │               │    Vision / SmolVLM       │
         │ - Porter Stemming BM25    │               │ - Scene Analysis          │
         │ - Dense Vector Matrix     │               │ - Local Image OCR         │
         │ - User Affinity Logs      │               │ - Metadata Extraction     │
         └───────────────────────────┘               └───────────────────────────┘
```

---

## 📦 Quick Start

### Option A: Download Standalone Release (Recommended)

1. Go to the [Releases](https://github.com/nitinvishway/AIDE/releases) page.
2. Choose your package:
   * **Full Offline Bundle (`AIDE-v1.0.0-Full-Windows.zip`):** Includes `AIDE.exe`, all pre-installed Gemma 2 2B LLM weights, vision models, and runtime DLLs.
   * **Lightweight Shell (`aide.exe`):** Standalone 17 MB executable for environments with pre-existing model weights.
3. Extract and run `AIDE.exe`.

---

### Option B: Build from Source

#### Prerequisites
* **Node.js** (v18+) & `npm`
* **Rust** (stable toolchain with `cargo`)
* **Python** (v3.10+)
* **WebView2 Runtime** (installed by default on Windows 10/11)

#### 1. Clone & Setup Frontend / Tauri
```bash
git clone https://github.com/nitinvishway/AIDE.git
cd AIDE

# Install JavaScript dependencies
npm install

# Build frontend assets
npm run build
```

#### 2. Setup Python File Intelligence Service
```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic numpy sentence-transformers watchdog

# Start the background intelligence service
python -m uvicorn file_intel.api:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Run Desktop Application
```bash
# Launch in development mode
npm run tauri dev

# Or compile optimized release executable
npm run tauri build
```

---

## 📡 API Reference

AIDE exposes a clean local REST API via FastAPI at `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and service status |
| `GET` | `/api/stats` | Total files, chunks, categories, and database stats |
| `GET` | `/api/search?q={query}&category={cat}&top_k={k}` | Execute two-stage hybrid search & re-ranking |
| `POST` | `/api/ask` | Query the autonomous agent with document grounding |
| `POST` | `/api/index/file` | Ingest and embed a specific file path |
| `POST` | `/api/index/directory` | Recursively scan and index a directory |
| `POST` | `/api/index/sync_memories` | Re-sync desktop memory records and captures |

---

## 📊 Systematic Evaluation & Benchmarks

The file intelligence engine includes an automated evaluation harness (`tests/test_file_intel.py` and `file_intel/evals/eval_runner.py`):

```bash
python -m unittest tests/test_file_intel.py
python -m unittest tests/test_multimodal_search.py
```

### Benchmark Metrics

| Evaluation Metric | Baseline (Pure Vector) | AIDE Two-Stage Hybrid + Personalization |
| :--- | :--- | :--- |
| **MRR (Mean Reciprocal Rank)** | 0.68 | **0.89** (+30.8%) |
| **NDCG@4** | 0.71 | **0.91** (+28.1%) |
| **Precision@4** | 0.62 | **0.84** (+35.4%) |
| **Query Latency** | ~65 ms | **< 35 ms** (Indexed FTS + In-Memory BLOB) |
| **Test Suite Pass Rate** | — | **100% (8/8 Unit & Multimodal Tests Passing)** |

---

## 🛠️ Tech Stack

* **Desktop Application:** [Tauri v2](https://tauri.app), Rust 2021, [React 19](https://react.dev), TypeScript, Vite
* **Local LLM Engine:** [Google Gemma 2 (2B IT)](https://ai.google.dev/gemma), llama.cpp runtime, GGUF quantization
* **Backend Services:** Python, [FastAPI](https://fastapi.tiangolo.com/), Uvicorn, Pydantic
* **Search & Retrieval:** SQLite3 with FTS5, Sentence-Transformers (`all-MiniLM-L6-v2`), NumPy matrix cosine similarity
* **File Observer:** [watchdog](https://github.com/gorakhargosh/watchdog) event-driven filesystem listener
* **Vision Model:** SmolVLM (256M) multimodal vision projector

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
