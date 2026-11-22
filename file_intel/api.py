"""
api.py

FastAPI REST Service for Desktop & Web Integration.
Enables React (App.tsx / dashboard.html) and Tauri to communicate directly
with the semantic file index, hybrid retriever, and LLM agent.
"""

from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .semantic_index import SemanticFileIndex
from .retriever_ranker import HybridFileRetriever
from .file_agent import FileAgent
from .evals.eval_runner import FileIntelEvaluator

app = FastAPI(title="AgentSix File Intelligence API", version="1.0.0")

# Enable CORS for desktop webviews and local dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
index = SemanticFileIndex()
retriever = HybridFileRetriever(index)
agent = FileAgent(index, retriever)
evaluator = FileIntelEvaluator(index, retriever, agent)


@app.on_event("startup")
def startup_event():
    # Clean out developer workspace code from user's personal file index
    index.remove_paths_containing("1626\\src")
    index.remove_paths_containing("1626\\Py module")
    # Ingest user desktop captures and memories
    count = index.index_memory_log()
    print(f"[API Startup] Ingested {count} memory records from agentsix.")


@app.post("/api/index/sync_memories")
def sync_memories():
    count = index.index_memory_log()
    return {"status": "ok", "indexed_memories": count}


class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    top_k: int = 5
    active_workspace: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    active_workspace: Optional[str] = None
    pinned_files: Optional[List[str]] = None


class IndexFileRequest(BaseModel):
    path: str
    content: Optional[str] = None


class IndexDirRequest(BaseModel):
    path: str
    recursive: bool = True


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AgentSix File Intelligence"}


@app.get("/api/stats")
def get_stats():
    return index.get_stats()


@app.post("/api/index/file")
def index_file(req: IndexFileRequest):
    fid = index.index_file(req.path, content=req.content)
    if not fid:
        raise HTTPException(status_code=400, detail="Failed to index file or file not found")
    return {"file_id": fid, "status": "indexed"}


@app.post("/api/index/directory")
def index_directory(req: IndexDirRequest):
    count = index.index_directory(req.path, recursive=req.recursive)
    return {"indexed_files_count": count, "status": "completed"}


@app.get("/api/search")
def search_get(q: str, category: Optional[str] = None, top_k: int = 5):
    results = retriever.search_hybrid(
        query=q,
        category=category,
        top_k=top_k
    )
    return [
        {
            "chunk_id": r.chunk_id,
            "file_name": r.file_name,
            "file_path": r.file_path,
            "category": r.category,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "snippet": r.text_snippet,
            "final_score": r.final_score,
            "dense_score": r.dense_score,
            "recency_score": r.recency_score,
            "markdown_link": r.markdown_link,
            "tags": r.tags
        }
        for r in results
    ]


@app.post("/api/search")
def search(req: SearchRequest):
    results = retriever.search_hybrid(
        query=req.query,
        category=req.category,
        active_workspace=req.active_workspace,
        top_k=req.top_k
    )
    return [
        {
            "chunk_id": r.chunk_id,
            "file_name": r.file_name,
            "file_path": r.file_path,
            "category": r.category,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "snippet": r.text_snippet,
            "final_score": r.final_score,
            "dense_score": r.dense_score,
            "recency_score": r.recency_score,
            "markdown_link": r.markdown_link,
            "tags": r.tags
        }
        for r in results
    ]


@app.post("/api/agent/ask")
def ask_agent(req: AskRequest):
    if req.active_workspace:
        agent.memory.set_active_workspace(req.active_workspace)
    if req.pinned_files:
        for f in req.pinned_files:
            agent.memory.pin_file(f)

    answer = agent.ask(req.question)
    return {
        "question": req.question,
        "answer": answer,
        "pinned_files": agent.memory.pinned_files
    }


@app.post("/api/evals/run")
def run_evals():
    report = evaluator.run_full_suite()
    return report
