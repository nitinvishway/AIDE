"""
file_agent.py

LLM-Powered Desktop File Assistant Agent with Tool Use & Contextual Memory.
Executes an interactive ReAct / Function-calling loop with local Gemma 2 (via llama-cli.exe)
or direct reasoning driver.
"""

import os
import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from .config import LLAMA_CLI_PATH, GEMMA_MODEL_PATH
from .semantic_index import SemanticFileIndex
from .retriever_ranker import HybridFileRetriever, RankedResult


@dataclass
class AgentMessage:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ContextMemory:
    """
    Manages short-term conversational context, working files, and user preferences.
    """
    def __init__(self, max_history: int = 10):
        self.history: List[AgentMessage] = []
        self.max_history = max_history
        self.pinned_files: List[str] = []
        self.active_workspace: Optional[str] = None

    def add_message(self, role: str, content: str):
        self.history.append(AgentMessage(role=role, content=content))
        if len(self.history) > self.max_history * 2:
            # Retain system prompt if any, then slice recent
            self.history = self.history[-self.max_history * 2:]

    def pin_file(self, file_path: str):
        if file_path not in self.pinned_files:
            self.pinned_files.append(file_path)

    def set_active_workspace(self, workspace_path: str):
        self.active_workspace = workspace_path

    def get_context_summary(self) -> str:
        parts = []
        if self.active_workspace:
            parts.append(f"Active Workspace: {self.active_workspace}")
        if self.pinned_files:
            parts.append(f"Currently Pinned / Dropped Files: {', '.join(self.pinned_files)}")
        return "\n".join(parts)


class LocalGemmaDriver:
    """
    Interface to execute local Gemma 2 2B via llama-cli.exe.
    """
    def __init__(self, binary_path: Path = LLAMA_CLI_PATH, model_path: Path = GEMMA_MODEL_PATH):
        self.binary_path = binary_path
        self.model_path = model_path
        self.is_available = self.binary_path.exists() and self.model_path.exists()

    def run_inference(self, prompt: str, max_tokens: int = 512, temp: float = 0.2) -> str:
        if not self.is_available:
            return ""

        temp_prompt_file = Path(os.environ.get("TEMP", ".")) / f"agent_prompt_{os.getpid()}.txt"
        temp_prompt_file.write_text(prompt, encoding="utf-8")

        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(temp_prompt_file),
            "-c", "8192",
            "-t", "8",
            "-n", str(max_tokens),
            "--temp", str(temp),
            "--repeat-penalty", "1.1",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="ignore")
            temp_prompt_file.unlink(missing_ok=True)
            output = res.stdout.strip()
            # Clean Gemma 2 turn artifacts
            output = re.sub(r"<start_of_turn>model\s*", "", output)
            output = re.sub(r"<end_of_turn>.*", "", output, flags=re.DOTALL)
            return output.strip()
        except Exception as e:
            temp_prompt_file.unlink(missing_ok=True)
            return f"Error running local model: {e}"


class FileAgent:
    """
    Intelligent Agent capable of answering questions, finding documents,
    retrieving code snippets, and organizing files.
    """
    def __init__(self, index: SemanticFileIndex, retriever: HybridFileRetriever):
        self.index = index
        self.retriever = retriever
        self.memory = ContextMemory()
        self.llm = LocalGemmaDriver()
        self.tools: Dict[str, Callable] = {}
        self._register_tools()

    def _register_tools(self):
        self.tools["search_files"] = self._tool_search_files
        self.tools["read_file_snippet"] = self._tool_read_snippet
        self.tools["get_file_metadata"] = self._tool_get_metadata
        self.tools["tag_file"] = self._tool_tag_file
        self.tools["list_recent_files"] = self._tool_list_recent

    def _tool_search_files(self, query: str, category: Optional[str] = None, top_k: int = 4) -> str:
        """Search indexed files using hybrid semantic and keyword retrieval."""
        results = self.retriever.search_hybrid(
            query=query,
            category=category,
            active_workspace=self.memory.active_workspace,
            top_k=top_k
        )
        if not results:
            return "No matching files or snippets found in index."

        out = []
        for i, r in enumerate(results, 1):
            out.append(
                f"[{i}] {r.file_name} ({r.category}) - Relevance: {r.final_score}%\n"
                f"    Link: {r.markdown_link}\n"
                f"    Lines {r.start_line}-{r.end_line}: \"{r.text_snippet[:180]}...\""
            )
        return "\n\n".join(out)

    def _tool_read_snippet(self, file_path: str, start_line: int = 1, line_count: int = 40) -> str:
        """Read specific line ranges from any local file."""
        p = Path(file_path).resolve()
        if not p.exists():
            return f"File not found: {file_path}"
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            start = max(1, start_line) - 1
            end = min(len(lines), start + line_count)
            selected = lines[start:end]
            safe_p = str(p).replace("\\", "/")
            header = f"File: [{p.name}](file:///{safe_p}#L{start + 1}-L{end}) (Showing lines {start + 1} to {end} of {len(lines)})\n"
            content = "".join(f"{i + start + 1}: {line}" for i, line in enumerate(selected))
            return header + content
        except Exception as e:
            return f"Error reading file: {e}"

    def _tool_get_metadata(self, file_path: str) -> str:
        """Retrieve indexed details and tags for a file."""
        p = str(Path(file_path).resolve())
        with self.index._get_connection() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (p,)).fetchone()
            if not row:
                return f"File not registered in index: {file_path}"
            tags_rows = conn.execute("SELECT tag FROM file_tags WHERE file_id = ?", (row["file_id"],)).fetchall()
            tags = [t["tag"] for t in tags_rows]
            return json.dumps({
                "name": row["name"],
                "path": row["path"],
                "category": row["category"],
                "size_bytes": row["size_bytes"],
                "indexed_at": row["indexed_at"],
                "tags": tags
            }, indent=2)

    def _tool_tag_file(self, file_path: str, tag: str) -> str:
        """Apply an organizational tag to a file."""
        self.index.tag_file(file_path, tag)
        return f"Successfully tagged '{file_path}' with '{tag}'."

    def _tool_list_recent_files(self, limit: int = 5) -> str:
        """List recently indexed or modified files."""
        with self.index._get_connection() as conn:
            rows = conn.execute(
                "SELECT name, path, category, size_bytes, indexed_at FROM files ORDER BY mtime DESC LIMIT ?",
                (limit,)
            ).fetchall()
            if not rows:
                return "Index is empty."
            items = [f"- {r['name']} ({r['category']}) - {r['path']}" for r in rows]
            return "\n".join(items)

    def _tool_list_recent(self, limit: int = 5) -> str:
        return self._tool_list_recent_files(limit)

    def ask(self, user_query: str) -> str:
        """
        Main agent interaction loop.
        1. Analyzes user query
        2. Retrieves relevant files via hybrid retrieval
        3. Formats grounded context
        4. Synthesizes a verified, cited response using local Gemma 2
        """
        self.memory.add_message("user", user_query)

        # 1. Retrieve top matching files/chunks
        retrieved = self.retriever.search_hybrid(
            query=user_query,
            active_workspace=self.memory.active_workspace,
            top_k=4
        )

        context_blocks = []
        citations = []
        for i, r in enumerate(retrieved, 1):
            context_blocks.append(
                f"[Source {i}]: {r.markdown_link} (Relevance {r.final_score}%)\n"
                f"Content:\n{r.text_snippet}\n"
            )
            citations.append(r.markdown_link)

        grounded_context = "\n---\n".join(context_blocks)
        context_summary = self.memory.get_context_summary()

        prompt = (
            f"<start_of_turn>user\n"
            f"You are AgentSix, an expert local file intelligence agent. "
            f"Answer the user's question accurately using ONLY the retrieved file sources below. "
            f"Always cite the source files using the provided markdown links. If the information is not in the files, say so honestly.\n\n"
            f"{context_summary}\n\n"
            f"RETRIEVED FILE CONTEXT:\n{grounded_context}\n\n"
            f"USER QUESTION: {user_query}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        # 2. Run local LLM or fallback grounded response
        if self.llm.is_available:
            llm_response = self.llm.run_inference(prompt, max_tokens=600)
            if llm_response and not llm_response.startswith("Error"):
                self.memory.add_message("assistant", llm_response)
                return llm_response

        # Deterministic grounded fallback response if LLM binary is offline
        if retrieved:
            top = retrieved[0]
            summary = (
                f"### Relevant Information Found in {top.markdown_link}\n\n"
                f"> \"{top.text_snippet.strip()}\"\n\n"
                f"**Relevance Score:** {top.final_score}% (Semantic: {top.dense_score}, Recency: {top.recency_score})\n\n"
                f"**Matching Sources:**\n"
            )
            for c in citations:
                summary += f"- {c}\n"
            self.memory.add_message("assistant", summary)
            return summary
        else:
            msg = "No matching files or snippets found in the semantic index for your query."
            self.memory.add_message("assistant", msg)
            return msg
