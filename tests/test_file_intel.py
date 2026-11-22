"""
Comprehensive Integration & Unit Test Suite for file_intel.
Tests:
1. Text Chunking & Embeddings
2. Semantic Indexing & Dynamic Sync (Create, Modify, Delete)
3. FTS5 Keyword & Dense Hybrid Search
4. Two-Stage Ranking & Personalization (Recency & Affinity)
5. Agent Tool Calling & Context Memory
6. REST API Endpoints via FastAPI TestClient
"""

import os
import time
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path

from file_intel.config import DEFAULT_CHUNK_SIZE
from file_intel.embeddings import TextChunker, get_embedder, FastSemanticEmbedder
from file_intel.semantic_index import SemanticFileIndex
from file_intel.retriever_ranker import HybridFileRetriever, RankedResult
from file_intel.file_agent import FileAgent, ContextMemory
from file_intel.api import app
from fastapi.testclient import TestClient


class TestFileIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_file_intel_"))
        cls.test_db_path = cls.test_dir / "test_index.db"
        cls.index = SemanticFileIndex(db_path=cls.test_db_path)
        cls.retriever = HybridFileRetriever(cls.index)
        cls.agent = FileAgent(cls.index, cls.retriever)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_chunker_and_embedder(self):
        """Test intelligent text chunking and vector normalization."""
        chunker = TextChunker(chunk_size=120, chunk_overlap=30)
        sample_text = (
            "Line 1: High-performance vector embeddings for local files.\n"
            "Line 2: Chunking preserves sentence and paragraph boundaries.\n"
            "Line 3: Fast subword hashing guarantees deterministic results.\n"
            "Line 4: Testing overlap and line number extraction."
        )
        chunks = chunker.chunk_text(sample_text)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertGreater(chunks[0].end_line, 0)

        embedder = get_embedder()
        vec = embedder.embed_text(chunks[0].text)
        self.assertEqual(len(vec), 384)
        # Verify L2 normalization: norm should be ~1.0
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_02_semantic_index_lifecycle(self):
        """Test indexing, incremental update, and deletion lifecycle."""
        # Create a temp file
        doc_path = self.test_dir / "sample_doc.txt"
        doc_path.write_text("Alpha: This document contains secure encryption keys and network configs.", encoding="utf-8")

        # Initial index
        fid1 = self.index.index_file(str(doc_path))
        self.assertIsNotNone(fid1)

        stats = self.index.get_stats()
        self.assertGreaterEqual(stats["total_files"], 1)

        # Incremental sync: indexing again without modification should return same fid without re-chunking
        fid2 = self.index.index_file(str(doc_path))
        self.assertEqual(fid1, fid2)

        # Modify file
        time.sleep(0.05)
        doc_path.write_text("Beta: Updated file with database schema migration and SQLite tables.", encoding="utf-8")
        fid3 = self.index.index_file(str(doc_path))
        self.assertEqual(fid1, fid3)

        # Verify search picks up modified content
        results = self.retriever.search_hybrid("database schema migration", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertIn("database schema", results[0].text_snippet.lower())

        # Test deletion
        self.index.delete_file(str(doc_path))
        results_after_delete = self.retriever.search_hybrid("database schema migration", top_k=1)
        self.assertEqual(len(results_after_delete), 0)

    def test_03_hybrid_retrieval_and_rrf(self):
        """Test hybrid search with BM25 keyword + dense semantic vector fusion."""
        file_a = self.test_dir / "ai_model.py"
        file_a.write_text(
            "import torch\nclass TransformerLayer:\n    def forward(self, x):\n        return x * 2\n",
            encoding="utf-8"
        )
        file_b = self.test_dir / "finance_report.csv"
        file_b.write_text(
            "Q1,Revenue,$50000\nQ2,Revenue,$75000\nQ3,Profit,$25000\n",
            encoding="utf-8"
        )
        self.index.index_file(str(file_a))
        self.index.index_file(str(file_b))

        # Test code search
        results_code = self.retriever.search_hybrid("TransformerLayer forward pass", category="code")
        self.assertGreater(len(results_code), 0)
        self.assertEqual(results_code[0].file_name, "ai_model.py")

        # Test spreadsheet search
        results_fin = self.retriever.search_hybrid("Quarterly revenue profit", category="spreadsheet")
        self.assertGreater(len(results_fin), 0)
        self.assertEqual(results_fin[0].file_name, "finance_report.csv")

    def test_04_personalization_boost(self):
        """Test user affinity and active workspace personalization scoring."""
        file_c = self.test_dir / "workspace_proj" / "main_task.py"
        file_c.parent.mkdir(exist_ok=True)
        file_c.write_text("def run_task():\n    print('Executing background automation task')", encoding="utf-8")
        self.index.index_file(str(file_c))

        # Log repeated access events for file_c to simulate user affinity
        for _ in range(5):
            self.index.log_file_access(str(file_c), "open")

        results = self.retriever.search_hybrid(
            query="background automation task",
            active_workspace=str(file_c.parent),
            top_k=1
        )
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0].affinity_score, 0.0)
        self.assertEqual(results[0].workspace_boost, 1.0)
        self.assertIn("main_task.py", results[0].markdown_link)

    def test_05_agent_tool_calling_and_memory(self):
        """Test Agent context memory, snippet reading, tagging, and metadata extraction."""
        test_file = self.test_dir / "config_sample.yaml"
        test_file.write_text("app_name: AgentSix\nversion: 2.1\nport: 8080\ndebug: true\n", encoding="utf-8")
        self.index.index_file(str(test_file))

        # Test tool: tag_file
        res_tag = self.agent._tool_tag_file(str(test_file), "production")
        self.assertIn("Successfully tagged", res_tag)

        # Test tool: get_file_metadata
        res_meta = self.agent._tool_get_metadata(str(test_file))
        self.assertIn("config_sample.yaml", res_meta)
        self.assertIn("production", res_meta)

        # Test tool: read_file_snippet
        res_snippet = self.agent._tool_read_snippet(str(test_file), start_line=1, line_count=2)
        self.assertIn("app_name: AgentSix", res_snippet)
        self.assertIn("#L1-L2", res_snippet)

        # Test tool: list_recent_files
        res_recent = self.agent._tool_list_recent(limit=3)
        self.assertIn("config_sample.yaml", res_recent)

        # Test context memory
        self.agent.memory.pin_file(str(test_file))
        summary = self.agent.memory.get_context_summary()
        self.assertIn("config_sample.yaml", summary)

    def test_06_fastapi_rest_endpoints(self):
        """Test REST API endpoints."""
        # GET /api/health
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        # GET /api/stats
        res_stats = self.client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        self.assertIn("total_files", res_stats.json())

        # POST /api/search
        res_search = self.client.post("/api/search", json={"query": "face clustering", "top_k": 2})
        self.assertEqual(res_search.status_code, 200)
        self.assertIsInstance(res_search.json(), list)

        # POST /api/agent/ask
        res_ask = self.client.post("/api/agent/ask", json={"question": "What is in the file index?"})
        self.assertEqual(res_ask.status_code, 200)
        self.assertIn("answer", res_ask.json())


if __name__ == "__main__":
    unittest.main()
