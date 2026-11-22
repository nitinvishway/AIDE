"""
retriever_ranker.py

Two-Stage Information Retrieval and Personalization Engine for Local Files.
Stage 1: Hybrid Retrieval (BM25 sparse full-text + Dense vector cosine similarity via RRF)
Stage 2: Cross-Factor Re-ranking with Personalization (Recency decay, User Affinity, Workspace Proximity)
"""

import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .embeddings import get_embedder
from .semantic_index import SemanticFileIndex


@dataclass
class RankedResult:
    chunk_id: str
    file_id: str
    file_path: str
    file_name: str
    category: str
    start_line: int
    end_line: int
    text_snippet: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    recency_score: float
    affinity_score: float
    workspace_boost: float
    final_score: float
    tags: List[str] = field(default_factory=list)
    markdown_link: str = ""


class HybridFileRetriever:
    """
    Two-stage retrieval and ranking engine designed for desktop file navigation.
    """

    def __init__(self, index: SemanticFileIndex):
        self.index = index
        self.embedder = get_embedder()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b_matrix: np.ndarray) -> np.ndarray:
        # Both are already L2 normalized
        return np.dot(vec_b_matrix, vec_a)

    def _detect_intent_category(self, query: str) -> Optional[str]:
        q = query.lower()
        if any(w in q for w in ["screenshot", "screen capture", "screencap", "image", "photo", "picture", "pic", "snapshot", "wallpaper", "visual"]):
            return "image"
        if any(w in q for w in ["video", "clip", "movie", "recording", "mp4", "mkv"]):
            if any(w in q for w in ["screenshot", "image", "photo", "picture", "frame", "screen"]):
                return "image"
            return "video"
        if any(w in q for w in ["pdf", "doc", "docx", "document", "sheet", "excel", "presentation", "slide", "memo"]):
            return "document"
        if any(w in q for w in ["source code", "rust code", "react component", "typescript", "python script", "def ", "function", "class"]):
            return "code"
        return None

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        active_workspace: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[RankedResult]:
        """Convenience alias for search_hybrid."""
        return self.search_hybrid(
            query=query,
            category=category,
            active_workspace=active_workspace,
            top_k=top_k,
            alpha=alpha,
        )

    def search_hybrid(
        self,
        query: str,
        category: Optional[str] = None,
        active_workspace: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,  # 1.0 = pure vector, 0.0 = pure BM25
    ) -> List[RankedResult]:
        """
        Executes two-stage retrieval and personalized ranking.
        """
        if not query.strip():
            return []

        q_vec = self.embedder.embed_text(query)
        conn = self.index._get_connection()

        # Category Intent Detection & Auto-Filtering
        filter_category = category
        if not filter_category or filter_category == "all":
            inferred = self._detect_intent_category(query)
            if inferred:
                count_row = conn.execute("SELECT COUNT(*) FROM files WHERE category = ?", (inferred,)).fetchone()
                if count_row and count_row[0] > 0:
                    filter_category = inferred

        # -------------------------------------------------------------
        # 1. Dense Vector Scoring
        # -------------------------------------------------------------
        filter_clause = "WHERE f.category = ?" if filter_category and filter_category != "all" else ""
        params = (filter_category,) if filter_category and filter_category != "all" else ()

        query_sql = f"""
            SELECT c.chunk_id, c.file_id, c.start_line, c.end_line, c.text, c.embedding,
                   f.path, f.name, f.category, f.mtime, f.size_bytes
            FROM file_chunks c
            JOIN files f ON c.file_id = f.file_id
            {filter_clause}
        """

        rows = conn.execute(query_sql, params).fetchall()
        if not rows:
            return []

        embeddings_list = []
        chunk_metadata = []

        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            embeddings_list.append(emb)
            chunk_metadata.append(r)

        emb_matrix = np.vstack(embeddings_list)
        dense_similarities = self._cosine_similarity(q_vec, emb_matrix)

        # Build dense ranks
        dense_sorted_indices = np.argsort(-dense_similarities)
        dense_ranks = {}
        for rank, idx in enumerate(dense_sorted_indices):
            chunk_id = chunk_metadata[idx]["chunk_id"]
            dense_ranks[chunk_id] = (rank + 1, float(dense_similarities[idx]))

        # -------------------------------------------------------------
        # 2. Sparse Keyword Scoring (SQLite FTS5 BM25)
        # -------------------------------------------------------------
        # Clean query tokens for FTS5 syntax, filtering common stopwords
        import re
        stopwords = {"and", "the", "for", "with", "this", "that", "from", "into", "over", "what", "where"}
        clean_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in stopwords]
        fts_query = " OR ".join(f'"{t}"*' for t in clean_tokens) if clean_tokens else query

        sparse_ranks = {}
        if fts_query:
            try:
                if filter_category and filter_category != "all":
                    fts_sql = f"""
                        SELECT fts.chunk_id, bm25(chunks_fts) as bm25_rank
                        FROM chunks_fts fts
                        JOIN file_chunks c ON fts.chunk_id = c.chunk_id
                        JOIN files f ON c.file_id = f.file_id
                        WHERE chunks_fts MATCH ? AND f.category = ?
                        ORDER BY bm25_rank ASC
                        LIMIT 200
                    """
                    fts_rows = conn.execute(fts_sql, (fts_query, filter_category)).fetchall()
                else:
                    fts_sql = f"""
                        SELECT chunk_id, bm25(chunks_fts) as bm25_rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY bm25_rank ASC
                        LIMIT 200
                    """
                    fts_rows = conn.execute(fts_sql, (fts_query,)).fetchall()
                for rank, row in enumerate(fts_rows):
                    # bm25 in sqlite: lower is better (usually negative). Map to positive score
                    sparse_ranks[row["chunk_id"]] = (rank + 1, abs(float(row["bm25_rank"])))
            except Exception:
                # Fallback if FTS syntax edge case
                pass

        # -------------------------------------------------------------
        # 3. Stage 1 Fusion: Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        k_rrf = 60.0
        candidate_results: List[RankedResult] = []
        now = time.time()

        # Pre-fetch user affinity logs
        access_counts = {}
        log_rows = conn.execute(
            "SELECT file_id, COUNT(*) as cnt FROM access_logs GROUP BY file_id"
        ).fetchall()
        for lr in log_rows:
            access_counts[lr["file_id"]] = lr["cnt"]

        # Pre-fetch tags
        file_tags_map = {}
        tag_rows = conn.execute("SELECT file_id, tag FROM file_tags").fetchall()
        for tr in tag_rows:
            file_tags_map.setdefault(tr["file_id"], []).append(tr["tag"])

        for idx, meta in enumerate(chunk_metadata):
            cid = meta["chunk_id"]
            fid = meta["file_id"]
            d_rank, d_sim = dense_ranks.get(cid, (len(rows) + 1, 0.0))
            s_rank, s_score = sparse_ranks.get(cid, (len(rows) + 1, 0.0))

            # Normalize dense sim to [0, 1]
            norm_dense = max(0.0, min(1.0, (d_sim + 1.0) / 2.0))

            # RRF score
            rrf_dense = 1.0 / (k_rrf + d_rank)
            rrf_sparse = 1.0 / (k_rrf + s_rank) if cid in sparse_ranks else 0.0
            hybrid_rrf = alpha * rrf_dense + (1.0 - alpha) * rrf_sparse

            # -------------------------------------------------------------
            # 4. Stage 2 Personalization & Re-ranking Factors
            # -------------------------------------------------------------
            # Recency decay (half-life of 30 days)
            age_days = max(0.0, (now - meta["mtime"]) / 86400.0)
            recency_score = math.exp(-0.023 * min(age_days, 180.0))

            # User affinity (logarithmic frequency boost)
            user_hits = access_counts.get(fid, 0)
            affinity_score = min(1.0, math.log1p(user_hits) / math.log1p(20))

            # Workspace / active directory proximity
            workspace_boost = 0.0
            if active_workspace:
                try:
                    p = Path(meta["path"]).resolve()
                    ws = Path(active_workspace).resolve()
                    if ws in p.parents or p == ws:
                        workspace_boost = 1.0
                except Exception:
                    pass

            # Final multi-factor composite score
            # Base hybrid relevance + dense match are primary (85%), while recency & affinity act as gentle tie-breakers (15%)
            final_score = (
                0.50 * (hybrid_rrf * 100.0) +
                0.35 * (norm_dense * 100.0) +
                0.05 * (recency_score * 100.0) +
                0.05 * (affinity_score * 100.0) +
                0.05 * (workspace_boost * 100.0)
            )

            # Generate markdown file link with forward slashes: [name](file:///path#Lstart-Lend)
            safe_path = meta["path"].replace("\\", "/")
            md_link = f"[{meta['name']}](file:///{safe_path}#L{meta['start_line']}-L{meta['end_line']})"

            tags = file_tags_map.get(fid, [])

            candidate_results.append(RankedResult(
                chunk_id=cid,
                file_id=fid,
                file_path=meta["path"],
                file_name=meta["name"],
                category=meta["category"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                text_snippet=meta["text"],
                dense_score=round(norm_dense, 4),
                sparse_score=round(s_score, 4),
                hybrid_score=round(hybrid_rrf * 100.0, 4),
                recency_score=round(recency_score, 4),
                affinity_score=round(affinity_score, 4),
                workspace_boost=round(workspace_boost, 4),
                final_score=round(final_score, 2),
                tags=tags,
                markdown_link=md_link
            ))

        # Sort descending by final composite score
        candidate_results.sort(key=lambda x: x.final_score, reverse=True)
        return candidate_results[:top_k]
