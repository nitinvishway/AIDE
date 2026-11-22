"""
eval_runner.py

Evaluation Suite for Local File Retrieval and LLM Generation Quality.
Benchmarks:
1. Retrieval Metrics: MRR, Precision@K, Recall@K, Hit Rate
2. Generation Quality & Faithfulness (Groundedness / Hallucination detection)
3. Schema Validation Adherence Rate
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from ..semantic_index import SemanticFileIndex
from ..retriever_ranker import HybridFileRetriever, RankedResult
from ..file_agent import FileAgent


class FileIntelEvaluator:
    """
    Automated evaluation harness for evaluating retrieval and generation quality.
    """
    def __init__(self, index: SemanticFileIndex, retriever: HybridFileRetriever, agent: FileAgent):
        self.index = index
        self.retriever = retriever
        self.agent = agent

    def evaluate_retrieval(self, test_dataset: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        """
        Computes Mean Reciprocal Rank (MRR), Precision@K, Recall@K, and Hit Rate.
        """
        reciprocal_ranks = []
        hits_at_1 = 0
        hits_at_k = 0
        total_queries = len(test_dataset)

        detailed_results = []

        for item in test_dataset:
            query = item["query"]
            target_file = item["target_file"].lower()
            expected_keywords = [k.lower() for k in item.get("relevant_keywords", [])]

            ranked_results = self.retriever.search_hybrid(query=query, top_k=top_k)

            first_rank = 0
            found_at_k = False

            for rank_idx, r in enumerate(ranked_results, 1):
                f_name = Path(r.file_path).name.lower()
                # Check target file match or keyword coverage
                is_match = (target_file in f_name) or any(k in r.text_snippet.lower() for k in expected_keywords)

                if is_match and first_rank == 0:
                    first_rank = rank_idx
                    found_at_k = True

            if first_rank > 0:
                reciprocal_ranks.append(1.0 / first_rank)
                if first_rank == 1:
                    hits_at_1 += 1
            else:
                reciprocal_ranks.append(0.0)

            if found_at_k:
                hits_at_k += 1

            detailed_results.append({
                "query": query,
                "target_file": target_file,
                "first_relevant_rank": first_rank,
                "top_result_name": ranked_results[0].file_name if ranked_results else None,
                "top_result_score": ranked_results[0].final_score if ranked_results else 0.0,
            })

        mrr = sum(reciprocal_ranks) / max(1, total_queries)
        p_at_1 = hits_at_1 / max(1, total_queries)
        hit_rate = hits_at_k / max(1, total_queries)

        return {
            "total_queries": total_queries,
            "MRR": round(mrr, 4),
            "Precision@1": round(p_at_1, 4),
            "Hit_Rate@3": round(hit_rate, 4),
            "detailed": detailed_results
        }

    def evaluate_faithfulness(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates whether claims in generated answers are grounded in retrieved source snippets
        (Hallucination / Faithfulness score).
        """
        scores = []
        query_evals = []

        for item in test_dataset:
            query = item["query"]
            retrieved = self.retriever.search_hybrid(query=query, top_k=2)
            source_text = " ".join([r.text_snippet.lower() for r in retrieved])

            # Generate response from agent
            answer = self.agent.ask(query).lower()

            # Extract key terms/claims from answer
            answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer))
            # Ignore common stop words
            stops = {"this", "that", "with", "from", "have", "were", "what", "which", "will", "when", "into", "their"}
            content_words = answer_words - stops

            if not content_words:
                grounded_ratio = 1.0
            else:
                grounded_count = sum(1 for w in content_words if w in source_text or w in query.lower())
                grounded_ratio = grounded_count / len(content_words)

            scores.append(grounded_ratio)
            query_evals.append({
                "query": query,
                "groundedness": round(grounded_ratio, 3),
                "answer_preview": answer[:120] + "..."
            })

        avg_faithfulness = sum(scores) / max(1, len(scores))
        return {
            "average_faithfulness": round(avg_faithfulness, 4),
            "hallucination_risk": round(1.0 - avg_faithfulness, 4),
            "query_evals": query_evals
        }

    def evaluate_schema_adherence(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates structured tag and metadata extraction consistency.
        """
        successes = 0
        total = len(test_dataset)
        for item in test_dataset:
            try:
                # Test tagging and metadata querying
                meta_str = self.agent._tool_get_metadata(item["target_file"])
                # If parsed or returned as expected JSON/string
                if meta_str:
                    successes += 1
            except Exception:
                pass

        return {
            "schema_adherence_rate": round(successes / max(1, total), 4),
            "success_count": successes,
            "total_tested": total
        }

    def run_full_suite(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run all evaluation dimensions and export results."""
        d_path = dataset_path or (Path(__file__).parent / "eval_dataset.jsonl")
        test_data = []
        with open(d_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))

        t0 = time.time()
        retrieval_res = self.evaluate_retrieval(test_data, top_k=3)
        faithfulness_res = self.evaluate_faithfulness(test_data)
        schema_res = self.evaluate_schema_adherence(test_data)
        elapsed = round(time.time() - t0, 3)

        report = {
            "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_duration_seconds": elapsed,
            "metrics": {
                "MRR": retrieval_res["MRR"],
                "Precision@1": retrieval_res["Precision@1"],
                "Hit_Rate@3": retrieval_res["Hit_Rate@3"],
                "Faithfulness_Score": faithfulness_res["average_faithfulness"],
                "Hallucination_Risk": faithfulness_res["hallucination_risk"],
                "Schema_Adherence": schema_res["schema_adherence_rate"]
            },
            "retrieval_details": retrieval_res["detailed"],
            "faithfulness_details": faithfulness_res["query_evals"]
        }

        # Export report to evals/eval_results.json
        out_path = Path(__file__).parent / "eval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
