"""
cli.py

Unified Command Line Interface for File Intelligence:
- Indexing files & directories
- Hybrid semantic & BM25 search
- Agent questioning with citations
- Real-time directory watching
- Automated benchmark evaluations
"""

import sys
import argparse
from pathlib import Path

from .semantic_index import SemanticFileIndex, LiveDirectoryWatcher
from .retriever_ranker import HybridFileRetriever
from .file_agent import FileAgent
from .evals.eval_runner import FileIntelEvaluator


def main():
    parser = argparse.ArgumentParser(description="AgentSix File Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", type=str, help="Path to file or folder to index")
    index_parser.add_argument("--recursive", action="store_true", default=True, help="Scan recursively")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the file index")
    search_parser.add_argument("query", type=str, help="Search terms or semantic query")
    search_parser.add_argument("--category", type=str, default=None, help="Filter category: document, code, image, other")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask the LLM File Agent a question")
    ask_parser.add_argument("question", type=str, help="Question to ask based on indexed files")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch a directory for real-time changes")
    watch_parser.add_argument("path", type=str, help="Directory path to monitor")

    # Eval command
    subparsers.add_parser("eval", help="Run the automated retrieval and quality evaluation suite")

    # Stats command
    subparsers.add_parser("stats", help="Display index database statistics")

    args = parser.parse_args()

    index = SemanticFileIndex()
    retriever = HybridFileRetriever(index)
    agent = FileAgent(index, retriever)

    if args.command == "index":
        target = Path(args.path).resolve()
        if target.is_file():
            print(f"Indexing file: {target}")
            fid = index.index_file(str(target))
            print(f"Done. File ID: {fid}")
        elif target.is_dir():
            print(f"Scanning and indexing directory: {target}")
            count = index.index_directory(str(target), recursive=args.recursive)
            print(f"Successfully indexed {count} files.")
        else:
            print(f"Error: Path does not exist: {args.path}")

    elif args.command == "search":
        print(f"\nSearching for: \"{args.query}\"")
        results = retriever.search_hybrid(query=args.query, category=args.category, top_k=args.top_k)
        if not results:
            print("No matching files found.")
            return

        print(f"\nFound {len(results)} matching files / snippets:\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r.file_name} ({r.category})")
            print(f"    Relevance Score: {r.final_score}% (Semantic: {r.dense_score}, Recency: {r.recency_score})")
            print(f"    Link: {r.markdown_link}")
            print(f"    Snippet: \"{r.text_snippet[:160]}...\"\n")

    elif args.command == "ask":
        print(f"\nQuestion: {args.question}")
        print("Analyzing index and retrieving grounded sources...")
        answer = agent.ask(args.question)
        print("\n" + "="*60)
        print(answer)
        print("="*60 + "\n")

    elif args.command == "watch":
        target = Path(args.path).resolve()
        if not target.is_dir():
            print(f"Path is not a directory: {target}")
            return
        print(f"Starting real-time directory watcher on: {target}")
        print("Press Ctrl+C to stop.")
        watcher = LiveDirectoryWatcher(index, str(target))
        watcher.start()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
            print("Watcher stopped.")

    elif args.command == "eval":
        print("\nRunning Evaluation Suite on benchmark dataset...")
        evaluator = FileIntelEvaluator(index, retriever, agent)
        report = evaluator.run_full_suite()

        print("\n" + "="*50)
        print("       EVALUATION BENCHMARK RESULTS")
        print("="*50)
        for k, v in report["metrics"].items():
            print(f"  {k:22}: {v}")
        print("="*50)
        print(f"Duration: {report['evaluation_duration_seconds']}s")
        print("Full report saved to file_intel/evals/eval_results.json\n")

    elif args.command == "stats":
        stats = index.get_stats()
        print("\n" + "="*40)
        print("       FILE INDEX DATABASE STATS")
        print("="*40)
        print(f"Total Files Indexed   : {stats['total_files']}")
        print(f"Total Chunks Embedded : {stats['total_chunks']}")
        print("Categories Breakdown  :")
        for cat, cnt in stats["categories"].items():
            print(f"  - {cat:15}: {cnt}")
        print(f"Database Path         : {stats['db_path']}")
        print("="*40 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
