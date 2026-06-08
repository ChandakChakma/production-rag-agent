#!/usr/bin/env python
"""
Run the full evaluation suite against the loaded knowledge base.
Outputs RAGAS + LLM-Judge scores to console and saves results JSON.

Run: python scripts/run_eval_suite.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

console = Console()

# ── Test dataset ──────────────────────────────────────────────────────────────
# In production, load from a curated golden dataset file.
TEST_DATASET = [
    {
        "question": "What is retrieval-augmented generation (RAG)?",
        "ground_truth": "RAG is a technique that combines information retrieval with language model generation to ground outputs in retrieved documents.",
    },
    {
        "question": "What is hybrid retrieval?",
        "ground_truth": "Hybrid retrieval combines dense vector search with sparse BM25 keyword search, fused using techniques like reciprocal rank fusion.",
    },
    {
        "question": "What is a cross-encoder reranker?",
        "ground_truth": "A cross-encoder scores query-document pairs jointly, providing more accurate relevance scores than bi-encoders at the cost of higher latency.",
    },
    {
        "question": "What is the ReAct agent architecture?",
        "ground_truth": "ReAct interleaves reasoning and acting, allowing agents to think through a problem, call tools, observe results, and iterate.",
    },
    {
        "question": "What metrics does RAGAS use to evaluate RAG?",
        "ground_truth": "RAGAS evaluates faithfulness, answer relevancy, context precision, and context recall.",
    },
]


def main():
    console.print("\n[bold cyan]📊 Production RAG System — Evaluation Suite[/bold cyan]\n")

    from src.config.settings import get_settings
    from src.evaluation.ragas_eval import RAGASEvaluator
    from src.ingestion.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.vector_store import get_vector_store
    from src.utils.logger import configure_logging
    from openai import OpenAI

    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")

    console.print(f"[dim]Eval LLM: {settings.eval_llm_model} | Samples: {len(TEST_DATASET)}[/dim]\n")

    # ── Build pipeline ────────────────────────────────────────────────
    vector_store = get_vector_store()
    embedder = Embedder()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    reranker = CrossEncoderReranker()
    client = OpenAI(api_key=settings.openai_api_key)

    # ── Auto-generate answers + contexts ─────────────────────────────
    console.print("[yellow]⟳[/yellow] Generating answers for test questions...")
    dataset = []
    for ex in TEST_DATASET:
        q = ex["question"]
        emb = embedder.embed_text(q)
        candidates = retriever.retrieve(q, top_k=settings.retrieval_top_k)
        ranked = reranker.rerank(q, candidates, top_k=5)
        contexts = [r.text for r in ranked]

        ctx = "\n\n".join(contexts) if contexts else "No context retrieved."
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{
                "role": "user",
                "content": f"Answer using context only:\n{ctx}\n\nQuestion: {q}\nAnswer:"
            }],
            temperature=0.0,
            max_tokens=500,
        )
        answer = resp.choices[0].message.content or ""
        dataset.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ex["ground_truth"],
        })

    console.print(f"[green]✓[/green] Generated {len(dataset)} answers\n")

    # ── Run evaluation ────────────────────────────────────────────────
    console.print("[yellow]⟳[/yellow] Running evaluation (RAGAS + LLM-Judge)...")
    t0 = time.perf_counter()
    evaluator = RAGASEvaluator()
    report = evaluator.evaluate(dataset)
    elapsed = time.perf_counter() - t0

    # ── Print results ─────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")

    def grade(v: float, higher_better: bool = True) -> str:
        effective = v if higher_better else 1 - v
        if effective >= 0.85:
            return "[bold green]A[/bold green]"
        if effective >= 0.70:
            return "[yellow]B[/yellow]"
        return "[red]C[/red]"

    metrics = [
        ("faithfulness", report.faithfulness, True),
        ("answer_relevancy", report.answer_relevancy, True),
        ("context_precision", report.context_precision, True),
        ("context_recall", report.context_recall, True),
        ("llm_correctness", report.llm_correctness, True),
        ("llm_groundedness", report.llm_groundedness, True),
        ("hallucination_rate", report.llm_hallucination, False),
    ]
    for name, val, higher in metrics:
        table.add_row(name, f"{val:.3f}", grade(val, higher))

    console.print(table)
    console.print(f"\n[dim]Samples: {report.num_samples} | Eval time: {elapsed:.1f}s[/dim]")

    if report.errors:
        console.print(f"\n[yellow]Warnings:[/yellow] {report.errors}")

    # ── Save results ──────────────────────────────────────────────────
    out_path = "data/eval_report.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report.to_dict(), "dataset_size": len(dataset)}, f, indent=2)
    console.print(f"\n[green]✓[/green] Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
