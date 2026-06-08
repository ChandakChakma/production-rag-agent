"""
RAGAS evaluation pipeline — final production version.
Fixes:
  1. uvloop conflict: runs ragas in a background thread with its own event loop
  2. Surfaces all errors explicitly instead of silently returning 0
  3. OPENAI_API_KEY injected into env before ragas loads
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import os
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_settings
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGASReport:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    llm_correctness: float = 0.0
    llm_groundedness: float = 0.0
    llm_hallucination: float = 0.0
    num_samples: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "llm_correctness": round(self.llm_correctness, 4),
            "llm_groundedness": round(self.llm_groundedness, 4),
            "llm_hallucination": round(self.llm_hallucination, 4),
            "num_samples": self.num_samples,
            "errors": self.errors,
        }

    def summary(self) -> str:
        ragas_ok = self.faithfulness > 0 or self.answer_relevancy > 0
        lines = [
            f"{'=' * 52}",
            f"  EVALUATION REPORT — {self.num_samples} samples",
            f"{'=' * 52}",
            "  [RAGAS metrics]",
            f"  faithfulness        {self.faithfulness:.3f}" + ("" if ragas_ok else "  ← see errors"),
            f"  answer_relevancy    {self.answer_relevancy:.3f}",
            f"  context_precision   {self.context_precision:.3f}",
            f"  context_recall      {self.context_recall:.3f}",
            "  [LLM-Judge metrics]",
            f"  llm_correctness     {self.llm_correctness:.3f}",
            f"  llm_groundedness    {self.llm_groundedness:.3f}",
            f"  hallucination_rate  {self.llm_hallucination:.3f}",
            f"{'=' * 52}",
        ]
        if self.errors:
            for e in self.errors:
                lines.append(f"  ⚠  {e}")
        return "\n".join(lines)


class RAGASEvaluator:
    """
    Runs RAGAS metrics + LLM-judge scores on a test dataset.

    test_dataset format:
      [{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}]
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        # RAGAS reads the API key directly from the environment
        os.environ["OPENAI_API_KEY"] = self._settings.openai_api_key
        self._judge = LLMJudge()
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            from datasets import Dataset  # noqa: F401
            from ragas.metrics import (  # noqa: F401
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            logger.info("ragas_available")
            return True
        except Exception as exc:
            logger.warning(
                "ragas_unavailable",
                error=str(exc),
                fix="pip install ragas==0.1.14 datasets",
            )
            return False

    def evaluate(self, test_dataset: list[dict[str, Any]]) -> RAGASReport:
        report = RAGASReport(num_samples=len(test_dataset))

        if not test_dataset:
            return report

        if self._ragas_available:
            report = self._run_ragas(test_dataset, report)
        else:
            report.errors.append(
                "RAGAS unavailable — only LLM-Judge metrics computed. "
                "Run: pip install ragas==0.1.14 datasets"
            )

        report = self._run_llm_judge(test_dataset, report)
        logger.info("evaluation_complete", report=report.to_dict())
        return report

    def _run_ragas(self, dataset: list[dict[str, Any]], report: RAGASReport) -> RAGASReport:
        """
        Run ragas inside a background thread with its own plain asyncio event loop.

        WHY: ragas uses nest_asyncio to patch the running event loop so it can
        run async code synchronously. uvicorn runs on uvloop, and nest_asyncio
        cannot patch uvloop — it raises "Can't patch loop of type uvloop.Loop".

        FIX: submit ragas to a ThreadPoolExecutor. The new thread has no event
        loop yet; we create a plain asyncio one there. nest_asyncio can patch
        plain asyncio loops freely. The result is returned via a Future.
        """

        def _ragas_in_thread() -> dict[str, float]:
            # Create a fresh plain asyncio loop for this thread only
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from datasets import Dataset
                from ragas import evaluate
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )

                rows = []
                for d in dataset:
                    ctx = d.get("contexts", [])
                    if isinstance(ctx, str):
                        ctx = [ctx]
                    rows.append({
                        "question":     str(d.get("question", "")),
                        "answer":       str(d.get("answer", "")),
                        "contexts":     [str(c) for c in ctx] if ctx else [""],
                        "ground_truth": str(d.get("ground_truth", "")),
                    })

                hf_dataset = Dataset.from_list(rows)
                logger.info("ragas_eval_starting", samples=len(rows))

                result = evaluate(
                    hf_dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )

                scores = result.to_pandas()
                logger.info("ragas_eval_done", columns=list(scores.columns))

                def _safe_mean(col: str) -> float:
                    if col in scores.columns:
                        v = scores[col].dropna().mean()
                        return 0.0 if math.isnan(v) else float(v)
                    return 0.0

                return {
                    "faithfulness":      _safe_mean("faithfulness"),
                    "answer_relevancy":  _safe_mean("answer_relevancy"),
                    "context_precision": _safe_mean("context_precision"),
                    "context_recall":    _safe_mean("context_recall"),
                }
            finally:
                loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_ragas_in_thread)
                scores = future.result(timeout=120)  # 2-minute timeout

            report.faithfulness      = scores["faithfulness"]
            report.answer_relevancy  = scores["answer_relevancy"]
            report.context_precision = scores["context_precision"]
            report.context_recall    = scores["context_recall"]

        except concurrent.futures.TimeoutError:
            report.errors.append("RAGAS timed out after 120s — try fewer samples")
        except Exception as exc:
            logger.error("ragas_run_failed", error=str(exc), exc_info=True)
            report.errors.append(f"RAGAS error: {exc}")

        return report

    def _run_llm_judge(self, dataset: list[dict[str, Any]], report: RAGASReport) -> RAGASReport:
        try:
            batch_size = self._settings.eval_batch_size
            all_scores = []

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                examples = [
                    {
                        "question":     d.get("question", ""),
                        "answer":       d.get("answer", ""),
                        "context":      " ".join(
                            d["contexts"] if isinstance(d.get("contexts"), list)
                            else [d.get("context", "")]
                        ),
                        "ground_truth": d.get("ground_truth", ""),
                    }
                    for d in batch
                ]
                scores = self._judge.score_batch(examples)
                all_scores.extend(scores)
                logger.info("judge_batch_done", batch=i // batch_size + 1, size=len(batch))

            if all_scores:
                report.llm_correctness   = sum(s.correctness   for s in all_scores) / len(all_scores)
                report.llm_groundedness  = sum(s.groundedness  for s in all_scores) / len(all_scores)
                report.llm_hallucination = sum(s.hallucination for s in all_scores) / len(all_scores)

        except Exception as exc:
            logger.error("llm_judge_failed", error=str(exc), exc_info=True)
            report.errors.append(f"LLM Judge error: {exc}")

        return report