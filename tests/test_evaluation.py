"""Tests for LLM judge and evaluation pipeline."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


class TestLLMJudge:
    def _make_judge(self, mock_content: str):
        with patch("src.evaluation.llm_judge.OpenAI") as MockOpenAI:
            client = MagicMock()
            MockOpenAI.return_value = client
            response = MagicMock()
            response.choices[0].message.content = mock_content
            client.chat.completions.create.return_value = response

            from src.evaluation.llm_judge import LLMJudge
            judge = LLMJudge()
            judge._client = client
            return judge

    def test_score_returns_valid_struct(self):
        mock_json = json.dumps({
            "correctness": 0.9,
            "groundedness": 0.85,
            "conciseness": 0.8,
            "hallucination": 0.05,
            "reasoning": "The answer is accurate and grounded.",
        })
        judge = self._make_judge(mock_json)
        scores = judge.score(
            question="What is RAG?",
            answer="RAG combines retrieval with generation.",
            context="RAG stands for Retrieval-Augmented Generation.",
            ground_truth="RAG is a technique combining retrieval and generation.",
        )
        assert 0.0 <= scores.correctness <= 1.0
        assert 0.0 <= scores.groundedness <= 1.0
        assert 0.0 <= scores.hallucination <= 1.0

    def test_score_handles_invalid_json(self):
        judge = self._make_judge("not valid json {{{")
        scores = judge.score("q", "a", "c")
        # Should not raise; returns default scores
        assert scores.correctness == 0.0


class TestRAGASEvaluator:
    def test_evaluate_with_llm_judge_only(self):
        with patch("src.evaluation.ragas_eval.LLMJudge") as MockJudge:
            mock_judge = MagicMock()
            from src.evaluation.llm_judge import JudgeScores
            mock_judge.score_batch.return_value = [
                JudgeScores(correctness=0.9, groundedness=0.85, conciseness=0.8, hallucination=0.05)
            ] * 3
            MockJudge.return_value = mock_judge

            from src.evaluation.ragas_eval import RAGASEvaluator
            evaluator = RAGASEvaluator()
            evaluator._ragas_available = False  # skip RAGAS for unit test
            evaluator._judge = mock_judge

            dataset = [
                {
                    "question": "What is RAG?",
                    "answer": "RAG is retrieval augmented generation.",
                    "contexts": ["RAG combines retrieval with generation."],
                    "ground_truth": "RAG = Retrieval-Augmented Generation.",
                }
            ] * 3

            report = evaluator.evaluate(dataset)
            assert report.num_samples == 3
            assert report.llm_correctness > 0.0
            assert report.llm_hallucination < 1.0

    def test_empty_dataset_returns_zero_scores(self):
        from src.evaluation.ragas_eval import RAGASEvaluator
        with patch("src.evaluation.ragas_eval.LLMJudge"):
            evaluator = RAGASEvaluator()
            evaluator._ragas_available = False
            report = evaluator.evaluate([])
            assert report.num_samples == 0
            assert report.faithfulness == 0.0

    def test_report_summary_format(self):
        from src.evaluation.ragas_eval import RAGASReport
        report = RAGASReport(
            faithfulness=0.91,
            answer_relevancy=0.88,
            num_samples=50,
        )
        summary = report.summary()
        assert "0.910" in summary
        assert "50" in summary
