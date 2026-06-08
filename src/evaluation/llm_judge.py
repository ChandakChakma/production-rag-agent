"""
LLM-as-Judge evaluation — uses GPT-4 to score RAG outputs on multiple axes.

Metrics scored (0.0 – 1.0):
  - correctness      : Is the answer factually correct vs ground truth?
  - groundedness     : Is every claim in the answer supported by the context?
  - conciseness      : Is the answer appropriately brief (no padding)?
  - hallucination    : Does the answer contain unsupported claims? (lower = better)

Each metric is scored via a structured prompt asking for JSON output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JudgeScores:
    correctness: float = 0.0
    groundedness: float = 0.0
    conciseness: float = 0.0
    hallucination: float = 0.0  # lower is better
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "conciseness": self.conciseness,
            "hallucination": self.hallucination,
            "reasoning": self.reasoning,
        }


_JUDGE_PROMPT = """\
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Evaluate the following RAG output and return a JSON object with scores from 0.0 to 1.0.

QUESTION: {question}
RETRIEVED CONTEXT: {context}
GENERATED ANSWER: {answer}
GROUND TRUTH (if available): {ground_truth}

Score each dimension:
- correctness: Is the answer factually aligned with the ground truth? (1.0 = perfect match)
- groundedness: Is every claim in the answer supported by the retrieved context? (1.0 = fully grounded)
- conciseness: Is the answer appropriately concise without unnecessary padding? (1.0 = perfectly concise)
- hallucination: How much of the answer is NOT supported by the context? (0.0 = no hallucination, 1.0 = fully hallucinated)

Also provide a brief reasoning (2-3 sentences).

Respond ONLY with valid JSON:
{{
  "correctness": <float>,
  "groundedness": <float>,
  "conciseness": <float>,
  "hallucination": <float>,
  "reasoning": "<string>"
}}
"""


class LLMJudge:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.eval_llm_model
        logger.info("llm_judge_initialized", model=self._model)

    @retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
    def score(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str = "",
    ) -> JudgeScores:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],  # truncate to avoid token overflow
            answer=answer,
            ground_truth=ground_truth or "Not provided",
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            return JudgeScores(
                correctness=float(data.get("correctness", 0.0)),
                groundedness=float(data.get("groundedness", 0.0)),
                conciseness=float(data.get("conciseness", 0.0)),
                hallucination=float(data.get("hallucination", 1.0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("judge_parse_error", error=str(exc), raw=raw[:200])
            return JudgeScores(reasoning=f"Parse error: {exc}")

    def score_batch(
        self, examples: list[dict[str, str]]
    ) -> list[JudgeScores]:
        """
        examples: [{"question": ..., "answer": ..., "context": ..., "ground_truth": ...}]
        """
        results = []
        for ex in examples:
            score = self.score(
                question=ex.get("question", ""),
                answer=ex.get("answer", ""),
                context=ex.get("context", ""),
                ground_truth=ex.get("ground_truth", ""),
            )
            results.append(score)
        return results
