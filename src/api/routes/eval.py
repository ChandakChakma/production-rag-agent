"""Evaluation route — runs RAGAS + LLM-judge on a test dataset."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class EvalExample(BaseModel):
    question: str
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""


class EvalRequest(BaseModel):
    test_dataset: list[EvalExample] = Field(..., min_length=1)
    collection: str = "default"
    auto_generate_answers: bool = Field(
        False, description="If true, run RAG to generate answers before eval"
    )


class EvalResponse(BaseModel):
    report: dict[str, Any]
    summary: str


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest, request: Request) -> EvalResponse:
    components = request.app.state.components

    dataset = [ex.model_dump() for ex in req.test_dataset]

    if req.auto_generate_answers:
        dataset = await _auto_generate(dataset, components)

    try:
        from src.evaluation.ragas_eval import RAGASEvaluator
        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(dataset)
        return EvalResponse(report=report.to_dict(), summary=report.summary())
    except Exception as exc:
        logger.error("eval_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


async def _auto_generate(dataset: list[dict], components: dict) -> list[dict]:
    """Run direct RAG to fill in answers + contexts before evaluation."""
    embedder = components["embedder"]
    retriever = components["retriever"]
    reranker = components["reranker"]
    settings = components["settings"]

    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

    for ex in dataset:
        q = ex["question"]
        emb = embedder.embed_text(q)
        candidates = retriever.retrieve(q, top_k=settings.retrieval_top_k)
        ranked = reranker.rerank(q, candidates, top_k=5)
        ex["contexts"] = [r.text for r in ranked]

        ctx = "\n\n".join(ex["contexts"])
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": f"Answer using context only:\n{ctx}\n\nQ: {q}"}],
            temperature=0.0,
            max_tokens=600,
        )
        ex["answer"] = resp.choices[0].message.content or ""

    return dataset
