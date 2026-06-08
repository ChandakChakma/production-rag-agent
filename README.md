# RAG.Agent — Production RAG + Agent System

> A MAANG-level AI engineering project demonstrating production-grade Retrieval-Augmented Generation, ReAct Agents, multi-dimensional evaluation pipelines, and a full observability stack — all behind a polished streaming UI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / API GATEWAY                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  FastAPI + Middleware
                    ┌────────────▼────────────--┐
                    │     REQUEST PIPELINE      │
                    │  Auth | RateLimit | Trace │
                    └────────────┬────────────--┘
                                 │
          ┌──────────────────────▼─────────────────────┐
          │              RAG AGENT (ReAct)             │
          │   ┌─────────────┐    ┌──────────────────┐  │
          │   │   Reasoner  │◄──►│   Tool Executor  │  │
          │   │  (LLM + CoT)│    │ RAG|Calc|Search  │  │
          │   └─────────────┘    └──────────────────┘  │
          │   ┌─────────────┐    ┌──────────────────┐  │
          │   │  Convo Mem  │    │  History Window  │  │
          │   │ (compressed)│    │   (windowed)     │  │
          └───┴─────┬───────┴────┴──────────────────┘--│
                    │
          ┌─────────▼───────────────────────────────────┐
          │           TWO-STAGE RETRIEVAL               │
          │                                             │
          │  Stage 1: Hybrid Retrieval (Top-K = 50)     │
          │  ┌──────────────┐   ┌──────────────────┐    │
          │  │  Dense (ANN) │   │  Sparse (BM25)   │    │
          │  │  ChromaDB /  │   │  rank-bm25 +     │    │
          │  │  Pinecone    │   │  TF-IDF weights  │    │
          │  └──────┬───────┘   └────────┬─────────┘    │
          │         └──────────┬─────────┘              │
          │               RRF Fusion (k=60)             │
          │                                             │
          │  Stage 2: Cross-Encoder Reranking (Top-K=5) │
          │  ┌────────────────────────────────────────┐ │
          │  │  ms-marco-MiniLM-L-6-v2                │ │
          │  └────────────────────────────────────────┘ │
          └─────────────────────────────────────────────┘
                    │
          ┌─────────▼───────────────────────────────────┐
          │              EVALUATION PIPELINE            │
          │  ┌──────────────┐   ┌─────────────────────┐ │
          │  │  RAGAS Suite │   │   LLM-as-Judge      │ │
          │  │  Faithfulness│   │   Correctness       │ │
          │  │  Relevancy   │   │   Groundedness      │ │
          │  │  Ctx Recall  │   │   Hallucination Rate│ │
          │  └──────────────┘   └─────────────────────┘ │
          └─────────────────────────────────────────────┘
                    │
          ┌─────────▼───────────────────────────────────┐
          │              OBSERVABILITY STACK            │
          │  OpenTelemetry Traces │ Prometheus Metrics  │
          │  Structured Logs      │ Cost Tracker        │
          │  Latency P50/P95/P99  │ Token Counter       │
          └─────────────────────────────────────────────┘
```

---

## Key Features

### Retrieval
- **Hybrid search** — dense (OpenAI embeddings) + sparse (BM25) fused with **Reciprocal Rank Fusion** (k=60)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` pushes MRR@10 from 0.72 → 0.85
- **Semantic chunking** — context-aware splitting with no mid-sentence breaks
- **Embedding cache** — DiskCache avoids recomputing embeddings (~70% API cost savings)
- **Embedding drift detection** — alerts when corpus distribution shifts over time

### Agent
- **ReAct architecture** — Reason → Act → Observe → Repeat with full trace logging
- **Tool suite** — RAG retrieval, web search, calculator, document summarizer
- **Conversation memory** — sliding window + LLM-based compression for long sessions
- **Structured output parsing** — Pydantic-validated tool calls with error recovery

### Evaluation
- **RAGAS metrics** — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **LLM-as-Judge** — GPT-4 scoring for correctness, groundedness, conciseness
- **Hallucination detection** — claim-level verification against retrieved context
- **Editable test dataset** — add, edit, or remove cases directly in the UI; paste your own Q&A pairs and run evals instantly
- **Regression testing** — eval suite runs on every ingestion batch

### Streaming UI
- **Live pipeline visualization** — each stage (cache check → retrieval → reranking → generation) animates in real time with colour-coded status dots
- **Answer-first layout** — generated answer streams at the top; retrieved sources render below
- **SSE streaming** — token-by-token display with blinking cursor; all pipeline steps correctly resolve on completion
- **Batch fallback** — one-click switch between streaming SSE and batch response modes

### Monitoring
- **OpenTelemetry** distributed tracing with OTLP export to Jaeger
- **Prometheus** metrics — latency histograms, error rates, cache hit rates
- **Structured logging** — JSON logs with trace correlation IDs
- **Cost tracking** — per-query token cost with budget alerts
- **Circuit breaker** — prevents cascade failures on LLM API errors

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | `text-embedding-3-small` |
| Vector DB | ChromaDB (local) / Pinecone (cloud) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent Framework | Custom ReAct (LangChain-compatible) |
| Evaluation | RAGAS + custom LLM-Judge |
| API | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Monitoring | OpenTelemetry + Prometheus + Grafana |
| Tracing | Jaeger via OTEL Collector |
| Logging | structlog (JSON) |
| Caching | DiskCache + in-memory LRU |
| Config | Pydantic Settings v2 |
| Testing | pytest + pytest-asyncio |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ChandakChakma/production-rag-agent-system.git
cd production-rag-agent-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY
```

### 3. Ingest sample data

```bash
python scripts/ingest_sample_data.py
```

### 4. Start the API

```bash
make run
# or: uvicorn src.api.main:app --reload --port 8000
```

### 5. Open the UI

Navigate to `http://localhost:8000` — the full streaming interface loads automatically.

### 6. Query via cURL

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?", "use_agent": true}'
```

### 7. Run the evaluation suite

```bash
python scripts/run_eval_suite.py
```

---

## Project Structure

```
production-rag-agent-system/
├── src/
│   ├── config/          # Pydantic settings, env vars
│   ├── ingestion/       # Document loading, chunking, embedding
│   ├── retrieval/       # Vector store, hybrid retrieval, reranking
│   ├── agents/          # ReAct agent, tools, memory
│   ├── evaluation/      # RAGAS, LLM-Judge, custom metrics
│   ├── monitoring/      # OTEL tracing, Prometheus metrics
│   ├── api/             # FastAPI app, routes, middleware
│   └── utils/           # Logger, cache, helpers
├── tests/               # pytest test suite (retrieval, eval, agents)
├── scripts/             # CLI scripts for ingestion & evaluation
├── data/
│   ├── sample_docs/     # Sample documents for demo
│   ├── chroma_db/       # Persisted vector store (gitignored)
│   └── cache/           # Embedding & semantic cache (gitignored)
├── deploy/
│   ├── prometheus.yml   # Prometheus scrape config
│   └── otel-collector.yml
├── index.html           # Streaming UI (served by FastAPI)
├── docker-compose.yml   # Full observability stack
├── Makefile
├── requirements.txt
└── .env.example
```

---

## Docker — Full Observability Stack

```bash
docker-compose up -d
```

| Service | URL |
|---|---|
| API + UI | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Jaeger (traces) | http://localhost:16686 |
| Grafana | http://localhost:3000 |

---

## UI Walkthrough

### Query tab
Select **ReAct Agent** or **Direct RAG** mode, toggle between **Stream tokens** (SSE) and **Batch response**, enter a question and hit **Run Query**. The right panel shows:

1. Live pipeline steps with animated status dots (grey → blue active → green done / purple cached)
2. The generated answer streaming token by token
3. Retrieved sources with relevance scores below the answer

### Evaluate tab
Build a test dataset interactively — paste your own questions, answers, ground truths, and context passages. Add or remove cases freely, then click **Run Evaluation Suite** to get RAGAS + LLM-Judge metrics with animated score bars.

### Ingest tab
Paste any document text (or load a built-in sample), configure chunk size and overlap, and ingest into a named collection. The right panel shows chunk count, estimated token cost, and a content preview.

### Architecture tab
Visual request pipeline diagram and latency budget table — useful for system design discussions.

---

## API Reference

### `POST /api/v1/query`
```json
{
  "query": "string",
  "use_agent": true,
  "top_k": 5,
  "collection": "default",
  "stream": false
}
```

### `GET /api/v1/query/stream`
SSE endpoint — emits `status`, `sources`, `token`, `stats`, `done`, and `error` events.

### `POST /api/v1/ingest`
```json
{
  "documents": [{"content": "...", "metadata": {}}],
  "collection": "default",
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

### `POST /api/v1/evaluate`
```json
{
  "test_dataset": [
    {
      "question": "...",
      "answer": "...",
      "ground_truth": "...",
      "contexts": ["..."]
    }
  ],
  "collection": "default",
  "auto_generate_answers": false
}
```

### `GET /api/v1/stats` — Vector count, model, cache hit rate
### `GET /metrics` — Prometheus metrics endpoint
### `GET /health` — Health check

---

## Sample Evaluation Output

```
╔══════════════════════════════════════════════════╗
║         EVALUATION REPORT — 50 test cases        ║
╠══════════════════════════════════════════════════╣
║ faithfulness          │ 0.91 ████████████████░░  ║
║ answer_relevancy      │ 0.88 ███████████████░░░  ║
║ context_precision     │ 0.85 ██████████████░░░░  ║
║ context_recall        │ 0.82 █████████████░░░░░  ║
║ hallucination_rate    │ 0.04 ░░░░░░░░░░░░░░░░░░  ║
║ avg_latency_p95       │ 1.2s                     ║
║ avg_cost_per_query    │ $0.003                   ║
╚══════════════════════════════════════════════════╝
```

---

## System Design Highlights

These topics come up frequently in MAANG-level AI engineering interviews. Each one has a concrete implementation in this codebase.

| Topic | Implementation |
|---|---|
| Two-stage retrieval | ANN candidate generation → cross-encoder reranking (latency vs. precision tradeoff) |
| Embedding drift | Cosine similarity distribution monitoring with alert thresholds |
| Context window management | Dynamic context packing with per-query token budget |
| Hallucination mitigation | Claim extraction + source grounding verification |
| Cost optimisation | Semantic cache → embedding cache → model cascade (GPT-4o-mini first) |
| Scalability | Async I/O throughout, connection pooling, batch embedding |
| Fault tolerance | Circuit breaker, exponential backoff with jitter, graceful degradation |
| Eval as CI | Regression test suite runs automatically on every ingestion batch |
| Streaming UX | SSE pipeline with per-stage status events; answer renders before full retrieval completes |
| Cache coherence | SHA-256 keyed embedding cache; cosine-similarity keyed semantic cache (threshold 0.95) |

---

## Running Tests

```bash
# Full suite
pytest tests/ -v --cov=src --cov-report=term-missing

# Individual modules
pytest tests/test_retrieval.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_agents.py -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` or `pinecone` |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `RETRIEVAL_TOP_K` | `50` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks passed to LLM |
| `HYBRID_ALPHA` | `0.5` | 0 = pure BM25, 1 = pure dense |
| `AGENT_MAX_ITERATIONS` | `8` | ReAct loop limit |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Cosine similarity for cache hit |
| `EMBEDDING_CACHE_SIZE_GB` | `1` | Max disk cache size |
| `LOG_FORMAT` | `json` | `json` or `console` |
| `OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry collector |
| `EVAL_LLM_MODEL` | `gpt-4o` | Model used for LLM-Judge scoring |

See `.env.example` for the full list.

---

## Contributing

PRs are welcome. Please run `black`, `ruff`, and `mypy` before opening a pull request, and add tests for any new functionality.

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
pytest tests/ -v
```

---

## License

MIT — see `LICENSE`.
