# DocuChat

DocuChat is a Retrieval-Augmented Generation (RAG) web app: upload a document (`.txt` or `.pdf`) and ask questions about it in a ChatGPT-style chat interface. It chunks and embeds the document with HuggingFace embeddings, stores them in an in-memory Chroma vector store, and generates grounded, streamed answers using Azure OpenAI — served through a Flask backend.

Retrieval quality isn't just claimed — it's measured. [`eval/eval_retrieval.py`](eval/eval_retrieval.py) is a 40-question benchmark (built from the actual sample corpus) that compares naive vector search against hybrid BM25+vector search, both at the retrieval level and end-to-end through real generated answers. See [Retrieval evaluation](#retrieval-evaluation) below.

The repo also ships [`learning/`](learning/) — a set of numbered, standalone scripts that walk through core RAG concepts step by step (ingestion, retrieval, chunking strategies, multi-query retrieval, hybrid search, reranking). Useful if you want to see how each technique works in isolation before/instead of using the full webapp.

## Architecture

```
Browser  <-- stream -->  Flask (app/routes.py)
                            |
                            +-- app/rag.py          load + split uploaded doc
                            +-- Chroma (in-memory)  per-session vector store
                            +-- app/session_store.py  per-session chat history (in-memory)
                            +-- Azure OpenAI          answer generation (streamed)
```

Each browser session (a signed cookie, no login) gets its own isolated Chroma collection and chat history — uploads and conversations never leak across users. See [Production notes](#production-notes) for the scaling trade-off this implies.

## Project structure

```
.
├── app/                     # Flask app (the product)
│   ├── __init__.py          # create_app() factory
│   ├── config.py            # env-driven configuration
│   ├── rag.py                # document loading/splitting/vectorstore helpers
│   ├── session_store.py      # per-session state (in-memory)
│   ├── routes.py             # / , /api/status, /api/upload, /api/chat, /api/reset
│   ├── static/, templates/   # chat UI
│   └── uploads/               # scratch space for uploads (deleted after processing)
├── run.py                   # local dev entrypoint
├── wsgi.py                  # production entrypoint (gunicorn/waitress)
├── eval/eval_retrieval.py   # retrieval + end-to-end accuracy benchmark
├── learning/                 # 13 numbered scripts/notebooks teaching RAG concepts
├── data/sample_docs/          # sample corpus (5 company profiles) used by eval/ and learning/
├── tests/                    # pytest suite (no Azure calls required)
├── Dockerfile
└── .github/workflows/ci.yml  # lint + test on push/PR
```

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with a chat model deployed (e.g. `gpt-4o-mini`, `gpt-4o`) — see [Setting up Azure OpenAI](#setting-up-azure-openai)

## Setup

1. **Clone and enter the repo, create a virtual environment**

   ```bash
   git clone <your-repo-url>
   cd DocuChat
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   | Variable | Description |
   |---|---|
   | `AZURE_OPENAI_API_KEY` | From Azure Portal → your Azure OpenAI resource → Keys and Endpoint |
   | `AZURE_OPENAI_ENDPOINT` | The bare resource endpoint, e.g. `https://<your-resource-name>.openai.azure.com/` (**not** the Azure AI Foundry project endpoint — they're different and not interchangeable) |
   | `AZURE_OPENAI_API_VERSION` | Defaults to `2024-05-01-preview`; only change if your deployment needs a different version |
   | `AZURE_OPENAI_DEPLOYMENT_NAME` | The deployment name you gave your chat model in Azure AI Foundry → Deployments (can differ from the underlying model name) |
   | `FLASK_SECRET_KEY` | Any random string, used to sign session cookies. Generate one with `python -c "import secrets; print(secrets.token_hex(32))"` |

### Setting up Azure OpenAI

1. Create an Azure OpenAI resource in the [Azure Portal](https://portal.azure.com).
2. In [Azure AI Foundry](https://ai.azure.com), open your resource's project → **Deployments** → deploy a chat-completion-capable model (e.g. `gpt-4o-mini`).
   - Reasoning/Codex-family models often only support the Responses API, not Chat Completions — stick to a standard GPT chat model.
3. Grab the API key and endpoint from Azure Portal → your resource → **Keys and Endpoint**.

## Running the app

**Local (dev):**

```bash
python run.py
```

Open `http://127.0.0.1:5000`, upload a `.txt` or `.pdf` file (try one from `data/sample_docs/`), and start asking questions.

**Local (production-like WSGI server):**

```bash
# Linux/macOS
gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 wsgi:app
# Windows
waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
```

**Docker:**

```bash
docker build -t docuchat .
docker run --env-file .env -p 5000:5000 docuchat
```

## Retrieval evaluation

[`eval/eval_retrieval.py`](eval/eval_retrieval.py) grades retrieval and generation against a 40-question, fact-grounded test set built from `data/sample_docs/` (e.g. "Who founded Nvidia and when?" → the retrieved/generated text must contain "Jensen Huang" and "1993").

```bash
python learning/1_ingestion_pipeline.py     # builds data/chroma_db from data/sample_docs/, once
python eval/eval_retrieval.py               # retrieval-only, free (no LLM calls)
python eval/eval_retrieval.py --end-to-end  # also grades real generated answers (uses Azure OpenAI)
```

Measured results on this corpus:

| Metric | Naive vector search | Hybrid (BM25 + vector) |
|---|---|---|
| Retrieval Hit Rate@5 | 67.5% | **80.0%** |
| Retrieval MRR@5 | 0.537 | **0.672** |
| End-to-end answer accuracy | 70.0% | **80.0%** |

The webapp itself currently uses naive vector search (`db.as_retriever(search_kwargs={"k": 5})` in `app/routes.py`) — the hybrid strategy is validated here and is the natural next upgrade for `app/rag.py`.

## Running the learning scripts

Each numbered script under `learning/` is self-contained and runnable from the repo root, e.g.:

```bash
python learning/1_ingestion_pipeline.py
```

Some notebooks/scripts need extra dependencies not required by the webapp — see [`requirements-learning.txt`](requirements-learning.txt).

| Script | Topic |
|---|---|
| `1_ingestion_pipeline.py` | Loading, chunking, embedding, and storing documents in Chroma |
| `2_retrieval_pipeline.py` | Querying the vector store for relevant chunks |
| `3_answer_generation.py` | Generating answers from retrieved context with Azure OpenAI |
| `4_history_aware_generation.py` | Conversational RAG with chat history |
| `5_recursive_character_text_spliiter.py` | Recursive character text splitting |
| `6_semantic_chunking.py` | Semantic chunking |
| `7_agentic_chunking.py` | Agentic (LLM-guided) chunking |
| `8_multi_modal_rag.ipynb` | Multi-modal RAG (text + images) |
| `9_retrieval_methods.py` | Comparing different retrieval strategies |
| `10_multi_query_retrieval.py` | Multi-query retrieval |
| `11_reciprocal_rank_fusion.py` | Reciprocal Rank Fusion |
| `12_hybrid_search.ipynb` | Hybrid search (keyword + vector) |
| `13_reranker.ipynb` | Reranking retrieved results |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

`tests/test_routes.py` exercises the full upload → chat → reset flow through Flask's test client with a fake LLM injected via `create_app(llm=..., embedding_model=...)` — no Azure credentials or network calls needed, so it runs the same locally and in CI (`.github/workflows/ci.yml`).

## Production notes

This is a portfolio-scale deployment, and it's honest about where that shows:

- **Session state is in-memory** (`app/session_store.py`): uploaded documents and chat history live in a process-local dict keyed by session cookie. This is why the Docker image and the gunicorn command both run **1 worker** — splitting requests across multiple processes would make sessions "disappear" depending on which worker handled a given request. Threads still provide concurrency within that worker. Scaling beyond one worker would mean moving session + vector-store state to a shared backend (e.g. Redis + a hosted vector DB) instead.
- **Uploaded files are session-scoped and ephemeral**: saved to `app/uploads/` only long enough to chunk and embed, then deleted; nothing is persisted to disk after that.
- **Retrieval is currently naive vector search** in the live app, even though the hybrid strategy measurably outperforms it (see above) — swapping `app/rag.py`'s retriever is the natural next step, not yet wired into the webapp.

## Notes

- `.env` is git-ignored — never commit real API keys. Use `.env.example` as the template.
- `data/chroma_db/` (learning scripts' persisted vector store) and `app/uploads/` are git-ignored.
