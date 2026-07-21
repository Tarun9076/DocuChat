# DocuChat

DocuChat is a Retrieval-Augmented Generation (RAG) web app that lets you upload a document (`.txt` or `.pdf`) and ask questions about it in a ChatGPT-style chat interface. It chunks and embeds the document with HuggingFace embeddings, stores them in Chroma, and generates grounded answers using Azure OpenAI — all served through a Flask backend.

This repo also includes a set of numbered standalone scripts that walk through core RAG concepts step by step (ingestion, retrieval, chunking strategies, multi-query retrieval, hybrid search, reranking) — useful if you want to learn how each piece works in isolation before/instead of using the full webapp.

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with a chat model deployed (e.g. `gpt-4o-mini`, `gpt-4o`) — see [Setting up Azure OpenAI](#setting-up-azure-openai) below

## Setup

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>
   cd rag-for-beginners
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example file and fill in your own values:

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
   - Note: reasoning/Codex-family models often only support the Responses API, not Chat Completions — stick to a standard GPT chat model for this app.
3. Grab the API key and endpoint from Azure Portal → your resource → **Keys and Endpoint**.

## Running the web app

```bash
cd webapp
python app.py
```

Open `http://127.0.0.1:5000`, upload a `.txt` or `.pdf` file, and start asking questions about it.

## Running the standalone learning scripts

Each numbered script is self-contained and can be run directly, e.g.:

```bash
python 1_ingestion_pipeline.py
```

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

Sample source documents used by these scripts live in [`docs/`](docs/).

## Project structure

```
.
├── webapp/                 # DocuChat web app (Flask backend + chat UI)
│   ├── app.py
│   ├── templates/
│   └── static/
├── docs/                    # Sample documents used by the numbered scripts
├── 1_ingestion_pipeline.py  # Numbered learning scripts (see table above)
├── ...
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- `.env` is git-ignored — never commit real API keys. Use `.env.example` as the template.
- The webapp keeps each visitor's uploaded document and chat history in memory, scoped to their browser session — documents aren't persisted to disk or shared between users.
