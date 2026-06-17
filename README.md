# RAG Evaluation & Observability Framework

A production-style RAG evaluation system over SEC 10-K filings — implementing
two chunking strategies, hybrid retrieval, RAGAS-based scoring, and full
experiment tracking via MLflow, Weights & Biases, and Arize Phoenix.

## Stack
Python 3.11 · RAGAS · MLflow · Weights & Biases · ChromaDB · OpenAI API · Streamlit

## Project Structure
```
rag-eval-framework/
├── src/
│   ├── ingestion/          # SEC EDGAR downloader, HTML parser, SQLite registry
│   ├── chunking/           # Fixed-size vs document-aware chunking strategies
│   ├── embedding/          # OpenAI embedder, ChromaDB vector store, BM25 index
│   ├── retrieval/          # Vector-only and hybrid (RRF) retrievers
│   ├── generation/         # GPT-4o-mini generator with prompt versioning
│   ├── evaluation/         # RAGAS evaluation pipeline + test questions
│   ├── observability/      # MLflow + W&B experiment tracking
│   └── dashboard/          # Streamlit UI
├── data/
│   ├── raw/                # Downloaded SEC filings (gitignored)
│   ├── chromadb/           # Vector store (gitignored)
│   └── registry.db         # SQLite document registry (gitignored)
├── configs/config.yaml
├── prompts/
│   ├── v1_basic.txt
│   └── v2_with_citations.txt
└── main.py
```

## Setup
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:
```
OPENAI_API_KEY=sk-proj-...
WANDB_API_KEY=...
```

## Usage

**Run experiments (both chunking strategies):**
```bash
python main.py
```

**Run a single strategy:**
```bash
python main.py --strategy fixed_size --prompt v2
```

**Launch Streamlit dashboard:**
```bash
streamlit run src/dashboard/app.py
```

**View MLflow UI:**
```bash
mlflow ui
```

## Progress
- [x] Layer 1 — Data ingestion (AAPL, MSFT, JPM 10-K filings)
- [x] Layer 2 — Chunking (fixed-size vs document-aware)
- [x] Layer 3 — Embedding (OpenAI text-embedding-3-small + ChromaDB + BM25)
- [x] Layer 4 — Retrieval (vector-only and hybrid RRF)
- [x] Layer 5 — Generation (GPT-4o-mini + prompt versioning v1/v2)
- [x] Layer 6 — Evaluation (RAGAS: faithfulness, relevancy, precision, recall)
- [x] Layer 7 — Observability (MLflow + W&B)
- [x] Streamlit dashboard (query UI + experiment comparison)
- [ ] Docker + GCP Cloud Run
