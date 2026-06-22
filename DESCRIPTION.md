# RAG Evaluation & Observability Framework — Full Project Description

## What We Built

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions about SEC 10-K annual filings for Apple (AAPL), Microsoft (MSFT), and JPMorgan Chase (JPM). The system covers every layer of the RAG stack — from document ingestion through generation, evaluation, observability, and cloud deployment — and includes a 6-way ablation study backed by real RAGAS evaluation scores.

The live application is deployed at: https://rag-eval-framework-clc6ju46dq-uc.a.run.app

---

## Purpose

The purpose of this project was to build a rigorous, evaluable RAG system — not just something that produces answers, but one where you can measure *how well* it produces answers and *why* certain design choices lead to better outcomes. Financial filings were chosen as the domain because they are dense, structured, and contain a mix of narrative prose and precise numerical data (tables, dollar figures, percentages), which stress-tests every part of a RAG pipeline in realistic ways.

The secondary purpose was to treat deployment as a first-class concern — the system runs on GCP Cloud Run with CI/CD via GitHub Actions, meaning any code change automatically triggers a build and deploys the updated application without manual intervention.

---

## How We Built It — Layer by Layer

### 1. Ingestion

We downloaded 10-K filings directly from the SEC EDGAR API for the three companies, fetching the two most recent filings per company. The filings arrive as large HTML documents with inline CSS, nested tables, and section headers.

The core challenge was extracting clean, structured text from financial HTML. `BeautifulSoup.get_text()` on financial tables produces orphaned numbers — a table with rows like `Revenue | 387,537 | 365,817` becomes `387,537\n365,817` with no column labels. We wrote a custom `_render_table()` function that converts each `<tr>` into a pipe-delimited row before text extraction, preserving `Label | value1 | value2` structure that the LLM can reason over correctly.

Parsed sections are stored in a SQLite registry (`data/registry.db`) keyed by accession number and section name. This means re-running the pipeline skips re-downloading and re-parsing filings that have already been processed.

### 2. Chunking

Two chunking strategies were implemented and compared:

**FixedSizeChunker** splits text into chunks of 2000 characters with a 200-character overlap. It is simple, predictable, and produces uniform chunks — but it splits mid-sentence and mid-paragraph, which can break the semantic coherence of a passage.

**DocumentAwareChunker** respects paragraph boundaries. It accumulates paragraphs up to a maximum of 3000 characters and never splits a paragraph unless it exceeds the minimum size (200 characters) on its own. This preserves the semantic unit of each passage.

These two strategies produced 1309 and 996 chunks respectively, and their effects on answer quality were directly measurable in RAGAS scores.

### 3. Embedding

We used OpenAI's `text-embedding-3-small` model to embed all chunks into 1536-dimensional dense vectors. Embedding is done in batches of 100 with exponential retry on rate limit errors. Each batch takes roughly 1-2 seconds, so the full corpus of ~2300 chunks embeds in around 2 minutes.

We chose `text-embedding-3-small` over `text-embedding-3-large` because it is roughly 5x cheaper with only marginal quality difference on retrieval tasks — the reranker downstream compensates for any precision loss at the retrieval stage.

### 4. Sparse Encoding (TF-IDF)

To complement dense semantic retrieval, we built a `SparseEncoder` class wrapping sklearn's `TfidfVectorizer` with 30,000 features, sublinear TF scaling, and a word-level tokenizer. After fitting on the full corpus, it produces sparse vectors in Pinecone's required format: `{"indices": [...], "values": [...]}`.

The vocabulary is saved as a small pkl file (~280-300 KB per strategy) that ships inside the Docker image. This is critical for the deployment model — the sparse encoder needs to encode queries at runtime, so the vocabulary must be available in the container.

### 5. Vector Store (Pinecone)

All dense and sparse vectors are stored in a single Pinecone serverless index (`rag-eval-hybrid`) using the **dotproduct** similarity metric. Each vector upserted to Pinecone contains both `values` (dense) and `sparse_values` (TF-IDF sparse), and Pinecone combines both signals in a single query call server-side.

We use separate namespaces within the same index for each chunking strategy (`sec_fixed_size`, `sec_document_aware`), which lets both strategies coexist in the same index without interference.

**Why Pinecone over ChromaDB or FAISS:** ChromaDB and FAISS store the index on disk. This means the Docker image must bundle gigabytes of vector data, and the container becomes stateful — you cannot scale to zero or spin up multiple instances without managing shared storage. Pinecone is cloud-hosted; the container stays completely stateless, scales to zero between requests, and any new instance connects to the same index without any data transfer. The trade-off is ~30ms of network latency per query versus a microsecond local lookup, which is irrelevant at our scale.

**Why dotproduct over cosine:** Pinecone's native sparse-dense hybrid querying requires the dotproduct metric. Cosine metric normalizes vectors before computing similarity, which strips the magnitude information that the sparse TF-IDF scores rely on. Dotproduct preserves the raw signal from both dense and sparse components.

### 6. Retrieval

Three retrieval modes were implemented:

**VectorRetriever** sends only the dense query embedding to Pinecone. It is the fallback when the sparse encoder vocabulary is not available.

**HybridRetriever** sends both a dense embedding and a TF-IDF sparse vector in a single Pinecone query. The sparse vector values are scaled by `sparse_weight=0.3` and `_SPARSE_SCALE=3.0` to balance the magnitude of sparse TF-IDF scores against dense dot-product scores. Pinecone combines them server-side and returns a single ranked list — no local re-ranking or fusion step needed.

The earlier implementation used local BM25 (via the `rank-bm25` library) combined with Reciprocal Rank Fusion, running entirely in memory. This was replaced with Pinecone native hybrid for two reasons: (1) the BM25 pkl files were large enough (~5 MB each) to cause silent failures in the GitHub Actions build context, and (2) Pinecone native hybrid eliminates the two-step retrieval and fusion entirely, reducing latency and code complexity.

**CrossEncoderReranker** uses `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers to rerank a candidate pool of 15 chunks down to the top 5. Cross-encoders jointly attend to the query and each passage simultaneously, producing much more accurate relevance scores than bi-encoders, at the cost of being O(n) rather than O(1) at query time. It is available locally but disabled in the Cloud Run deployment because sentence-transformers pulls in PyTorch (~2 GB), which makes the Docker image impractically large.

### 7. Generation

The generator uses `gpt-4o-mini` to produce answers from the retrieved context. Two prompt versions were implemented and compared:

**v1** gives the model the retrieved passages and asks for a concise, direct answer.

**v2** adds an instruction to cite specific passages and indicate uncertainty when the context is insufficient.

The generator tracks cost per query by computing `prompt_tokens * $0.15/M + completion_tokens * $0.60/M` from the OpenAI response usage object.

Answers are streamed token by token using OpenAI's `stream=True` with `stream_options={"include_usage": True}`. The `stream_options` flag is needed because usage data (token counts, cost) only arrives in the final streaming chunk — without it, we would have to choose between streaming and cost tracking. A mutable `usage_out` dict is passed by reference into the generator's `stream()` method and populated when the final chunk arrives, so the caller can display cost metrics after the stream completes.

### 8. Evaluation

We used **RAGAS** (Retrieval-Augmented Generation Assessment) to evaluate 8 questions across the three companies. RAGAS computes four metrics:

- **Faithfulness**: Does the answer contain only claims that can be verified from the retrieved context? Measures hallucination.
- **Answer Relevancy**: How directly does the answer address the question?
- **Context Precision**: Are the retrieved chunks actually relevant to the question?
- **Context Recall**: Do the retrieved chunks cover all the information needed to answer?

Each metric is computed using the LLM itself as a judge, comparing the generated answer against both the retrieved context and a reference ground truth answer. The full evaluation suite of 8 questions costs less than half a cent to run.

### 9. Observability

Four observability tools were integrated:

**MLflow** tracks experiment parameters (strategy, prompt version, top\_k, model, reranker) and metrics (all RAGAS scores, cost, token counts) in a local SQLite database. Each run is queryable and comparable from the MLflow UI.

**Weights & Biases** tracks the same metrics remotely, provides a hosted experiment comparison dashboard, and stores per-question result tables for drill-down analysis.

**LangSmith** traces every OpenAI API call by wrapping the client with `wrap_openai()`. Each generation call appears as a traced run with the full prompt, response, latency, and cost visible in the LangSmith UI.

**Arize Phoenix** collects OpenTelemetry spans from OpenAI calls, providing a local observability UI for inspecting LLM traces during development.

### 10. Dashboard

A Streamlit dashboard provides a query interface and experiment comparison view. Users can select the chunking strategy, retriever type, prompt version, and (locally) enable the cross-encoder reranker. Answers stream progressively as tokens arrive rather than appearing all at once, with token counts and cost displayed after the stream completes. The experiment comparison tab renders a table of all RAGAS results with green highlights on best-performing values per metric, plus bar charts for visual comparison.

### 11. Deployment

The application runs on **GCP Cloud Run** with 2 GB memory and auto-scaling between 0 and 3 instances. When no requests are in flight the service scales to zero, incurring no cost.

The Docker image is built and deployed automatically via **GitHub Actions** on every push to main. The CI/CD pipeline uses **Workload Identity Federation** instead of service account keys. WIF exchanges short-lived GitHub OIDC tokens for GCP credentials at deploy time, meaning no key files are stored anywhere — not in the repo, not in GitHub Secrets (beyond the WIF provider resource name). This was required because the GCP organization policy blocked service account key creation.

The container is completely stateless: all vector data lives in Pinecone, the sparse encoder vocabularies are small enough (~280 KB each) to ship inside the image, and there are no local files written at runtime.

---

## Key Design Decisions and Trade-offs

### Pinecone native sparse-dense over BM25 + RRF

The initial hybrid implementation used local BM25 with Reciprocal Rank Fusion. BM25 pkl files were stored in the repo and copied into the Docker image. In practice, the files (~5 MB each) were silently excluded from the GitHub Actions build context — the Docker layer was only 525 KB, meaning the pkl files never made it into the image, and the app silently fell back to vector-only retrieval in production. Pinecone native hybrid eliminates the local file dependency entirely: the TF-IDF vocabulary is the only thing that needs to travel into the container, and at ~280 KB it is small enough to commit without issue.

### Stateless container design

Every design decision around persistence asked the same question: does this need to live in the container, or can it live in a cloud service? Dense vectors → Pinecone. Experiment logs → MLflow/W&B. The only things in the container are code, the prompt templates, the experiment result JSONs (static, read-only), and the TF-IDF vocabularies. This makes the container reproducible, horizontally scalable, and safe to restart at any time.

### DocumentAware chunking for faithfulness, FixedSize for recall

Paragraph-aware chunking prevents the model from receiving half a sentence as context, which is a direct cause of hallucination. On our dataset, DocumentAware chunking with the new Pinecone hybrid index achieved faithfulness=0.938 vs. 1.000 for FixedSize — a reversal from the old BM25+RRF baseline where DocumentAware led. The result depends on the retrieval method, not just the chunking strategy. FixedSize chunks are smaller and denser, so they match a wider variety of queries (higher recall), while DocumentAware chunks are semantically coherent units that anchor the LLM to a single idea (higher precision under certain retrievers).

### Lazy import for sentence-transformers

The CrossEncoderReranker import of `sentence_transformers` was originally at the top of the module. On Cloud Run, where sentence-transformers is not installed, this crashed the entire application at startup — not when the reranker was used, but when the module was first imported. Moving the import inside `__init__()` means the module loads cleanly and only raises an error if someone actually instantiates the reranker. The dashboard detects availability at startup with a try/except and disables the checkbox with a tooltip explanation if sentence-transformers is not installed.

### Prompt versioning

Prompt templates are stored as separate files (`prompts/v1.txt`, `prompts/v2.txt`) rather than hardcoded strings. The `PromptManager` class loads the correct template at construction time and can be hot-swapped during a session. This made it straightforward to run the ablation study across prompt versions without changing any pipeline code, and to add new prompt versions without touching the generator.

### gpt-4o-mini over gpt-4o

At ~$0.0004 per query, gpt-4o-mini makes the evaluation suite cost less than half a cent for 8 questions. gpt-4o would cost roughly 10x more per query with marginal improvement on structured financial Q&A, where the task is primarily reading comprehension over retrieved text rather than complex reasoning.

---

## Ablation Results Summary

| Run | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-----|:-----------:|:----------------:|:-----------------:|:--------------:|
| fixed\_size · v1 (hybrid) | **1.000** | **0.998** | 0.746 | **0.594** |
| fixed\_size · v2 (hybrid) | 0.895 | **0.996** | 0.782 | **0.594** |
| document\_aware · v1 (hybrid) | 0.938 | 0.996 | **0.847** | **0.594** |
| document\_aware · v2 (hybrid) | 0.854 | 0.992 | **0.877** | 0.531 |
| fixed\_size · v1 · rerank | 0.844 | 0.871 | 0.745 | 0.500 |
| document\_aware · v1 · rerank | 0.875 | **0.998** | 0.695 | **0.625** |

The Pinecone native hybrid index (dotproduct, sparse+dense) meaningfully improves both faithfulness and context recall compared to the earlier BM25+RRF implementation. The reranker improves recall on document\_aware chunks but trades off faithfulness on fixed\_size chunks, where the smaller chunk size produces a noisier top-5 after cross-encoder scoring.

---

## Lessons Learned

**Binary files in CI/CD build contexts have size limits.** Git-committed pkl files that work locally can silently disappear in a GitHub Actions Docker build if the build context transfer exceeds limits. Always verify what actually made it into the image rather than assuming the COPY succeeded.

**Streaming and usage tracking can coexist.** OpenAI's `stream_options={"include_usage": True}` sends token counts in the final streaming chunk, making it possible to display cost after the stream completes without making a separate non-streaming call.

**GCP API enablement is order-dependent.** Workload Identity Federation requires both `iam.googleapis.com` and `iamcredentials.googleapis.com` to be enabled. The WIF setup documentation only surfaces the first. The second fails silently until the first actual credential exchange attempt during deployment.

**RAGAS recall and precision trade in opposite directions across chunking strategies.** Smaller chunks improve recall because they match more diverse queries. Larger chunks improve precision because each chunk is a coherent semantic unit. The right choice depends on whether the use case prioritizes breadth of coverage or citation quality.

**Reranking improves results on coherent chunks, not noisy ones.** The cross-encoder reranker works best when the candidate pool contains passage-length coherent text. With fixed-size chunks that may cut mid-sentence, the reranker has less signal to work with and can surface plausible-looking but misleading passages.
