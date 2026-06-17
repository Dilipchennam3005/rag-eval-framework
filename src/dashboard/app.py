"""
Streamlit dashboard for the RAG Evaluation Framework.

Run with:
    streamlit run src/dashboard/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.embedding.bm25_index import BM25Index
from src.retrieval import HybridRetriever, VectorRetriever, CrossEncoderReranker
from src.generation import Generator


@st.cache_resource
def load_config():
    return yaml.safe_load(Path("configs/config.yaml").read_text())


def _bm25_path(strategy: str) -> Path:
    cfg = load_config()
    return Path(cfg["data"]["bm25_index_path"].replace(".pkl", f"_{strategy}.pkl"))


@st.cache_resource
def load_retriever(strategy: str, retriever_type: str):
    cfg = load_config()
    embedder = Embedder(**cfg["embedding"])
    vector_store = VectorStore(
        index_name=cfg["pinecone"]["index_name"],
        namespace=f"sec_{strategy}",
        dimension=cfg["pinecone"]["dimension"],
    )

    if retriever_type == "Hybrid":
        bm25 = BM25Index(str(_bm25_path(strategy)))
        if bm25.load():
            return HybridRetriever(embedder, vector_store, bm25, **cfg["retrieval"]["hybrid"])

    return VectorRetriever(embedder, vector_store)


@st.cache_resource
def load_generator(prompt_version: str):
    cfg = load_config()
    return Generator(model=cfg["generation"]["model"], prompt_version=prompt_version)


def load_experiment_results() -> list[dict]:
    results = []
    for path in sorted(Path("experiments").glob("*_results.json")):
        data = json.loads(path.read_text())
        results.append({
            "run": path.stem.replace("_results", ""),
            **data.get("scores", {}),
            "num_questions": data.get("num_questions", 0),
        })
    return results


st.set_page_config(page_title="RAG Eval Framework", layout="wide")
st.title("RAG Evaluation & Observability Framework")
st.caption("SEC 10-K filings · AAPL · MSFT · JPM")

tab_query, tab_compare = st.tabs(["Query", "Experiment Results"])

# ── Query tab ──────────────────────────────────────────────────────────────
with tab_query:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        strategy = st.selectbox("Chunking strategy", ["fixed_size", "document_aware"])
    with col2:
        retriever_type = st.selectbox("Retriever", ["Hybrid", "Vector"])
    with col3:
        prompt_version = st.selectbox("Prompt version", ["v1", "v2"])
    with col4:
        use_reranker = st.checkbox("Cross-encoder rerank", value=False)

    if retriever_type == "Hybrid" and not _bm25_path(strategy).exists():
        st.info(
            "BM25 index not found — running in vector-only mode. "
            "Run `python main.py` locally to build the index and enable hybrid retrieval."
        )

    question = st.text_input(
        "Ask a question about AAPL, MSFT, or JPM 10-K filings",
        placeholder="What was Apple's revenue in 2024?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and generating..."):
            try:
                retriever = load_retriever(strategy, retriever_type)
                generator = load_generator(prompt_version)

                cfg = load_config()
                top_k = cfg["retrieval"]["top_k"]
                fetch_k = top_k * 3 if use_reranker else top_k
                chunks = retriever.retrieve(question, top_k=fetch_k)
                if use_reranker:
                    reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model"])
                    chunks = reranker.rerank(question, chunks, top_k=top_k)
                result = generator.generate(question, chunks)

                st.subheader("Answer")
                st.write(result["answer"])

                with st.expander(f"Retrieved chunks ({len(chunks)})"):
                    for i, chunk in enumerate(chunks, 1):
                        meta = chunk.get("metadata", {})
                        st.markdown(
                            f"**[{i}] {meta.get('ticker', '?')} · "
                            f"{meta.get('filing_date', '?')} · "
                            f"{meta.get('section_name', '?')}** "
                            f"(score: {chunk.get('score', 0):.3f})"
                        )
                        st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total tokens", f"{result['total_tokens']:,}")
                col_b.metric("Cost", f"${result['cost_usd']:.5f}")
                col_c.metric("Model", result["model"])

                with st.expander("Token breakdown"):
                    st.json({
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["total_tokens"],
                        "cost_usd": result["cost_usd"],
                    })

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the index has been built by running `python main.py` first.")

# ── Experiment results tab ─────────────────────────────────────────────────
with tab_compare:
    st.subheader("Experiment Comparison")

    exp_results = load_experiment_results()
    if not exp_results:
        st.info("No experiment results yet. Run `python main.py` to generate them.")
    else:
        df = pd.DataFrame(exp_results)
        ragas_cols = [c for c in df.columns if c in (
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        )]
        cost_cols = [c for c in df.columns if "cost" in c or "token" in c]
        display_cols = ["run"] + ragas_cols + cost_cols + ["num_questions"]
        display_df = df[[c for c in display_cols if c in df.columns]]

        st.dataframe(
            display_df.style.highlight_max(subset=ragas_cols, color="#d4edda")
                            .highlight_min(subset=[c for c in cost_cols if c in display_df.columns], color="#d4edda"),
            use_container_width=True,
        )

        if ragas_cols:
            st.subheader("RAGAS Scores by Run")
            st.bar_chart(df.set_index("run")[ragas_cols])

        if cost_cols:
            st.subheader("Cost by Run")
            cost_chart_cols = [c for c in cost_cols if c in df.columns]
            if cost_chart_cols:
                st.bar_chart(df.set_index("run")[cost_chart_cols[:1]])
