"""
Main experiment runner.
Runs both chunking strategies through the full RAG pipeline
and logs results to MLflow and W&B.

Usage:
    python main.py
    python main.py --strategy fixed_size
    python main.py --strategy document_aware --prompt v2
"""
import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

from src.ingestion.document_registry import DocumentRegistry
from src.chunking import FixedSizeChunker, DocumentAwareChunker
from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.embedding.sparse_encoder import SparseEncoder
from src.retrieval import VectorRetriever, HybridRetriever, CrossEncoderReranker
from src.generation import Generator
from src.evaluation.evaluator import RAGASEvaluator, load_test_questions
from src.observability import MLflowTracker, WandbTracker, init_phoenix


def load_config() -> dict:
    return yaml.safe_load(Path("configs/config.yaml").read_text())


def build_index(cfg: dict, strategy: str) -> tuple:
    """Chunk all sections from the registry, embed, and index them."""
    registry = DocumentRegistry(cfg["data"]["registry_db"])
    sections = registry.get_all_sections()
    logger.info(f"Loaded {len(sections)} sections from registry")

    chunker = (
        FixedSizeChunker(**cfg["chunking"]["fixed_size"])
        if strategy == "fixed_size"
        else DocumentAwareChunker(**cfg["chunking"]["document_aware"])
    )
    chunks = chunker.chunk(sections)
    logger.info(f"Chunked into {len(chunks)} chunks using {strategy}")

    embedder = Embedder(**cfg["embedding"])
    vector_store = VectorStore(
        index_name=cfg["pinecone"]["index_name"],
        namespace=f"sec_{strategy}",
        dimension=cfg["pinecone"]["dimension"],
    )
    sparse_path = cfg["data"]["sparse_encoder_path"].replace(".pkl", f"_{strategy}.pkl")
    sparse_encoder = SparseEncoder(sparse_path)

    if vector_store.count() == len(chunks) and sparse_encoder.load():
        logger.info(f"Index already built ({vector_store.count()} vectors) — skipping embedding")
    else:
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
        sparse_encoder.fit(texts)
        sparse_vectors = sparse_encoder.encode_documents(texts)
        vector_store.add_chunks(chunks, embeddings, sparse_vectors)
        logger.info(f"Index built: {vector_store.count()} dense+sparse vectors")

    return embedder, vector_store, sparse_encoder


def run_experiment(strategy: str, prompt_version: str, cfg: dict, use_reranker: bool = False) -> dict:
    embedder, vector_store, sparse_encoder = build_index(cfg, strategy)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        sparse_encoder=sparse_encoder,
        sparse_weight=cfg["retrieval"]["hybrid"]["sparse_weight"],
    )
    reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model"]) if use_reranker else None

    gen_cfg = cfg["generation"].copy()
    gen_cfg["prompt_version"] = prompt_version
    generator = Generator(**gen_cfg)

    questions = load_test_questions()
    top_k = cfg["retrieval"]["top_k"]
    results = []
    for q in questions:
        # Retrieve more candidates when reranking so the reranker has room to work
        fetch_k = top_k * 3 if reranker else top_k
        chunks = retriever.retrieve(q["question"], top_k=fetch_k)
        if reranker:
            chunks = reranker.rerank(q["question"], chunks, top_k=top_k)
        gen_result = generator.generate(q["question"], chunks)
        gen_result["ground_truth"] = q["ground_truth"]
        gen_result["reranked"] = use_reranker
        results.append(gen_result)

    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate_batch(results)

    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    avg_cost = total_cost / len(results) if results else 0

    return {
        "strategy": strategy,
        "prompt_version": prompt_version,
        "num_questions": len(questions),
        "scores": scores,
        "total_cost_usd": round(total_cost, 5),
        "avg_cost_per_query_usd": round(avg_cost, 6),
        "total_tokens": total_tokens,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fixed_size", "document_aware", "both"], default="both")
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--rerank", action="store_true", help="Add cross-encoder reranking")
    args = parser.parse_args()

    cfg = load_config()
    strategies = (
        ["fixed_size", "document_aware"] if args.strategy == "both" else [args.strategy]
    )

    phoenix_session = init_phoenix()

    mlflow = MLflowTracker(
        experiment_name=cfg["observability"]["mlflow_experiment"],
        tracking_uri=cfg["observability"]["mlflow_tracking_uri"],
    )
    wandb = WandbTracker(project=cfg["observability"]["wandb_project"])

    all_results = {}
    for strategy in strategies:
        run_name = f"{strategy}_{args.prompt}{'_rerank' if args.rerank else ''}"
        logger.info(f"\n{'='*60}\nRunning experiment: {run_name}\n{'='*60}")

        params = {
            "strategy": strategy,
            "prompt_version": args.prompt,
            "top_k": cfg["retrieval"]["top_k"],
            "model": cfg["generation"]["model"],
            "reranker": cfg["retrieval"]["reranker_model"] if args.rerank else "none",
        }

        with mlflow:
            mlflow.start_run(run_name=run_name, params=params)
            experiment = run_experiment(strategy, args.prompt, cfg, use_reranker=args.rerank)
            mlflow.log_metrics({
                **experiment["scores"],
                "total_cost_usd": experiment["total_cost_usd"],
                "avg_cost_per_query_usd": experiment["avg_cost_per_query_usd"],
                "total_tokens": experiment["total_tokens"],
            })

        wandb.start_run(run_name=run_name, config=params)
        wandb.log({
            **experiment["scores"],
            "total_cost_usd": experiment["total_cost_usd"],
            "avg_cost_per_query_usd": experiment["avg_cost_per_query_usd"],
            "total_tokens": experiment["total_tokens"],
        })
        wandb.log_table("eval_results", experiment["results"])
        wandb.end_run()

        all_results[run_name] = experiment

        out_path = Path("experiments") / f"{run_name}_results.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(experiment, indent=2, default=str))
        logger.info(f"Results saved to {out_path}")

    logger.info("\nAll experiments complete.")
    for name, exp in all_results.items():
        s = exp["scores"]
        logger.info(
            f"  {name}: faithfulness={s.get('faithfulness', 0):.3f}, "
            f"relevancy={s.get('answer_relevancy', 0):.3f}, "
            f"precision={s.get('context_precision', 0):.3f}, "
            f"recall={s.get('context_recall', 0):.3f}"
        )


if __name__ == "__main__":
    main()
