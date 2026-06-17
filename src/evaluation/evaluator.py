import json
from pathlib import Path
from loguru import logger

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


TEST_QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"


def load_test_questions(ticker: str = None) -> list[dict]:
    questions = json.loads(TEST_QUESTIONS_PATH.read_text())
    if ticker:
        questions = [q for q in questions if q.get("ticker") == ticker]
    return questions


class RAGASEvaluator:

    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    def evaluate_batch(self, results: list[dict]) -> dict:
        """
        Run RAGAS over a list of generation results.

        Each result must have:
          - question: str
          - answer: str
          - contexts: list[str]
          - ground_truth: str
        """
        logger.info(f"Running RAGAS evaluation on {len(results)} samples")

        dataset = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ])

        scores = evaluate(dataset, metrics=self.metrics)
        score_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

        logger.info(
            f"RAGAS scores — faithfulness={score_dict.get('faithfulness', 0):.3f}, "
            f"answer_relevancy={score_dict.get('answer_relevancy', 0):.3f}, "
            f"context_precision={score_dict.get('context_precision', 0):.3f}, "
            f"context_recall={score_dict.get('context_recall', 0):.3f}"
        )
        return score_dict

    def evaluate_single(self, result: dict) -> dict:
        return self.evaluate_batch([result])
