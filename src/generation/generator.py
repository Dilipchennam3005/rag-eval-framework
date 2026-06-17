import os
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
from .prompt_manager import PromptManager

load_dotenv()

# gpt-4o-mini pricing (per token)
_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000


def _build_client() -> OpenAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Wrap with LangSmith tracing if key is present
    if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
            logger.info("LangSmith tracing enabled for generator")
        except ImportError:
            pass
    return client


class Generator:

    def __init__(self, model: str = "gpt-4o-mini", prompt_version: str = "v1"):
        self.model = model
        self.prompt_manager = PromptManager(version=prompt_version)
        self.client = _build_client()
        logger.info(f"Generator: model={model}, prompt={prompt_version}")

    def generate(self, question: str, chunks: list[dict]) -> dict:
        prompt = self.prompt_manager.build(question, chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage
        cost_usd = (
            usage.prompt_tokens * _INPUT_COST_PER_TOKEN
            + usage.completion_tokens * _OUTPUT_COST_PER_TOKEN
        )

        result = {
            "question": question,
            "answer": answer,
            "model": self.model,
            "prompt_version": self.prompt_manager.version,
            "num_chunks": len(chunks),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(cost_usd, 6),
            "contexts": [c["text"] for c in chunks],
            "context_metadata": [c.get("metadata", {}) for c in chunks],
        }

        logger.info(
            f"Generated answer: {len(answer)} chars, "
            f"{usage.total_tokens} tokens, ${cost_usd:.5f}"
        )
        return result

    def switch_prompt(self, version: str) -> None:
        self.prompt_manager.switch_version(version)
