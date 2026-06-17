from pathlib import Path


PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptManager:

    def __init__(self, version: str = "v1"):
        self.version = version
        self._template = self._load(version)

    def _load(self, version: str) -> str:
        candidates = list(PROMPTS_DIR.glob(f"{version}*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No prompt file found for version '{version}' in {PROMPTS_DIR}")
        return candidates[0].read_text(encoding="utf-8")

    def build(self, question: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            ticker = meta.get("ticker", "?")
            date = meta.get("filing_date", "?")
            section = meta.get("section_name", "?")
            context_parts.append(f"[{i}] {ticker} {date} — {section}:\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        return self._template.format(context=context, question=question)

    def switch_version(self, version: str) -> None:
        self.version = version
        self._template = self._load(version)
