import re
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger


SECTION_HEADERS = [
    "risk factors",
    "properties",
    "legal proceedings",
    "mine safety disclosures",
    "market for registrant",
    "selected financial data",
    "management",
    "quantitative and qualitative disclosures",
    "financial statements",
    "changes in and disagreements",
    "controls and procedures",
    "other information",
    "directors",
    "executive compensation",
    "security ownership",
    "certain relationships",
    "principal accountant",
    "business",
]


def parse_htm_filing(file_path: Path) -> dict:
    """
    Parse a 10-K HTM filing into structured sections.
    Returns a dict with metadata and a list of sections.
    """
    logger.info(f"Parsing: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove script and style tags
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()

    # Extract plain text
    full_text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in full_text.splitlines()]
    lines = [line for line in lines if line]
    full_text = "\n".join(lines)

    # Extract sections by scanning for known headers
    sections = extract_sections(full_text)

    # Extract basic metadata from filename
    filename = file_path.stem  # e.g. aapl-20240928
    parts = filename.split("-")
    ticker = parts[0].upper()
    filing_date = "-".join(parts[1:]) if len(parts) > 1 else "unknown"

    result = {
        "file_path": str(file_path),
        "ticker": ticker,
        "filing_date": filing_date,
        "full_text_length": len(full_text),
        "num_sections": len(sections),
        "sections": sections,
    }

    logger.info(f"Parsed {ticker} filing: {len(sections)} sections, {len(full_text):,} chars")
    return result


def extract_sections(text: str) -> list[dict]:
    """
    Split full text into named sections based on known 10-K headers.
    """
    sections = []
    lines = text.split("\n")

    current_section = "preamble"
    current_lines = []

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line matches a known section header
        matched_header = None
        for header in SECTION_HEADERS:
            if line_lower.startswith(f"item") and header in line_lower:
                matched_header = header
                break

        if matched_header:
            # Save the previous section
            if current_lines:
                sections.append({
                    "section_name": current_section,
                    "text": "\n".join(current_lines).strip(),
                    "char_count": len("\n".join(current_lines)),
                })
            current_section = matched_header
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    if current_lines:
        sections.append({
            "section_name": current_section,
            "text": "\n".join(current_lines).strip(),
            "char_count": len("\n".join(current_lines)),
        })

    # Filter out tiny sections (likely parsing noise)
    sections = [s for s in sections if s["char_count"] > 200]
    return sections


def parse_all_filings(raw_dir: str = "data/raw") -> list[dict]:
    """
    Parse all downloaded HTM filings in the raw directory.
    """
    raw_path = Path(raw_dir)
    all_parsed = []

    htm_files = list(raw_path.rglob("*.htm"))
    logger.info(f"Found {len(htm_files)} HTM files to parse")

    for file_path in htm_files:
        try:
            parsed = parse_htm_filing(file_path)
            all_parsed.append(parsed)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")

    return all_parsed


if __name__ == "__main__":
    results = parse_all_filings()
    for r in results:
        print(f"\n{'='*50}")
        print(f"Ticker: {r['ticker']} | Date: {r['filing_date']}")
        print(f"Sections found: {r['num_sections']}")
        for s in r["sections"][:5]:  # Show first 5 sections
            print(f"  - {s['section_name']}: {s['char_count']:,} chars")