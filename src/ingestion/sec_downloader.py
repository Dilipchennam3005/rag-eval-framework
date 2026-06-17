import time
import requests
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "User-Agent": "rag-eval-framework dilip@example.com",
    "Accept-Encoding": "gzip, deflate",
}

BASE_URL = "https://www.sec.gov"
DATA_URL = "https://data.sec.gov"


def get_filing_urls(cik: str, filing_type: str = "10-K", num_filings: int = 2) -> list[dict]:
    cik_padded = cik.zfill(10)
    url = f"{DATA_URL}/submissions/CIK{cik_padded}.json"
    logger.info(f"Fetching filing list for CIK {cik}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    filings = data["filings"]["recent"]

    results = []
    for i, form in enumerate(filings["form"]):
        if form == filing_type:
            accession = filings["accessionNumber"][i]
            results.append({
                "accession_number": accession,
                "accession_clean": accession.replace("-", ""),
                "filing_date": filings["filingDate"][i],
                "form": form,
                "primary_document": filings["primaryDocument"][i],
            })
        if len(results) >= num_filings:
            break

    logger.info(f"Found {len(results)} {filing_type} filings")
    return results


def download_filing(cik: str, filing_meta: dict, save_dir: Path) -> Path:
    cik_int = int(cik)
    accession_clean = filing_meta["accession_clean"]
    primary_doc = filing_meta["primary_document"]

    # Direct URL — EDGAR submissions JSON already has the correct primary document name
    doc_url = f"{BASE_URL}/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_doc}"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / primary_doc

    if save_path.exists():
        logger.info(f"Already downloaded: {save_path}")
        return save_path

    logger.info(f"Downloading: {doc_url}")
    response = requests.get(doc_url, headers=HEADERS)

    if response.status_code == 404:
        # Try fetching the index page to find the real filename
        index_url = f"{BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}&type={filing_meta['form']}&dateb=&owner=include&count=10&search_text="
        logger.warning(f"Primary doc 404, filing may use different filename: {doc_url}")
        raise ValueError(f"404 for {doc_url} — check primary document name")

    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Saved: {save_path} ({len(response.content) / 1024:.1f} KB)")
    time.sleep(0.5)
    return save_path


def download_company_filings(
    ticker: str,
    cik: str,
    filing_type: str = "10-K",
    num_filings: int = 2,
    raw_dir: str = "data/raw"
) -> list[dict]:
    save_dir = Path(raw_dir) / ticker
    filing_metas = get_filing_urls(cik, filing_type, num_filings)
    downloaded = []
    for meta in filing_metas:
        try:
            path = download_filing(cik, meta, save_dir)
            downloaded.append({
                "ticker": ticker,
                "cik": cik,
                "filing_date": meta["filing_date"],
                "form": meta["form"],
                "local_path": str(path),
                "accession_number": meta["accession_number"],
            })
        except Exception as e:
            logger.error(f"Failed to download {meta['accession_number']}: {e}")
    return downloaded


if __name__ == "__main__":
    results = download_company_filings(
        ticker="AAPL",
        cik="0000320193",
        num_filings=2
    )
    for r in results:
        print(r)