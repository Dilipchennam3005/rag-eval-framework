import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from sqlalchemy import (
    create_engine, Column, String, Integer,
    DateTime, Text, Float, create_engine
)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    cik = Column(String(20), nullable=False)
    filing_date = Column(String(20), nullable=False)
    form_type = Column(String(20), nullable=False)
    local_path = Column(String(500), nullable=False)
    accession_number = Column(String(50), nullable=False, unique=True)
    num_sections = Column(Integer, default=0)
    full_text_length = Column(Integer, default=0)
    parsed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="downloaded")


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    accession_number = Column(String(50), nullable=False)
    ticker = Column(String(10), nullable=False)
    filing_date = Column(String(20), nullable=False)
    section_name = Column(String(100), nullable=False)
    char_count = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentRegistry:
    def __init__(self, db_path: str = "data/registry.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        logger.info(f"Registry initialized at {db_path}")

    def register_document(self, download_meta: dict, parse_result: dict) -> int:
        with Session(self.engine) as session:
            # Check if already exists
            existing = session.query(Document).filter_by(
                accession_number=download_meta["accession_number"]
            ).first()

            if existing:
                logger.info(f"Document already registered: {download_meta['accession_number']}")
                return existing.id

            doc = Document(
                ticker=download_meta["ticker"],
                cik=download_meta["cik"],
                filing_date=download_meta["filing_date"],
                form_type=download_meta["form"],
                local_path=download_meta["local_path"],
                accession_number=download_meta["accession_number"],
                num_sections=parse_result["num_sections"],
                full_text_length=parse_result["full_text_length"],
                status="parsed",
            )
            session.add(doc)

            # Register all sections
            for section in parse_result["sections"]:
                sec = Section(
                    accession_number=download_meta["accession_number"],
                    ticker=download_meta["ticker"],
                    filing_date=download_meta["filing_date"],
                    section_name=section["section_name"],
                    char_count=section["char_count"],
                    text=section["text"],
                )
                session.add(sec)

            session.commit()
            logger.info(
                f"Registered: {download_meta['ticker']} "
                f"{download_meta['filing_date']} "
                f"({parse_result['num_sections']} sections)"
            )
            return doc.id

    def get_all_sections(self, ticker: str = None) -> list[dict]:
        with Session(self.engine) as session:
            query = session.query(Section)
            if ticker:
                query = query.filter_by(ticker=ticker)
            sections = query.all()
            return [
                {
                    "id": s.id,
                    "accession_number": s.accession_number,
                    "ticker": s.ticker,
                    "filing_date": s.filing_date,
                    "section_name": s.section_name,
                    "char_count": s.char_count,
                    "text": s.text,
                }
                for s in sections
            ]

    def get_stats(self) -> dict:
        with Session(self.engine) as session:
            num_docs = session.query(Document).count()
            num_sections = session.query(Section).count()
            tickers = [r[0] for r in session.query(Document.ticker).distinct()]
            return {
                "num_documents": num_docs,
                "num_sections": num_sections,
                "tickers": tickers,
            }


if __name__ == "__main__":
    from sec_downloader import download_company_filings
    from parser import parse_htm_filing
    from pathlib import Path

    registry = DocumentRegistry()

    companies = [
        {"ticker": "AAPL", "cik": "0000320193"},
    ]

    for company in companies:
        downloads = download_company_filings(
            ticker=company["ticker"],
            cik=company["cik"],
            num_filings=2,
        )
        for download_meta in downloads:
            parse_result = parse_htm_filing(Path(download_meta["local_path"]))
            registry.register_document(download_meta, parse_result)

    stats = registry.get_stats()
    print(f"\nRegistry stats: {json.dumps(stats, indent=2)}")