FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY data/bm25_index_fixed_size.pkl data/bm25_index_fixed_size.pkl
COPY data/bm25_index_document_aware.pkl data/bm25_index_document_aware.pkl
COPY data/registry.db data/registry.db
COPY experiments/ experiments/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
