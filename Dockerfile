# Fundsconnect — production image (Render / Railway / Fly)
FROM python:3.12-slim

WORKDIR /app

# System deps (certs + build bits some wheels need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + data (rankings + current holdings snapshot)
COPY app.py fund_scorer.py metrics.py data_source.py \
     holdings_source.py holdings_fetch.py pe_enrich.py \
     overlap_engine.py nav_source.py ./
COPY static ./static
COPY funds.xlsx ranked.csv holdings.csv ./
# optional helpers (safe if used later)
COPY holdings_upload_TEMPLATE.csv ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Render injects $PORT — bind all interfaces
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
