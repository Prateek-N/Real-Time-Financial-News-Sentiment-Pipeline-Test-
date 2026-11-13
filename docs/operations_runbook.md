# Operations Runbook: Real-Time Financial News Sentiment Pipeline

## Architecture Overview
The pipeline consists of:
1. **Source Connectors:** Ingest push/poll events from 5 authorized publishers.
2. **Apache Kafka (KRaft):** Durable event streaming across `news.raw.v1`, `news.scored.v1`, and `news.dead_letter.v1`.
3. **Inference Workers:** PyTorch FinBERT workers with opportunistic micro-batching (`max_items=16, max_wait_ms=100`).
4. **Redis Cache:** Model-versioned caching (`finbert:{model_version}:{preprocess_version}:{sha256(headline)}`).
5. **PostgreSQL / Parquet:** Idempotent relational storage and cold historical Parquet archives.
6. **Streamlit Live Dashboard:** Real-time visual monitoring with auto-refreshing KPI strips and ticker filters.

## Common Operations & Commands

### 1. Environment Setup
```bash
cp .env.example .env
pip install -r requirements.txt
```

### 2. Historical Backfill (3 Months)
```bash
python -m src.batch.backfill --config config/backfill.yaml
```

### 3. Batch Scoring & 50,000 Dataset Export
```bash
python -m src.batch.score_archive --dataset data/raw_archive
```

### 4. Human Gold Evaluation Splits & Training
```bash
python -m src.training.create_splits --gold data/gold_labels.csv
python -m src.training.train --epochs 2 --batch-size 16
python -m src.training.evaluate --baseline ProsusAI/finbert --candidate artifacts/model
```

### 5. Sustained Load & Latency Benchmark (15 min @ 200/min)
```bash
python -m tests.load.benchmark --rate-per-min 200 --minutes 15 --mode replay
```

### 6. Build Evidence Audit Folder
```bash
python -m src.reporting.build_evidence
```

### 7. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Failure Recovery & Resilience
- **Database Disconnection:** The pipeline buffers uncommitted micro-batches and retries with exponential backoff.
- **Poison Messages:** Malformed JSON or invalid UTF-8 headlines are routed directly to `news.dead_letter.v1` with error metadata, preventing consumer stall.
- **Worker Crash:** Offsets are committed only after successful DB persistence, ensuring at-least-once delivery with idempotent deduplication on retry.

