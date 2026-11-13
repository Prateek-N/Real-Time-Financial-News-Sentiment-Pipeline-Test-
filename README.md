# Real-Time Financial News Sentiment Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/streaming-Apache_Kafka-black.svg)](https://kafka.apache.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FinBERT](https://img.shields.io/badge/HuggingFace-ProsusAI%2Ffinbert-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An end-to-end, high-throughput streaming data engineering and NLP portfolio system built strictly against the engineering specifications in `Real_Time_Financial_News_Sentiment_PRD.md`.

---

## 1. System Architecture

```text
AUTHORISED PUBLISHER FEEDS (5 distinct publishers: Reuters, Bloomberg, CNBC, MarketWatch, WSJ)
        | live push / approved APIs              | three-month archive
        v                                        v
[5 source adapters]                       [Backfill extractor]
        |                                        |
        v                                        v
[Validation + normalization + idempotency + deduplication]
        |                                        |
        v                                        v
Kafka: news.raw.v1                       Raw Parquet / object storage
        |                                        |
        v                                        v
[Inference worker(s): PyTorch + FinBERT]   [Batch inference: PyTorch]
        |             ^                          |           ^
        |             |                          |           |
        +--------> [Redis model cache] <---------+-----------+
        |
        +--> PostgreSQL / SQLite (articles, scores, publisher/ticker associations)
        |
        +--> Kafka: news.scored.v1
                        |
                        v
             [Streamlit live dashboard]

Observability: metrics + structured JSON logs + run manifest + failure/DLQ
Evaluation: human-annotated gold set --> baseline vs fine-tuned model --> report
```

---

## 2. Key Resume Claims & Evidence Contract

| Resume Claim | Engineering Implementation | Validation Artifact |
|---|---|---|
| **"Apache Kafka and FinBERT"** | Kafka durable partitioned streaming (`news.raw.v1`, `news.scored.v1`), micro-batched PyTorch FinBERT inference with `id2label` mapping (`0: positive`, `1: negative`, `2: neutral`). | `src/stream/`, `src/ml/inference.py` |
| **"Approximately 200 articles/min"** | Sustained 15-minute load benchmark processing $\ge 200$ unique articles/min with bounded Kafka lag and zero dropped messages. | `artifacts/perf/throughput_report.json`, `artifacts/perf/throughput_timeseries.csv` |
| **"From 5 news sources"** | 5 verified original publisher identities (`reuters`, `bloomberg`, `cnbc`, `marketwatch`, `wsj`) with audited licensing, timestamp semantics, and rate limits. | `artifacts/sources/source_manifest.csv`, `docs/source_manifest.md` |
| **"Sub-5-second publication-to-score latency"** | Measured $p95(\text{score\_completed\_at} - \text{published\_at}) < 5.0\text{s}$ on publisher-confirmed live events. | `artifacts/perf/live_latency_report.json`, `artifacts/perf/stage_latency_histogram.csv` |
| **"Streamlit dashboard"** | Interactive dashboard with live headline cards, 5-minute rolling sentiment trends, publisher/ticker filters, and confidence inspect drawers. | `src/dashboard/app.py`, `artifacts/ui/live_dashboard.png` |
| **"3 months of historical headlines"** | 3-month calendar backfill window (Aug 1, 2026 – Nov 1, 2026) across market days and sources. | `config/backfill.yaml`, `artifacts/data/history_manifest.json` |
| **"PyTorch and Redis caching"** | Model-versioned Redis cache (`finbert:{model_version}:{preprocess_version}:{sha256(headline)}`) providing multi-fold speedup on replayed/repeated queries. | `src/ml/cache.py`, `artifacts/cache/cold_vs_warm_run.json` |
| **"50,000 labeled records"** | Exactly 50,000 unique machine-labeled records exported to Parquet with SHA-256 manifest and data card. | `data/history_scored_50k.parquet`, `docs/data_card.md` |
| **"F1 improved 0.74 to 0.84"** | Controlled comparison on identical untouched held-out human-annotated test set; unweighted 3-class macro-F1 improved from $0.74$ to $0.84$ with paired bootstrap 95% CI. | `artifacts/eval/model_comparison.md`, `artifacts/eval/confusion_matrices.png`, `artifacts/eval/paired_bootstrap_ci.json` |

---

## 3. Quickstart & Acceptance Command Suite

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/unit tests/integration -v
```

### 3. Execute 3-Month Historical Backfill
```bash
python -m src.batch.backfill --config config/backfill.yaml
```

### 4. Score Historical Archive & Export 50,000 Records
```bash
python -m src.batch.score_archive --dataset data/raw_archive
```

### 5. Generate Gold Splits, Fine-tune & Evaluate Model
```bash
python -m src.training.create_splits --gold data/gold_labels.csv
python -m src.training.train --epochs 2 --batch-size 16
python -m src.training.evaluate --baseline ProsusAI/finbert --candidate artifacts/model
```

### 6. Run Sustained 200 Articles/Min Load Benchmark
```bash
python -m tests.load.benchmark --rate-per-min 200 --minutes 15 --mode replay
```

### 7. Compile Proof Package & Telemetry
```bash
python -m src.reporting.build_evidence
```

### 8. Launch Streamlit Live Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 4. Truthful Integrity Disclosures
- **Replay Capacity vs. Organic Volume:** Sustained 200 articles/min represents verified system scoring capacity. Organic live feeds may have lower volume during off-market hours.
- **Machine Pseudo-Labels:** The 50,000 historical Parquet records are model-predicted pseudo-labels, not human gold truth. Human evaluation is performed strictly on `data/gold_test.csv` (800 test instances).
- **Disclaimer:** Sentiment analytics are generated for research and educational purposes, not investment advice.

