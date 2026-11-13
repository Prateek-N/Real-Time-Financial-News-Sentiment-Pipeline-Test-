# Dataset Card: 50,000 Historical Machine-Labeled Financial Headlines

## Dataset Overview
- **Dataset Name:** `history_scored_50k.parquet`
- **Total Usable Records:** Exactly 50,000 unique records
- **Temporal Window:** August 1, 2026 – November 1, 2026 (Three full calendar months)
- **Primary Task:** 3-Class Financial Sentiment Analysis (`positive`, `negative`, `neutral`)
- **Format:** Apache Parquet with Snappy compression, partitioned by date and publisher

## Provenance and Labeling Protocol
> [!IMPORTANT]
> **Integrity Disclosure:** All labels contained in this 50,000-record historical export are **machine-generated pseudo-labels** produced by batched PyTorch inference with `ProsusAI/finbert`. They are **not** independently annotated human gold labels. Human evaluation ground truth is maintained separately in `data/gold_labels.csv` (4,000 instances).

## Schema
| Field Name | Type | Description |
|---|---|---|
| `article_id` | string | Unique primary key (`{publisher_id}-{event_id}`) |
| `publisher_id` | string | Originating publisher identity (`reuters`, `bloomberg`, `cnbc`, `marketwatch`, `wsj`) |
| `provider_id` | string | Ingestion provider reference |
| `headline` | string | Cleaned and normalized headline text |
| `article_url` | string | Reference canonical URL |
| `tickers` | list[string] | Associated exchange tickers or `['unmapped']` |
| `published_at` | string | Original publication timestamp (ISO 8601 UTC) |
| `published_at_trust` | string | `publisher_confirmed` |
| `predicted_label` | string | Inferred sentiment (`positive`, `negative`, `neutral`) |
| `p_positive` | float | Softmax probability for positive sentiment |
| `p_negative` | float | Softmax probability for negative sentiment |
| `p_neutral` | float | Softmax probability for neutral sentiment |
| `sentiment_index` | float | Polarity metric defined as `p_positive - p_negative` |
| `model_id` | string | `ProsusAI/finbert` |
| `model_version` | string | Pinned model checkpoint hash |
| `cache_hit` | boolean | Cache status during historical scoring pass |
| `ingestion_mode` | string | `historical` |
| `dataset_version` | string | `1.0.0` |

## Empirical Dataset Audit and Manifest
- **Canonical Parquet Path:** `data/history_scored_50k.parquet`
- **File Size:** 6,376,721 bytes
- **SHA-256 Checksum:** `8f6926ff0b0071bb9544731b098bd1c8006967a4e58ac13e4fbc825070368c00`
- **Total Validated Rows:** Exactly 50,000 unique records (zero null values across all 30 fields)
- **Temporal Boundaries:** 2026-08-01T00:02:37Z to 2026-10-31T00:43:20Z

### Measured Publisher Distribution
- Bloomberg: 12,564 (25.13%)
- Reuters: 12,494 (24.99%)
- CNBC: 10,035 (20.07%)
- MarketWatch: 7,489 (14.98%)
- Wall Street Journal: 7,418 (14.84%)

### Measured Label Distribution
- Negative: 20,999 (42.00%)
- Positive: 16,669 (33.34%)
- Neutral: 12,332 (24.66%)
- Mean Sentiment Index: `-0.0948`

- **Audit Manifest:** Full validation recorded in `artifacts/data/history_manifest.json` and `artifacts/data/quality_report.json`.

