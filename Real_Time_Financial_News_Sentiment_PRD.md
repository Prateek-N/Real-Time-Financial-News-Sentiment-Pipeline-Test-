# Product Requirements Document

## Real-Time Financial News Sentiment Pipeline

**Version:** 1.0  
**Document status:** Implementation-ready specification; quantitative outcomes pending measurement  
**Project type:** End-to-end data engineering + NLP/MLOps + analytics portfolio project  
**Primary stack:** Python, Apache Kafka, PyTorch, Hugging Face FinBERT, Redis, PostgreSQL, Parquet, Streamlit, Docker Compose  
**Target deployment:** Local workstation or single GPU VM, with a reproducible containerized demonstration  
**Owner:** Project developer  
**Last updated:** 13 November 2026

> **Integrity note.** This PRD translates the two resume bullets into buildable requirements and validation procedures. The bullets describe claimed historical accomplishments; this document does **not** establish that they have already been achieved. A metric becomes a defensible resume claim only after its required experiment, logs, and report exist. In particular, 0.74 and 0.84 are **target/reproduction values**, not measurements supplied by this PRD.

---

# 1. Executive summary

Build a financial-news sentiment system that ingests headlines from **five distinct, authorized original news publishers**, normalizes and deduplicates the events, carries them through Apache Kafka, scores them with a **three-class FinBERT model** (positive, neutral, negative), persists structured results, and exposes source-level and ticker-level market-sentiment signals through a live Streamlit dashboard.

In parallel, build a **three-month historical backfill and batch-inference pipeline** that produces **50,000 unique, model-labeled headline records**, uses **PyTorch** for inference and fine-tuning, and **Redis** to cache results for repeated/replayed texts. Evaluate the out-of-the-box model against an improved model on the **same human-annotated, held-out test set**, targeting a macro-F1 improvement **from 0.74 to 0.84**, subject to actual measurement.

The implementation must independently prove four things: publisher provenance, streaming capacity, genuine publication-to-score latency, and statistically valid model improvement. It must not substitute synthetic data for real publisher provenance, batch throughput for live throughput, model-generated labels for evaluation ground truth, or ingestion-to-score latency for publication-to-score latency.

# 2. Resume claims and evidence contract

| Resume statement | Precise engineering interpretation | What must be produced to support the claim |
|---|---|---|
| “Apache Kafka and FinBERT” | Kafka is on the actual ingestion-to-inference path; `ProsusAI/finbert` or a documented fine-tuned derivative is invoked for sentiment. | Architecture diagram, running services, topic listings, committed configuration, model identifier/version, sample raw and scored messages. |
| “Approximately 200 articles per minute” | **At least 200 unique valid articles/minute** received and scored over a **15-minute sustained benchmark**, without unbounded backlog; additionally report peak minute and actual live arrival rate. | Timestamped throughput run, processed unique counts, Kafka lag plot, hardware specification and p50/p95/p99 scoring latency. |
| “From 5 news sources” | Five publisher identities, not merely five API endpoints or five aliases from the same publisher; licensed/authorized access to each. | Publisher manifest, access method, license/terms notes, source identifiers, distinct publisher counts and contribution chart. |
| “Sub-5-second publication-to-score latency” | On publisher events with trustworthy publish timestamps and a low-latency delivery feed, **p95(`score_completed_at` − `published_at`) < 5 seconds** at the target throughput. | Raw publisher timestamps, clock/timezone policy, per-stage traces, percentile report, measurement coverage, comparison of publication vs ingestion latency. |
| “Streamlit dashboard” | A connected dashboard displays fresh scored stories and aggregated directional sentiment; data comes from the pipeline's persisted output. | Screenshots/video, working repository, filter and refresh demonstration. |
| “3 months of historical headlines” | Data covers a documented **three-calendar-month time window**, with a distribution across dates and sources. | Backfill configuration, manifest showing start/end, per-day counts, source coverage and gaps. |
| “PyTorch and Redis caching” | Torch-backed model inference/batch training; Redis caches model outputs by normalized content and model version, with observable hits/misses. | Inference code, cache key design, instrumentation, repeated-backfill benchmark and invalidation test. |
| “50,000 labeled records” | Exactly **50,000 distinct usable headlines with predictions** plus scores, provenance and model metadata. These are **machine-labeled**, not automatically human gold labels. | Immutable Parquet export, schema, unique-key and null checks, reproducible counts, hash and dataset card. |
| “F1 improved 0.74 to 0.84” | Macro-F1 is measured for baseline and improved checkpoints on an **identical untouched human-labeled test set**. | Annotation rubric, human labels, split manifests, eval script, per-class report, confusion matrices, exact unrounded scores and bootstrap uncertainty estimates. |

**Interpretation rule:** If the observed result differs, revise the resume rather than presenting acceptance targets as recorded achievements.

# 3. Problem, opportunity and users

Financial headlines arrive continuously across multiple publishers. Generic sentiment tools often underperform on investor-facing language (“loss narrowed”, “guidance cut”, “shares rise despite miss”), and static analysis does not make fresh cross-source signals easy to monitor. This project offers a traceable engineering workflow from publisher event to interpretable sentiment score, with a historical evaluation layer.

**Primary user:** Technical interviewer / hiring manager verifying an end-to-end data and ML project.  
**Secondary user:** Research analyst exploring headline sentiment by company, source and time.  
**Operator:** Developer responsible for source connectors, pipelines, model versions, benchmarks and quality monitoring.

**Core user jobs:** Watch new headlines; inspect the three sentiment probabilities and model confidence; filter by ticker/source; view a time-window sentiment trend; compare publisher coverage; replay historical periods; and investigate processing failures and model errors.

# 4. Goals, non-goals and assumptions

## 4.1 Goals

- G1 — Acquire real financial headlines from **five authorized and separately identifiable publishers** in the live and historical modes.
- G2 — Deliver a runnable Kafka streaming path from collection through normalized events to model scores and persistent storage.
- G3 — Benchmark **>=200 unique scored articles/min** for >=15 contiguous minutes under explicit infrastructure and message-mix conditions.
- G4 — Measure and, **if feed characteristics permit**, achieve **p95 publication-to-score <5 seconds**, with >=99% trustworthy timestamp coverage over the evaluated sample.
- G5 — Produce a reproducible 50,000-row model-labeled dataset covering three documented months.
- G6 — Evaluate baseline and improved FinBERT via a human-labeled held-out sample and target **macro-F1 >=0.84**, with baseline near 0.74 only if actually measured.
- G7 — Provide a Streamlit dashboard, operational telemetry, one-command demo and an evidence folder supporting each resume metric.

## 4.2 Non-goals

- No automated stock trading, broker integration, order placement or prediction of guaranteed returns.
- No claim of stock-price prediction accuracy or profitability from sentiment classification alone.
- No unlicensed scraping, circumvention of paywalls or redistribution of copyrighted article bodies.
- No guarantee that RSS polling can meet a publisher-to-score SLA; a sub-5-second goal requires demonstrably low-latency provider delivery.
- No “50,000 human-labeled examples” claim unless 50,000 records have in fact been independently annotated.

## 4.3 Dependencies and constraints

1. **Source access is the critical external dependency.** Obtain publisher-authorized low-latency push feeds or a vendor contract exposing publisher identity, publication timestamps, history and permitted storage. Candidate publisher names might include Reuters, CNBC, MarketWatch, Yahoo Finance and Benzinga **only where access rights are confirmed**. The actual five must be filled in the source manifest; examples are not proof of integrations.
2. **Rate limits and terms** may prevent retrieving 200 *new* unique live articles/min. In that case prove **system processing capacity** using licensed historical replay/synthetic workload, then separately disclose genuine live arrival rate; do not call replay traffic organic live news.
3. Use English-language headline sentiment initially. Source language and edition must be recorded.
4. The baseline model's use and dataset licenses, including non-commercial/share-alike restrictions, must be reviewed before public or commercial deployment.
5. All services synchronize system clocks (NTP); timestamps are stored as timezone-aware UTC ISO 8601 values.

# 5. Product scope and end-to-end user flows

## 5.1 Live mode

Publisher sends event → source connector authenticates and parses → event is assigned stable `article_id` → content/metadata validated → duplicate detection occurs → raw normalized event published to Kafka → inference consumer performs model-versioned Redis lookup or FinBERT scoring → score persisted in PostgreSQL → scored event published to Kafka → Streamlit reads updated results and renders recent headlines and sentiment statistics.

## 5.2 Historical mode

For selected three-month window, authorized archival endpoint or licensed bulk dataset → resumable pagination/backfill → clean, timestamp, provenance and dedupe → immutable raw Parquet → PyTorch batched inference + versioned Redis cache → scored Parquet and PostgreSQL → precisely 50,000 eligible rows for release → train/validation/test manifests and separately labeled evaluation → model comparison report.

## 5.3 Demo mode

Bring up containers → health checks → load a small allowed sample fixture → stream/replay through Kafka at a configured event rate → watch scoring and dashboard → run performance benchmark → open evidence report. **Demo replay must have a prominent REPLAY badge** and its latency must be labeled “replay/ingestion-to-score,” not publisher publication-to-score.

# 6. Architecture and component responsibilities

```text
AUTHORISED PUBLISHER FEEDS (5 distinct publishers)
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
        +--> PostgreSQL (news, scores, publisher/ticker associations)
        |
        +--> Kafka: news.scored.v1
                        |
                        v
             [Streamlit live dashboard]

Observability: metrics + structured JSON logs + run manifest + failure/DLQ
Evaluation: human-annotated gold set --> baseline vs fine-tuned model --> report
```

| Component | Technology | Required responsibility |
|---|---|---|
| Source adapters | Python asyncio, vendor SDK/HTTP or WebSocket | Authorized fetch, publisher normalization, provider retries and backpressure, source health and feed lag. |
| Message broker | Apache Kafka (KRaft for local deployment) | Durable append-only raw/scored topics, consumer groups, partitioned processing, dead-letter handling. |
| Cleaner | Pydantic, regex/Unicode normalization | Reject malformed records, canonical text, stable IDs and correct timestamp parse. |
| Online inference | Transformers + PyTorch | Load model once at boot, `eval()`, `torch.inference_mode()`, micro-batching, return 3 softmax probabilities. |
| Model cache | Redis | Version-aware content cache, hit/miss accounting and reprocessing optimization. |
| Durable query store | PostgreSQL | Unique source article keys, latest predictions, indexes for source/ticker/time queries. |
| Archive | Parquet + manifest | Immutable raw and scored batches, reproducible hashes and data-quality audit. |
| Web UI | Streamlit | Recent cards/table, filters, charts, model and feed status, timestamp visibility. |
| Training and eval | PyTorch + scikit-learn | Baseline, fine-tune, frozen test and metrics artifacts. |
| Observability | Prometheus / Grafana optional; JSON reports mandatory | Feed lags, queue lag, throughput, cache rate, errors, inference time, timestamp completeness. |

**Kafka minimum topics:** `news.raw.v1` (6 partitions), `news.scored.v1` (6 partitions), `news.dead_letter.v1` (3 partitions). Partition sizing is an **initial hypothesis**, not a measured requirement; adjust based on load tests. Key messages by a stable event identifier. `news.raw.v1` retention: >=7 days for recoverability, subject to article rights; do not use Kafka as the only permanent archive.

**Delivery semantics:** At-least-once processing is acceptable. Commit consumer offsets after score persistence + successful event publication (or adopt a documented transactional/outbox pattern). Database upserts and stable keys make retries idempotent. Do **not** describe this as exactly-once end-to-end without proving it across external storage.

# 7. Functional requirements

| ID | Priority | Requirement | Acceptance criteria |
|---|---|---|---|
| FR-01 | Must | Five authorized source connectors or one vendor feed with five verified original publisher identities. | Source manifest and counted records establish five distinct `publisher_id` values; each source can be temporarily disabled. |
| FR-02 | Must | Normalize events to a stable canonical schema. | 100% accepted events have article ID, publisher ID, headline, publication timestamp status, ingest timestamp and source URL or licensed provider ID. |
| FR-03 | Must | Detect duplicate reprints and provider retransmissions. | Exact ID duplicates cannot create duplicate scored rows; near-duplicates are flagged without destructive false dedupe. |
| FR-04 | Must | Send all accepted live events through Kafka. | Integration test captures raw topic event, consumer output and scored topic event for the same `article_id`. |
| FR-05 | Must | Run 3-class FinBERT scoring using PyTorch. | Probabilities map to model-config label names and sum approximately to 1; model/version recorded. |
| FR-06 | Must | Write scored records for fast filtered queries. | Upsert survives Kafka retry and preserves one logical score per (`article_id`, `model_version`). |
| FR-07 | Must | Show news and aggregated signals on Streamlit. | UI displays most recent headline, publisher, tickers, label, probability and publish/score times. |
| FR-08 | Must | Historical backfill with checkpoint/resume. | Restarting during backfill produces identical unique counts with no double-counted output. |
| FR-09 | Must | Batch inference uses torch and Redis cache. | Cache key includes normalized content + model revision; repeated run is reported with real hit rate. |
| FR-10 | Must | Export 50,000 scored historical records. | Exactly 50,000 unique `article_id` values; manifest covers the three-month window and includes reproducible hash. |
| FR-11 | Must | Produce human gold labels and locked evaluation test. | Double-annotation/adjudication process and documented split; no test exposure to tuning. |
| FR-12 | Must | Compare baseline and improved models. | Same gold test set, macro-F1, per-class metrics, confusion matrices and CI; result not rounded deceptively. |
| FR-13 | Must | Observe latency and performance. | Dashboards/reports track stage times, publisher feed age, message lag and throughput. |
| FR-14 | Should | Support replay mode and model-version comparison. | UI clearly marks replay, exposes selected model, and shows repeatable sample results. |
| FR-15 | Should | Basic authentication on any public dashboard. | Environment secrets do not leak; dashboard is protected or kept local-only. |

# 8. Data contracts and processing logic

## 8.1 Canonical raw event schema (JSON example)

```json
{
  "schema_version": "1.0",
  "article_id": "publisher-123456",
  "provider_event_id": "123456",
  "publisher_id": "publisher_a",
  "provider_id": "authorized_vendor",
  "headline": "Company reports higher annual revenue",
  "language": "en",
  "article_url": "https://publisher.example/article/123456",
  "tickers": ["EXAMPLE"],
  "published_at": "2026-11-13T12:00:00.000Z",
  "published_at_trust": "publisher_confirmed",
  "provider_received_at": "2026-11-13T12:00:01.100Z",
  "ingested_at": "2026-11-13T12:00:01.300Z",
  "ingestion_mode": "live",
  "content_sha256": "64-hex-character-digest",
  "source_license_id": "access-agreement-reference"
}
```

`published_at` may be null when the provider does not expose reliable timestamps; such records are useful for scoring but **excluded** from publication-to-score percentile statistics, and exclusion count must be reported. A URL may be substituted by a licensed stable provider ID if URLs are unavailable; record the reason. For replay, retain original publication time for provenance but record `replay_emitted_at` and measure a separate replay latency.

## 8.2 Canonical score schema

```json
{
  "article_id": "publisher-123456",
  "model_id": "ProsusAI/finbert",
  "model_version": "pinned-commit-or-checkpoint-sha",
  "preprocess_version": "v1",
  "label": "positive",
  "p_positive": 0.81,
  "p_negative": 0.07,
  "p_neutral": 0.12,
  "sentiment_index": 0.74,
  "cache_hit": false,
  "inference_started_at": "2026-11-13T12:00:01.500Z",
  "score_completed_at": "2026-11-13T12:00:02.040Z",
  "mode": "live"
}
```

`sentiment_index = p_positive - p_negative` ranges from −1 to +1; it is **sentiment strength**, not expected return or a buy/sell recommendation. Always validate the true label order from `model.config.id2label`, not from unrelated dataset integer codes.

## 8.3 Data quality and deduplication

- Required: nonblank headline, valid UTF-8, canonical publisher ID, parseable time or explicitly missing publisher time, provider ID/URL, explicit ingestion mode.
- Normalize whitespace, Unicode and HTML entities; do not remove negation, numbers, currency signs or finance-specific wording.
- Compute stable `article_id` from source-native ID when available; otherwise namespace source URL; retain original IDs.
- Exact duplicate rule: same publisher + provider ID or same authoritative URL; maintain a uniqueness constraint.
- Detect cross-publisher wire-service duplication with headline hash/semantic similarity, but **do not erase separate source provenance**. For evaluation, group near-identical story clusters into the same split.
- Ticker mapping must distinguish mentioned company from affected company. Show `unmapped` where uncertain, rather than hallucinating a symbol.
- Record rejected, missing-timestamp, non-English and duplicate counts per publisher.

## 8.4 Suggested PostgreSQL design

```sql
CREATE TABLE articles (
    article_id TEXT PRIMARY KEY,
    publisher_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_event_id TEXT,
    headline TEXT NOT NULL,
    article_url TEXT,
    published_at TIMESTAMPTZ,
    published_at_trust TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    ingestion_mode TEXT NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    tickers TEXT[] NOT NULL DEFAULT '{}'
);
CREATE TABLE sentiment_scores (
    article_id TEXT NOT NULL REFERENCES articles(article_id),
    model_version TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positive','negative','neutral')),
    p_positive DOUBLE PRECISION NOT NULL,
    p_negative DOUBLE PRECISION NOT NULL,
    p_neutral DOUBLE PRECISION NOT NULL,
    score_completed_at TIMESTAMPTZ NOT NULL,
    cache_hit BOOLEAN NOT NULL,
    PRIMARY KEY (article_id, model_version, preprocess_version)
);
CREATE INDEX ix_articles_published_at ON articles (published_at DESC);
CREATE INDEX ix_articles_publisher_date ON articles (publisher_id, published_at DESC);
```

Store raw licensed archive in Parquet partitioned by `published_date` and `publisher_id`; do not persist restricted full article bodies if the contract permits only titles/metadata.

# 9. Five-source acquisition strategy

**Definition:** “five news sources” means five **original publishers**, even if one authorized aggregator/vendor delivers all five. Five endpoints of one publisher do not qualify. A single aggregation API can be operationally valid if the underlying `publisher_id` is reliable and five publishers are independently represented.

Create `config/sources.yaml` with each `publisher_id`, display name, feed/vendor, endpoint type (push/poll), authorization proof/reference, allowed fields, allowed historical lookback, publisher timestamp semantics, quota, attribution URL and whether storage/republishing is permitted. **Do not hard-code or imply access to named publishers until verified.**

### Connector behavior

1. Subscribe through vendor WebSocket/webhook/push route when available; use approved polling only if measured performance allows.
2. Emit an ingestion heartbeat and `last_seen` age per source; keep disconnected source status visible on the dashboard.
3. Retry `429` and transient `5xx` responses with capped exponential backoff and jitter; respect rate-limit headers and quotas.
4. Stamp arrival time immediately at connector receipt, before downstream work; preserve source-published time unchanged.
5. On restart, continue from cursor if supported; otherwise replay recent provider window and depend on idempotency.
6. Validate that each source has a meaningful nonzero share during collection and log per-source rates. Do not fabricate articles to fill a publisher quota.

**Acceptance gate:** Five working integrations with lawful access are mandatory for the unqualified resume wording; if source access is blocked, the honest alternative is “five configured publisher adapters with replay” or “one licensed feed with five publishers represented,” as actually implemented.

# 10. Streaming inference and Redis behavior

## 10.1 Online worker

- Initialize tokenizer and pinned FinBERT checkpoint **once per process**; `model.eval()` and `torch.inference_mode()` are mandatory.
- Batch messages opportunistically (for example, up to 8–16 headlines or <=100 ms wait) to improve throughput without sacrificing latency. Make the maximum wait configurable and benchmark it.
- Truncate to a declared input length (e.g., 128 tokens for headlines), recording the truncation rate.
- Extract probabilities by `id2label` mapping. Record `p_positive`, `p_negative`, `p_neutral`, top label and model version.
- Write/upsert durable score, publish scored event, then commit offsets; retry transient failures; route poison events to dead-letter topic.
- Handle graceful shutdown so in-flight messages are completed or cleanly reprocessed.

## 10.2 Cache correctness

**Key:** `finbert:{model_checkpoint_sha}:{preprocess_version}:{sha256(normalized_headline)}`.  
**Value:** Label + three probabilities + model/version + cache creation timestamp.  
**TTL:** Start with 30 days; adjust for storage and terms.  
**Rule:** Redis stores **derived model outputs** only. Updating the checkpoint or normalization automatically creates new keys. Inferred results must never be reused across model versions.

At first pass over **50,000 unique headlines**, cache hits may be near zero; Redis's value is in repeated headline variants, retried messages, incremental backfills and reproducible reruns. Report cache hit rate separately for **cold** and **warm** runs, plus latency difference; never credit Redis with magically speeding up the first-ever inference for unseen text.

## 10.3 Sample worker pseudocode

```python
model = AutoModelForSequenceClassification.from_pretrained(PINNED_MODEL).eval()
tokenizer = AutoTokenizer.from_pretrained(PINNED_MODEL)
for batch in kafka_consumer.get_microbatch(max_items=16, max_wait_ms=100):
    records = validate_dedupe(batch)
    misses, cached = redis_lookup_by_version(records)
    if misses:
        tokens = tokenizer([r.headline for r in misses], padding=True,
                           truncation=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=-1)
        results = map_probs_by_model_id2label(probs, misses)
        redis_set_versioned(results, ttl_seconds=30 * 86400)
    scored = merge_preserving_order(cached, results)
    database_upsert(scored)
    publish_scored_events(scored)
    commit_offsets_after_success()
```

# 11. Three-month historical pipeline and 50,000-row deliverable

## 11.1 Time and size definitions

Specify exact calendar boundaries (e.g., `[2026-08-01T00:00:00Z, 2026-11-01T00:00:00Z)`), ensuring they span **three full calendar months**. The example is illustrative; select a licensed archive window with available coverage. Produce both pre-filter and post-filter counts by day/publisher.

A 50,000-record output over approximately 90 days averages **about 556 retained articles/day**. This is logically compatible with a pipeline that can process 200 articles/min **under benchmark load**. **It is not compatible with claiming a constant real arrival rate of 200/min throughout 90 days**; that would imply about 25.9 million articles. Keep capacity, real feed arrivals and historical retention as separate metrics.

## 11.2 Backfill workflow

1. Extract per-source paginated history by published date; save cursors and source errors; comply with licenses and rate limits.
2. Normalize schema, language, timestamps, URLs and source identities; retain raw provenance and rejected record logs.
3. Deduplicate within provider and by canonical ID; group near-duplicates only for evaluation leakage prevention.
4. Filter by approved scope (English financial/business headlines, acceptable length, authentic time and source).
5. Write immutable **raw Parquet** with a dated run ID and SHA-256 manifest.
6. Score with `DataLoader`/batched tokenization + `torch.inference_mode()`, writing cached predictions when valid.
7. Export **exactly 50,000 unique usable records** through a documented deterministic rule: chronological/stratified sampling of the eligible pool, fixed seed and source/day constraints. If fewer are legally available, **do not pad with duplicates or invented news**; do not claim 50,000.
8. Write `history_scored_50k.parquet`, source/date distribution, dataset card, schema validation and content hash.

**Dataset row minimum:** Article ID, publisher ID, published timestamp, headline or allowed reference, tickers, label, three probability values, `sentiment_index`, model ID/commit, cache_hit, ingestion mode, dataset version. A machine-generated label field must be named `predicted_label` or `pseudo_label`, not `gold_label`.

## 11.3 Data split and lineage

Maintain **two related but different datasets**: (A) the 50,000-row machine-scored archive and (B) an independently human-annotated **gold evaluation/training subset**. The gold subset can reference archive IDs without being confused with the pseudo-labeled export. Use a time-forward or story-group-disjoint partition; if training includes earlier historical stories, keep the most recent weeks as untouched testing and put a time gap between train/test when feasible. Verify no identical/near-identical story crosses partitions.

# 12. Sentiment labels, human annotation and training

## 12.1 Labeling rubric

Label from a reasonable investor's perspective **based only on information in the headline**:

- **Positive:** Signals a favorable development (e.g., earnings materially above expectations, raised guidance, regulatory approval).
- **Negative:** Signals an adverse development (e.g., downward guidance revision, fraud allegation, material revenue decline).
- **Neutral:** Factual update, no clear direction, mixed or insufficient context; avoid inferring market movement from unsupported assumptions.

Annotate the sentiment of the headline **as written**, not the actual next-day stock price. For mixed company-specific sentiment, apply an explicit target-entity rubric or mark ambiguous and exclude from the single-label gold benchmark. Keep class definitions fixed across train, validation and test.

## 12.2 Human gold set requirements

Recommended goal: **at least 4,000 independently human-labeled headline instances**, sampled across dates, publishers, company sizes and classes. Have two annotators label a shared substantial portion (preferably every gold test item); adjudicate conflicts by a third qualified reviewer. Report inter-annotator agreement (e.g., Cohen's kappa) and exclude irresolvable cases by documented policy. Do not use raw FinBERT predictions as sole ground truth.

Create train/validation/test manifests **before** fine-tuning, e.g., 60% / 20% / 20% by time/group where workable. Do not let the final test set influence hyperparameters or threshold selection. Lock labels and test IDs in versioned files.

## 12.3 Baseline and improved model

**Baseline:** Unmodified `ProsusAI/finbert` at pinned model revision; preprocessing and gold test records identical to improved model. Baseline reported as macro-F1 **only after measurement**; 0.74 is the resume target, not a guaranteed baseline.

**Improved model:** Fine-tune the same starting checkpoint using high-quality, training-only human labels, optionally augmented with **confidence-filtered pseudo labels** from the 50k archive after removing validation/test story clusters. Explore class-weighted loss, learning rate (e.g., 1e-5 to 3e-5), batch size 16/32, early stopping and 2–4 epochs. Choose checkpoint on **validation macro-F1**, not test. Calibrate probabilities on validation only if needed; changing confidence calibration alone usually does not change argmax F1 unless decision rules are explicitly adjusted. Save model revision, tokenizer, config, seed, package versions and GPU/CPU details.

**Example PyTorch training controls:** `torch.manual_seed(42)`, deterministic seed log, `AdamW`, linear warmup, gradient clipping, batch sampler audit and `model.train()` for fine-tuning. Run at least three seeds when budget permits and report variability.

# 13. Model evaluation protocol and the 0.74 → 0.84 claim

**Metric:** Three-class **macro-F1**, the unweighted mean of per-class F1 for positive, negative and neutral. Do not substitute accuracy or weighted F1. Formula: `F1_c = 2·precision_c·recall_c / (precision_c + recall_c)`; `macro_F1 = mean(F1_positive, F1_negative, F1_neutral)`.

**Controlled comparison:** identical fixed human test IDs, identical text normalization, documented base-model commit vs fine-tuned commit, no test fine-tuning or hyperparameter selection. Produce class support counts, per-class precision/recall/F1, confusion matrix, precision-weighted and macro aggregates (clearly labeled), calibration plot if using confidence, test-set source/date breakdown and error taxonomy (negation, mixed headlines, earnings surprise, named entities, duplicate stories).

**Target acceptance:** Baseline rounded macro-F1 = **0.74**; improved rounded macro-F1 = **0.84**, with exact unrounded values shown, absolute gain about **+0.10**. If exact values do not reproduce, record actual results and edit the resume. A +0.10 gain is **10 percentage points**, not a “10% relative improvement”; if baseline is exactly 0.74, relative gain is about 13.5%.

**Uncertainty:** Paired bootstrap of test examples/story clusters (>=1,000 resamples) to estimate a 95% confidence interval for `Δmacro_F1`. Ideally the interval excludes zero, but this must be **reported, not assumed**. Give a caveat if test data is too small or class-imbalanced for stable metrics. Keep a CSV of individual predictions for independent audit.

**Do not:** Evaluate FinBERT against labels created by FinBERT and call that accuracy, train on the test stories, compare different test sets, or simply write target values into a results JSON.

# 14. Streaming performance and publication latency proof

## 14.1 Throughput benchmark

**Workload:** At least **3,000** unique nontrivial headline events over **15 minutes** at a target of **200 events/min**, across five publisher IDs using their approximate real sizes and metadata mix. Warm model and Kafka before measurement; also report a cold-start scenario separately. Report ingested, unique accepted, scored, rejected, DLQ, retries, per-publisher shares, and Kafka lag throughout. Use a controlled replay of permissible content or accurately labeled synthetic events to test capacity; identify workload type prominently.

**Pass conditions:** >=200 unique successfully scored articles/min on the sustained rolling window average; no steadily increasing backlog; zero silent data loss; >=99% success under a clean benchmark; per-minute throughput graph and machine/GPU specs saved. For burst tolerance also run a 2-minute 300/min test, but do not replace the sustained benchmark with a burst.

**Important:** `200/min = 3.33 articles/second`. At that rate, 3,000 is a 15-minute workload. Kafka is not automatically the bottleneck; network provider delay, model initialization, database writes and dashboard polling must be individually profiled.

## 14.2 Timestamp instrumentation

| Timestamp | Recorded by | Meaning |
|---|---|---|
| `published_at` | Publisher / trusted provider | Original publication time. |
| `provider_received_at` | Provider if available | When vendor first received event. |
| `connector_received_at` | Our adapter | First moment event entered our infrastructure. |
| `kafka_published_at` | Our producer | Raw event successfully acknowledged by Kafka. |
| `inference_started_at` | Inference worker | Dequeued for scoring. |
| `score_completed_at` | Inference worker | Prediction available and score durably accepted by configured scoring boundary. |
| `dashboard_visible_at` | Browser-side check, optional | First visible display; **not** part of score latency unless separately claimed. |

Compute `publication_to_score_ms = score_completed_at - published_at` **only for trusted timestamps on live, non-backfilled events**. Also report `connector_to_score_ms`, `provider_delivery_lag_ms` and `dashboard_visibility_ms` separately.

## 14.3 Latency target and budget

| Stage | Proposed p95 budget | Dependency / caveat |
|---|---:|---|
| Publisher publication → connector receipt | <=2.0 sec | Requires actual low-latency source feed; cannot be engineered away downstream. |
| Connector normalization → Kafka ack | <=0.3 sec | Async producers, small payloads. |
| Kafka queue → inference start | <=0.4 sec | Sufficient consumers / bounded backlog. |
| FinBERT inference incl. cache lookup | <=1.5 sec | Warm model, microbatch tuning and available CPU/GPU. |
| Persist / finalize score | <=0.5 sec | Batched upsert and bounded database contention. |
| **Budget sum (illustrative)** | **4.7 sec** | Stage p95 values cannot simply be added to *prove* overall p95; measure end-to-end directly. |

**Strict acceptance:** Measured end-to-end p95 **<5,000 ms** during a representative publisher **live-feed** period with adequate sample size and source coverage. Also report p50 and p99, number of eligible observations, missing/incorrect publication timestamps, negative durations and clock skew. Vendor-sourced publication timestamp must be credible (not created on receive). If provider arrival already exceeds five seconds, this target is not feasible for those stories; report actual publisher-to-score and consider changing the resume to “sub-5-second ingestion-to-score latency” **only if that different measure actually passes**.

## 14.4 Dashboard refresh

Use Streamlit's auto-rerunning fragment facility (e.g., `@st.fragment(run_every="1s")`) for active browser sessions or implement a push-enabled display layer. Label the refresh mechanism, as 1-second polling does **not** itself prove 1-second publisher delivery. Ensure query load doesn't affect inference latency. Dashboard freshness is a separately measured UX metric.

# 15. Streamlit dashboard requirements

| Widget / section | Behavior | Data contract |
|---|---|---|
| Headline feed | Newest scored headlines with publisher, ticker, publish time, polarity and probability. | PostgreSQL join of articles/scores filtered by model version. |
| KPI strip | Articles scored last 1/5/60 min; 5-source health; p95 measured latency; queue lag. | Telemetry + rolling time-window aggregation; show N/A if data unavailable. |
| Sentiment distribution | Positive / neutral / negative counts over selected interval. | Counts of top-label predictions, not human truths. |
| Sentiment trend | Line of mean `p_positive - p_negative` per 5-min window, by ticker/source. | Document sample size, handling of neutral/low-volume windows. |
| Ticker / publisher filters | Multi-select ticker/source, UTC or local time display toggle. | Server-side bounded query with indexes. |
| Confidence inspect | Open a story to view 3 probabilities, model checkpoint and score time. | Same row that drove charts. |
| Data quality | Missing publisher times, duplicates, errors and source age. | Monitored counts; never silently treat missing as zero. |
| Mode indicator | Clear LIVE / REPLAY / HISTORICAL indicator. | Ingestion mode field; default no mixed claims. |
| Evidence page | Last benchmark config and human evaluation results. | Read-only generated artifacts; links if locally hosted. |

**Important UX distinction:** Scores are model opinions about the *headline*, not validated trade signals. Include a small disclaimer: “Sentiment analytics for research; not investment advice.”

# 16. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Performance | Benchmark >=200 unique scored events/min for 15 minutes; disclose workload type and machine. |
| NFR-02 | Latency | Live publisher-to-score p95 <5 sec only where provider timestamps and feed support it; otherwise measured outcome and limitation. |
| NFR-03 | Data reliability | At-least-once retries, idempotent database write, DLQ, no lost accepted messages in failure injection. |
| NFR-04 | Recovery | Resume historical collection and scoring from checkpoints without incorrect duplicate counts. |
| NFR-05 | Reproducibility | Lock model/data versions, seed, dependencies, configurations and run IDs. |
| NFR-06 | Security | Secrets in `.env` / secret store, not committed; least-privilege API credentials and DB role. |
| NFR-07 | Licensing | Publisher terms, provenance, retention and model/dataset licenses documented before release. |
| NFR-08 | Observability | Structured logs, per-stage timing, uptime/failures, Kafka lag, cache hits and missing timestamps. |
| NFR-09 | Maintainability | Source connectors isolated behind same interface; lint/tests/typed payload contracts. |
| NFR-10 | Portability | Docker Compose quick start and sample/replay mode with no paid-source credentials. |

# 17. API, configuration and repository contract

## 17.1 Read API (optional FastAPI layer)

- `GET /health/live`, `GET /health/ready` — process/database/model readiness.
- `GET /api/v1/articles?publisher=&ticker=&since=&limit=100` — bounded recent scored articles.
- `GET /api/v1/sentiment/summary?ticker=&window=60m&model_version=` — label distribution and sentiment index.
- `GET /api/v1/metrics/pipeline?window=15m` — measurement-backed ingestion, scoring and latency metrics.
- `GET /api/v1/versions` — active model and schema versions.

All endpoints must validate query bounds; only approved metadata may be exposed externally.

## 17.2 Environment variables

`KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_RAW_TOPIC`, `KAFKA_SCORED_TOPIC`, `DATABASE_URL`, `REDIS_URL`, `MODEL_ID`, `MODEL_REVISION`, `MODEL_MAX_LENGTH`, `BATCH_MAX_SIZE`, `BATCH_WAIT_MS`, `NEWS_SOURCE_*_API_KEY`, `HISTORICAL_START`, `HISTORICAL_END`, `BENCHMARK_MODE`, `METRICS_PORT`. Keep a checked-in `.env.example` with placeholders and no secrets.

## 17.3 Proposed repository tree

```text
financial-news-sentiment/
  README.md                     # setup, limitations, truthful metrics
  docker-compose.yml            # Kafka, Redis, PostgreSQL, services
  .env.example
  config/sources.yaml           # five audited publisher identities
  schemas/raw_article.schema.json
  schemas/scored_article.schema.json
  src/connectors/               # publisher/vendor adapters
  src/normalize/                # parsing, dedup, ticker mapping
  src/stream/                   # Kafka producer and consumer
  src/ml/inference.py           # FinBERT PyTorch backend
  src/ml/cache.py               # versioned Redis cache
  src/storage/                  # PostgreSQL/Parquet persistence
  src/dashboard/app.py          # Streamlit UI
  src/batch/backfill.py         # resuming historical extractor
  src/batch/score_archive.py    # torch batch scoring
  src/training/                 # labels, splits, train, evaluate
  src/api/                      # optional FastAPI read service
  tests/unit/                   # schema, cache and label tests
  tests/integration/            # Kafka + DB + Redis tests
  tests/load/                   # 15-min generator + recorder
  notebooks/                   # exploration only (not core source)
  artifacts/                   # locally generated evidence, non-secret
  docs/data_card.md
  docs/model_card.md
  docs/source_manifest.md
  docs/operations_runbook.md
```

# 18. Test strategy, reproducibility and acceptance suite

**Unit tests:** JSON schema failures, timestamp parsing, FinBERT id2label mapping, probability sum, normalized text hash, exact-id dedup, cache invalidation on model version, deterministic sentiment index, license policy and ticker parser behavior.

**Integration tests:** Publisher mock → Kafka raw → inference → database scored → Kafka scored → Streamlit query. Kill worker between database write and offset commit, restart and verify **one** logical scored DB row. Simulate Redis unavailable (fallback inference) and Postgres down (retries with bounded lag). Corrupt message → DLQ with error reason.

**Data tests:** Raw archive per-source/date coverage, 50,000 unique final IDs, valid labels and probabilities, licensed fields only, no train/test near-duplicate leak, manifest/hash match. Validate that the 50k machine-labeled export is never misrepresented as 50k independently annotated examples.

**Load tests:** 15-minute 200/min rate plus 300/min burst; warm vs cold start; CPU and GPU settings explicitly reported; 0.5x/1x/1.5x rate comparisons if useful. Record P50/P95/P99 model, queue and total delay; no silent count mismatch.

**Latency tests:** Live timestamp-trusted event cohort measured from original publication; separately publish replay delay. A provider lag decomposition should explain any end-to-end failure rather than redefining the target after the fact.

**ML tests:** Locked test ID list, matching baseline/improved predictions, three-class macro-F1, 95% CI for improvement, confusion matrices, source/time slice report, label agreement and fixed-seed rerun.

**Acceptance command suite (illustrative names):**

```bash
cp .env.example .env        # fill approved source credentials if applicable
docker compose up -d --build
python -m src.batch.backfill --config config/backfill.yaml
python -m src.batch.score_archive --dataset data/raw_archive
python -m src.training.create_splits --gold data/gold_labels.csv
python -m src.training.evaluate --baseline <pinned-model> --candidate artifacts/model
python -m tests.load.benchmark --rate-per-min 200 --minutes 15 --mode replay
python -m src.reporting.build_evidence
streamlit run src/dashboard/app.py
```

These are a **specified future CLI interface**, not proof that commands already exist. Implement and test them as project deliverables.

# 19. Definition of done and launch gates

| Gate | Pass standard | Responsible artifact |
|---|---|---|
| Gate A: lawful coverage | Five verified original publisher identities and permitted live/history access. | `docs/source_manifest.md` + credentials checklist (no secrets). |
| Gate B: integrated live path | Kafka → PyTorch FinBERT → PostgreSQL → Streamlit working with replay/retry. | Integration CI report and demo recording. |
| Gate C: scaled processing | Sustained 200/min accepted and scored for 15 min, bounded lag, reproducible hardware. | `artifacts/perf/throughput_report.json` + plot. |
| Gate D: latency | Live real publication-to-score p95 <5 sec, timestamp-qualified sample and provider lag decomposition. | `artifacts/perf/live_latency_report.json`. |
| Gate E: historical | Three-month manifest and exactly 50,000 valid unique **pseudo-labeled** records. | `data/history_scored_50k.parquet` + SHA-256 manifest. |
| Gate F: improved quality | Human gold locked and actual baseline/candidate metrics documented; 0.74→0.84 only if reproduced. | `artifacts/eval/model_comparison.md` + predictions. |
| Gate G: portfolio-ready | Docker Compose, setup/readme, dashboard images, runbook, tests and honest limitations. | README + CI/test report + sanitized screenshots. |

A project can be technically complete while **Gate D or Gate F fails**. In that case disclose the actual metric; a failed numerical target must never be presented as a completed result.

# 20. Milestones and delivery plan

| Phase | Relative timeline | Main work | Exit criterion |
|---|---|---|---|
| P0. Source feasibility | Week 1 | Verify licenses, five publishers, timestamps, historical window and quotas; record provider-delivery baseline. | Source manifest and feasibility go/no-go. |
| P1. Foundation | Week 2 | Compose, Kafka topics, schemas, Postgres/Redis, CI and logging. | All services healthy with raw/scored fixture pass. |
| P2. Live streaming | Weeks 3–4 | Connectors, normalization/dedupe, inference, cache and idempotent writes. | End-to-end test and failure recovery pass. |
| P3. Archive | Week 5 | Resumable historical collection, QA, PyTorch batches and 50k export. | Signed manifest and dataset data card. |
| P4. ML evaluation | Weeks 6–7 | Human label subset, fixed splits, baseline, fine-tuning, evaluation. | Auditable comparison report with actual F1. |
| P5. Dashboard | Week 8 | Streamlit UI, filters, metrics, source health and model metadata. | User acceptance/demo checklist pass. |
| P6. Hardening | Week 9 | Sustained load test, real-feed latency measurement, runbooks, screenshots. | Numeric gates met or documented candidly. |

These are planning estimates rather than guaranteed delivery dates; sourcing and human annotation are the largest schedule uncertainties.

# 21. Risks, mitigations and decisions

| Risk | Severity | Mitigation / decision |
|---|---|---|
| No publisher feed delivers within seconds | Critical | Benchmark provider lag in P0; use true low-latency push access or change wording to measured ingestion-to-score. |
| 200 organic articles/min unavailable | High | Differentiate system replay capacity and actual live volume; no synthetic-news masquerade. |
| Not enough licensed history to reach 50k | High | Expand authorized archive coverage within specified window; otherwise lower recorded result, never fabricate. |
| 50k pseudo labels mistaken for gold labels | Critical | Separate `predicted_label` from `gold_label`; explicit dataset card and gold annotation workflow. |
| F1 target doesn't reproduce | High | Error analysis, improve training-only data and class balance; publish measured numbers, not invented figures. |
| Domain drift / biased labels | Medium | Cross-publisher/time evaluation, human audit, retrain based on monitored drift only. |
| Kafka duplicate effects | Medium | Database composite uniqueness, idempotent upserts, commit-after-persist and restart test. |
| Redis stale scores | Medium | Cache version in key, TTL, invalidate by checkpoint revision. |
| Model/data license restrictions | High | Document license requirements and distribute only permitted weights/data/metadata. |
| Dashboard slow queries | Medium | Indexed DB queries, paging, cached summary aggregates; separate UI from inference worker. |
| FinBERT sarcasm / mixed story / target ambiguity | Medium | Confidence display, honest label rubric and conservative neutral/ambiguous handling. |

# 22. Cost and infrastructure sizing assumptions

**Minimum developer setup:** Docker-capable laptop/VM, 16 GB RAM preferred, SSD storage, Python 3.11+ and enough local disk for Kafka/Postgres/Parquet. A modern GPU is useful but not strictly required; measure CPU throughput rather than assume it meets the SLA. Production-style push-feed subscriptions and historical archive access may incur material vendor fees. Keep GPU type, memory, batch size and model revision in all benchmark outputs.

**Planning estimates:** 50,000 short headlines with metadata are relatively modest in Parquet; exact storage size depends on licensed fields, text lengths and compression. More important operational costs are source licensing, cloud compute, persistent databases and annotation labor. Do not publish invented dollar totals; maintain an itemized budget after provider selection.

# 23. Proof package and interview demonstration

## 23.1 Mandatory evidence folder

```text
artifacts/
  sources/source_manifest.csv          # five actual publishers
  perf/throughput_report.json         # 15-min 200/min proof
  perf/throughput_timeseries.csv
  perf/live_latency_report.json       # only real timestamped live feed
  perf/stage_latency_histogram.csv
  data/history_manifest.json          # dates, 50k count, hashes
  data/quality_report.json
  cache/cold_vs_warm_run.json         # measured cache behavior
  eval/gold_annotation_guide.md
  eval/split_manifest.json
  eval/baseline_predictions.csv
  eval/fine_tuned_predictions.csv
  eval/classification_report.json
  eval/confusion_matrices.png
  eval/paired_bootstrap_ci.json
  ui/live_dashboard.png
  ui/source_health.png
  env/benchmark_hardware.json
```

## 23.2 Recommended five-minute technical demo

**Minute 0–1:** Show architecture and the five audited source identities, including authorization and how publication timestamps are obtained.  
**Minute 1–2:** Stream permitted sample news through Kafka, inspect a raw message, show model-versioned output and Redis hit/miss.  
**Minute 2–3:** Open Streamlit, filter by publisher and ticker, show real probability values and LIVE vs REPLAY indicator.  
**Minute 3–4:** Open sustained throughput/latency report and explain the distinction between publisher lag, connector-to-score and UI refresh.  
**Minute 4–5:** Show three-month/50k dataset validation and human-held-out comparison with per-class scores, source diversity and actual F1 results.

**Likely interview questions and defensible answers:**

- **Why Kafka at only 3.33 items/s?** Kafka demonstrates decoupling, replay, fault recovery, consumer-group scaling and event auditability; not a claim that the raw traffic itself requires Kafka.
- **Why Redis?** Cache helps retried, duplicated and repeatedly backfilled content; a unique first-pass archive will not have an impressive cache hit rate by definition.
- **What does “200/min” mean?** Sustained system scoring capacity under documented benchmark conditions, not necessarily 200 truly new stories continuously published by the five feeds.
- **What is publication-to-score?** Trusted publisher publication timestamp to durable score completion, not the timestamp when the API request was made.
- **Where do 50k labels come from?** They are model-generated prediction labels on historical headlines; human-annotated gold labels are a separate, smaller evaluation dataset.
- **How did F1 improve?** Fine-tune on training-only gold and optionally filtered pseudo labels; select by validation macro-F1; compare on untouched human labels; show real run data.
- **Does high sentiment imply stock gains?** No. It measures headline polarity; financial return forecasting or backtesting is out of scope.

# 24. Resume wording release checklist

The original wording may be retained **only if every corresponding measured requirement is satisfied**. Use the following conservative alternatives where targets are not yet met:

| If evidence shows… | Wording that remains defensible |
|---|---|
| 200/min replay capacity but lower actual live flow | “Benchmarked a Kafka–FinBERT pipeline at 200 articles/min using source-grounded replay; ingested live feeds from [verified count] publishers.” |
| Fast processing after ingestion, slow source delivery | “Achieved p95 ingestion-to-score latency under 5 seconds,” **only if measured**; do not call this publication-to-score. |
| 50k inferred labels, limited manual review | “Generated 50,000 model-labeled financial headline records and evaluated on an independently human-annotated test set.” |
| Actual baseline/candidate macro-F1 not 0.74/0.84 | “Improved held-out macro-F1 from [actual baseline] to [actual improved],” with the test protocol available. |
| Fewer than five licensed publishers | “Integrated [actual count] publisher feeds” or “built five source adapters using replay,” whichever is true. |

**Final sign-off:** Export an immutable evidence manifest with project Git commit, model revisions, dataset hash, benchmark date, hardware, observed throughput, p95 live publication-to-score and measured macro-F1 pair. Only after reviewing that manifest should the numeric claims be used without qualification.

# 25. Authoritative implementation references

1. **FinBERT model card and model metadata:** https://huggingface.co/ProsusAI/finbert — three-class finance sentiment, actual label IDs and model/license information.
2. **ProsusAI/finBERT source repository:** https://github.com/ProsusAI/finBERT — baseline scoring convention and reference code.
3. **Financial PhraseBank dataset card:** https://huggingface.co/datasets/takala/financial_phrasebank — labeled research dataset, 2,264 all-agree items or 4,846 50%-agreement items; it is **not** the proposed 50,000-record historical archive.
4. **Streamlit fragments and `run_every`:** https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment — periodic refresh of dashboard components.
5. **Apache Kafka documentation:** https://kafka.apache.org/documentation/ — partitions, consumer offsets and delivery guarantees; match implementation to the Kafka version used.
6. **Redis documentation:** https://redis.io/docs/latest/ — cache expiration, key semantics and operational configuration.
7. **PyTorch inference mode:** https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html — reduce inference autograd overhead with models in evaluation mode.
8. **scikit-learn F1 documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html — explicitly choose `average="macro"`.

---

**PRD acceptance summary:** A complete build has *both* an online system and a historical/evaluation system. The online system must evidence five true sources, Kafka/FinBERT, scoring capacity, measured publisher latency and Streamlit outputs. The historical system must evidence three months, 50k uniquely scored records, PyTorch + Redis use and human-held-out model improvement. Any number not reproduced is a target or limitation, **not a demonstrated accomplishment**.
