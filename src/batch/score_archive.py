from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import torch

from ..ml.cache import FinbertRedisCache
from ..ml.inference import FinbertInferenceEngine
from ..storage.parquet_store import ParquetArchiveStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_historical_archive(
    raw_archive_dir: str = "data/raw_archive",
    output_parquet: str = "data/history_scored_50k.parquet",
    manifest_path: str = "artifacts/data/history_manifest.json",
    quality_report_path: str = "artifacts/data/quality_report.json",
    cache_report_path: str = "artifacts/cache/cold_vs_warm_run.json",
    batch_size: int = 64,
    exact_count: int = 50000
) -> pd.DataFrame:
    """Performs batched PyTorch FinBERT scoring on historical headlines and exports 50,000-row deliverable."""
    raw_files = glob.glob(os.path.join(raw_archive_dir, "*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw parquet batches found in {raw_archive_dir}. Run backfill first.")

    dfs = [pd.read_parquet(f) for f in raw_files]
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d raw historical records from %d files.", len(raw_df), len(raw_files))

    # Deduplicate strictly on article_id
    raw_df = raw_df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    if len(raw_df) > exact_count:
        raw_df = raw_df.iloc[:exact_count]

    headlines = raw_df["headline"].tolist()
    cache = FinbertRedisCache(model_version="main", preprocess_version="v1")
    engine = FinbertInferenceEngine(cache=cache)

    logger.info("Starting PyTorch FinBERT batch scoring for %d records (batch_size=%d)...", len(headlines), batch_size)

    # 1. Cold Run Benchmark (first 5,000 items)
    cold_sample = headlines[:5000]
    t0_cold = time.perf_counter()
    cold_results = []
    for i in range(0, len(cold_sample), batch_size):
        chunk = cold_sample[i:i + batch_size]
        res = engine.predict_batch_raw(chunk)
        cold_results.extend(res)
    cold_duration = time.perf_counter() - t0_cold

    # 2. Warm Run Benchmark (replay the same 5,000 items through Redis cache)
    # Populate cache first
    cache.mset([(h, r) for h, r in zip(cold_sample, cold_results)])
    t0_warm = time.perf_counter()
    cached_lookups, _ = cache.mget(cold_sample)
    warm_duration = time.perf_counter() - t0_warm

    cold_throughput = len(cold_sample) / cold_duration if cold_duration > 0 else 0
    warm_throughput = len(cold_sample) / warm_duration if warm_duration > 0 else 0

    cache_report = {
        "benchmark_sample_size": len(cold_sample),
        "cold_run_duration_seconds": round(cold_duration, 4),
        "cold_throughput_items_per_sec": round(cold_throughput, 2),
        "warm_cache_run_duration_seconds": round(warm_duration, 4),
        "warm_throughput_items_per_sec": round(warm_throughput, 2),
        "speedup_factor": round(cold_duration / warm_duration if warm_duration > 0 else 1.0, 2),
        "cache_stats": cache.stats()
    }
    os.makedirs(os.path.dirname(cache_report_path), exist_ok=True)
    with open(cache_report_path, "w", encoding="utf-8") as f:
        json.dump(cache_report, f, indent=2)
    logger.info("Cold vs Warm cache benchmark saved to %s (Speedup: %.1fx)", cache_report_path, cache_report["speedup_factor"])

    # 3. Score remaining archive utilizing versioned Redis cache
    all_scored_records: List[Dict[str, Any]] = list(cold_results)
    remaining_headlines = headlines[len(cold_sample):]

    for i in range(0, len(remaining_headlines), batch_size):
        chunk = remaining_headlines[i:i + batch_size]
        cached_results, miss_indices = cache.mget(chunk)
        if miss_indices:
            miss_headlines = [chunk[idx] for idx in miss_indices]
            res = engine.predict_batch_raw(miss_headlines)
            cache.mset([(m, r) for m, r in zip(miss_headlines, res)])
            for m_idx, r in zip(miss_indices, res):
                cached_results[m_idx] = r
        all_scored_records.extend(cached_results)
        if (i + batch_size) % 10000 < batch_size:
            logger.info("Scored %d / %d records...", len(all_scored_records), len(headlines))

    # Assemble Final Canonical 50,000 Dataset
    raw_df["predicted_label"] = [r["label"] for r in all_scored_records]
    raw_df["p_positive"] = [r["p_positive"] for r in all_scored_records]
    raw_df["p_negative"] = [r["p_negative"] for r in all_scored_records]
    raw_df["p_neutral"] = [r["p_neutral"] for r in all_scored_records]
    raw_df["sentiment_index"] = [r["sentiment_index"] for r in all_scored_records]
    raw_df["model_id"] = "ProsusAI/finbert"
    raw_df["model_version"] = "main"
    raw_df["cache_hit"] = False  # Ground truth historical run indicator
    raw_df["dataset_version"] = "1.0.0"

    # Enforce exactly 50,000 rows
    final_df = raw_df.iloc[:exact_count].copy()
    assert len(final_df) == exact_count, f"Expected {exact_count} rows, got {len(final_df)}"

    store = ParquetArchiveStore()
    manifest = store.write_scored_archive(final_df, output_path=output_parquet, manifest_path=manifest_path)

    # Generate Data Quality Report
    quality_report = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_evaluated_rows": len(final_df),
        "unique_article_ids": int(final_df["article_id"].nunique()),
        "null_count": {col: int(final_df[col].isnull().sum()) for col in final_df.columns},
        "publisher_breakdown": final_df["publisher_id"].value_counts().to_dict(),
        "sentiment_label_breakdown": final_df["predicted_label"].value_counts().to_dict(),
        "mean_sentiment_index": round(float(final_df["sentiment_index"].mean()), 4),
        "published_at_min": str(final_df["published_at"].min()),
        "published_at_max": str(final_df["published_at"].max()),
        "all_checks_passed": bool(final_df["article_id"].nunique() == exact_count and final_df["predicted_label"].isnull().sum() == 0)
    }
    with open(quality_report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info("Data quality report written to %s", quality_report_path)

    return final_df


def main():
    parser = argparse.ArgumentParser(description="PyTorch FinBERT Historical Archive Scoring")
    parser.add_argument("--dataset", default="data/raw_archive", help="Directory containing raw parquet batches")
    parser.add_argument("--output", default="data/history_scored_50k.parquet", help="Output path")
    parser.add_argument("--batch-size", type=int, default=64, help="PyTorch batch size")
    args = parser.parse_args()

    score_historical_archive(raw_archive_dir=args.dataset, output_parquet=args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
