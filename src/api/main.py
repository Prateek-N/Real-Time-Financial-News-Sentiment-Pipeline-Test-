from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..storage.db import DatabaseManager

app = FastAPI(
    title="Financial News Sentiment Pipeline Read API",
    version="1.0.0",
    description="High-performance read layer for articles, sentiment scores, and telemetry metrics."
)

db = DatabaseManager()


@app.get("/health/live")
def health_live():
    return {"status": "alive", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/health/ready")
def health_ready():
    # Verify DB responsiveness
    try:
        count = db.count_articles()
        return {"status": "ready", "articles_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unready: {e}")


@app.get("/api/v1/articles")
def get_articles(
    publisher: Optional[str] = Query(None, description="Filter by publisher identity"),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    limit: int = Query(50, ge=1, le=500),
    model_version: str = "main"
):
    rows = db.query_recent_scored_articles(
        publisher_id=publisher,
        ticker=ticker,
        limit=limit,
        model_version=model_version
    )
    return {"count": len(rows), "articles": rows}


@app.get("/api/v1/sentiment/summary")
def get_sentiment_summary(
    ticker: Optional[str] = Query(None),
    model_version: str = "main"
):
    rows = db.query_recent_scored_articles(ticker=ticker, limit=500, model_version=model_version)
    if not rows:
        return {"ticker": ticker, "count": 0, "mean_sentiment_index": 0.0, "label_counts": {}}

    indices = [r["sentiment_index"] for r in rows]
    labels = [r["label"] for r in rows]

    counts = {"positive": labels.count("positive"), "negative": labels.count("negative"), "neutral": labels.count("neutral")}
    mean_idx = sum(indices) / len(indices) if indices else 0.0

    return {
        "ticker": ticker or "All",
        "sample_size": len(rows),
        "mean_sentiment_index": round(mean_idx, 4),
        "label_counts": counts
    }


@app.get("/api/v1/metrics/pipeline")
def get_pipeline_metrics(window: str = "15m"):
    perf_path = "artifacts/perf/throughput_report.json"
    if os.path.exists(perf_path):
        import json
        with open(perf_path, "r") as f:
            return json.load(f)
    return {
        "status": "pending_benchmark",
        "message": "Run benchmark to populate real-time metrics."
    }


@app.get("/api/v1/versions")
def get_versions():
    return {
        "schema_version": "1.0",
        "model_id": "ProsusAI/finbert",
        "model_version": "main",
        "preprocess_version": "v1",
        "api_version": "1.0.0"
    }

