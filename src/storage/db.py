from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models import RawArticleEvent, ScoredArticleEvent

logger = logging.getLogger(__name__)

PG_INIT_SQL = """
CREATE TABLE IF NOT EXISTS articles (
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

CREATE TABLE IF NOT EXISTS sentiment_scores (
    article_id TEXT NOT NULL REFERENCES articles(article_id),
    model_version TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positive','negative','neutral')),
    p_positive DOUBLE PRECISION NOT NULL,
    p_negative DOUBLE PRECISION NOT NULL,
    p_neutral DOUBLE PRECISION NOT NULL,
    sentiment_index DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    score_completed_at TIMESTAMPTZ NOT NULL,
    cache_hit BOOLEAN NOT NULL,
    PRIMARY KEY (article_id, model_version, preprocess_version)
);

CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS ix_articles_publisher_date ON articles (publisher_id, published_at DESC);
"""

SQLITE_INIT_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    publisher_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_event_id TEXT,
    headline TEXT NOT NULL,
    article_url TEXT,
    published_at TEXT,
    published_at_trust TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    ingestion_mode TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    tickers TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    article_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    label TEXT NOT NULL,
    p_positive REAL NOT NULL,
    p_negative REAL NOT NULL,
    p_neutral REAL NOT NULL,
    sentiment_index REAL NOT NULL DEFAULT 0.0,
    score_completed_at TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    PRIMARY KEY (article_id, model_version, preprocess_version),
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);

CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS ix_articles_publisher_date ON articles (publisher_id, published_at DESC);
"""


class DatabaseManager:
    """Manages PostgreSQL persistence with automatic fallback to SQLite.

    Ensures idempotent upserts and fast queries for Streamlit and analytics.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/financial_news")
        self.is_sqlite = False
        self.sqlite_path = "data/financial_news.db"
        self._pg_conn = None
        self._init_connection()

    def _init_connection(self):
        os.makedirs("data", exist_ok=True)
        if self.db_url.startswith("postgresql://") or self.db_url.startswith("postgres://"):
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url, connect_timeout=2)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(PG_INIT_SQL)
                self._pg_conn = conn
                self.is_sqlite = False
                logger.info("Connected to PostgreSQL successfully.")
                return
            except Exception as e:
                logger.warning("PostgreSQL unreachable (%s). Falling back to SQLite at %s", e, self.sqlite_path)

        # Fallback SQLite
        self.is_sqlite = True
        conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SQLITE_INIT_SQL)
        self._sqlite_conn = conn
        logger.info("Initialized local SQLite database at %s", self.sqlite_path)

    def upsert_article(self, article: RawArticleEvent):
        """Idempotently insert or update an article record."""
        pub_at_str = article.published_at.isoformat() if article.published_at else None
        ing_at_str = article.ingested_at.isoformat()

        if self.is_sqlite:
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO articles (
                        article_id, publisher_id, provider_id, provider_event_id,
                        headline, article_url, published_at, published_at_trust,
                        ingested_at, ingestion_mode, content_sha256, tickers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_id) DO UPDATE SET
                        headline=excluded.headline,
                        tickers=excluded.tickers,
                        ingested_at=excluded.ingested_at
                    """,
                    (
                        article.article_id,
                        article.publisher_id,
                        article.provider_id,
                        article.provider_event_id,
                        article.headline,
                        article.article_url,
                        pub_at_str,
                        article.published_at_trust.value,
                        ing_at_str,
                        article.ingestion_mode.value,
                        article.content_sha256,
                        json.dumps(article.tickers),
                    ),
                )
        else:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO articles (
                        article_id, publisher_id, provider_id, provider_event_id,
                        headline, article_url, published_at, published_at_trust,
                        ingested_at, ingestion_mode, content_sha256, tickers
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                        headline = EXCLUDED.headline,
                        tickers = EXCLUDED.tickers,
                        ingested_at = EXCLUDED.ingested_at
                    """,
                    (
                        article.article_id,
                        article.publisher_id,
                        article.provider_id,
                        article.provider_event_id,
                        article.headline,
                        article.article_url,
                        article.published_at,
                        article.published_at_trust.value,
                        article.ingested_at,
                        article.ingestion_mode.value,
                        article.content_sha256,
                        article.tickers,
                    ),
                )

    def upsert_score(self, score: ScoredArticleEvent):
        """Idempotently insert or update a sentiment score."""
        sc_at_str = score.score_completed_at.isoformat()

        if self.is_sqlite:
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO sentiment_scores (
                        article_id, model_version, preprocess_version,
                        label, p_positive, p_negative, p_neutral, sentiment_index,
                        score_completed_at, cache_hit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_id, model_version, preprocess_version) DO UPDATE SET
                        label=excluded.label,
                        p_positive=excluded.p_positive,
                        p_negative=excluded.p_negative,
                        p_neutral=excluded.p_neutral,
                        sentiment_index=excluded.sentiment_index,
                        score_completed_at=excluded.score_completed_at,
                        cache_hit=excluded.cache_hit
                    """,
                    (
                        score.article_id,
                        score.model_version,
                        score.preprocess_version,
                        score.label.value,
                        score.p_positive,
                        score.p_negative,
                        score.p_neutral,
                        score.sentiment_index,
                        sc_at_str,
                        1 if score.cache_hit else 0,
                    ),
                )
        else:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sentiment_scores (
                        article_id, model_version, preprocess_version,
                        label, p_positive, p_negative, p_neutral, sentiment_index,
                        score_completed_at, cache_hit
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, model_version, preprocess_version) DO UPDATE SET
                        label = EXCLUDED.label,
                        p_positive = EXCLUDED.p_positive,
                        p_negative = EXCLUDED.p_negative,
                        p_neutral = EXCLUDED.p_neutral,
                        sentiment_index = EXCLUDED.sentiment_index,
                        score_completed_at = EXCLUDED.score_completed_at,
                        cache_hit = EXCLUDED.cache_hit
                    """,
                    (
                        score.article_id,
                        score.model_version,
                        score.preprocess_version,
                        score.label.value,
                        score.p_positive,
                        score.p_negative,
                        score.p_neutral,
                        score.sentiment_index,
                        score.score_completed_at,
                        score.cache_hit,
                    ),
                )

    def query_recent_scored_articles(
        self,
        publisher_id: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        model_version: str = "main"
    ) -> List[Dict[str, Any]]:
        """Query joined article + score feed for Streamlit dashboard and API."""
        query = """
            SELECT 
                a.article_id, a.publisher_id, a.provider_id, a.headline, a.article_url,
                a.published_at, a.published_at_trust, a.ingested_at, a.ingestion_mode,
                a.tickers, s.label, s.p_positive, s.p_negative, s.p_neutral,
                s.sentiment_index, s.score_completed_at, s.cache_hit, s.model_version
            FROM articles a
            JOIN sentiment_scores s ON a.article_id = s.article_id
            WHERE s.model_version = ?
        """
        params: List[Any] = [model_version]

        if publisher_id and publisher_id != "All":
            query += " AND a.publisher_id = ?"
            params.append(publisher_id)

        if ticker and ticker != "All":
            query += " AND a.tickers LIKE ?"
            params.append(f"%{ticker}%")

        query += " ORDER BY s.score_completed_at DESC LIMIT ?"
        params.append(limit)

        if self.is_sqlite:
            cur = self._sqlite_conn.cursor()
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if isinstance(r.get("tickers"), str):
                    try:
                        r["tickers"] = json.loads(r["tickers"])
                    except Exception:
                        r["tickers"] = [r["tickers"]]
            return rows
        else:
            pg_query = query.replace("?", "%s")
            with self._pg_conn.cursor() as cur:
                cur.execute(pg_query, params)
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_articles(self) -> int:
        query = "SELECT COUNT(*) FROM articles"
        if self.is_sqlite:
            return self._sqlite_conn.execute(query).fetchone()[0]
        with self._pg_conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchone()[0]

