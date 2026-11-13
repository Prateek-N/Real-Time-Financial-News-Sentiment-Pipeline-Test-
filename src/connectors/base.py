from __future__ import annotations

import abc
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import PublishedAtTrust, RawArticleEvent
from ..normalize.cleaner import clean_headline
from ..normalize.dedup import Deduplicator
from ..normalize.ticker_mapper import TickerMapper


class BaseNewsConnector(abc.ABC):
    """Abstract connector interface for publisher integrations.

    Maintains source health, rate limiting, heartbeat, and provenance.
    """

    def __init__(
        self,
        publisher_id: str,
        provider_id: str,
        auth_reference: str,
        rate_limit_per_min: int = 100,
        deduplicator: Optional[Deduplicator] = None,
        ticker_mapper: Optional[TickerMapper] = None,
    ):
        self.publisher_id = publisher_id
        self.provider_id = provider_id
        self.auth_reference = auth_reference
        self.rate_limit_per_min = rate_limit_per_min
        self.deduplicator = deduplicator or Deduplicator()
        self.ticker_mapper = ticker_mapper or TickerMapper()

        self.last_heartbeat_ts: float = time.time()
        self.total_fetched: int = 0
        self.total_accepted: int = 0
        self.total_rejected_duplicates: int = 0
        self.is_connected: bool = True

    @abc.abstractmethod
    def fetch_events(self) -> List[Dict[str, Any]]:
        """Fetch raw payload events from provider push/poll feed."""
        pass

    def normalize_and_validate(self, raw_item: Dict[str, Any], mode: str = "live") -> Optional[RawArticleEvent]:
        """Validates and normalizes raw item into canonical RawArticleEvent."""
        now_utc = datetime.now(timezone.utc)
        self.last_heartbeat_ts = time.time()
        self.total_fetched += 1

        headline_raw = raw_item.get("headline") or raw_item.get("title") or ""
        cleaned = clean_headline(headline_raw)
        if len(cleaned) < 5:
            return None

        event_id = str(raw_item.get("provider_event_id") or raw_item.get("id") or "")
        article_url = raw_item.get("article_url") or raw_item.get("url")

        # Check duplicate
        is_dup, reason, cluster_id = self.deduplicator.check_duplicate(
            article_id=f"{self.publisher_id}-{event_id}" if event_id else f"{self.publisher_id}-{hash(cleaned)}",
            publisher_id=self.publisher_id,
            headline=cleaned,
            article_url=article_url
        )
        if is_dup:
            self.total_rejected_duplicates += 1
            return None

        tickers = self.ticker_mapper.extract_tickers(cleaned)

        pub_at = raw_item.get("published_at")
        pub_trust = PublishedAtTrust.PUBLISHER_CONFIRMED if pub_at else PublishedAtTrust.MISSING_EXCLUDED

        event = RawArticleEvent(
            article_id=f"{self.publisher_id}-{event_id}" if event_id else f"{self.publisher_id}-{hash(cleaned)}",
            provider_event_id=event_id,
            publisher_id=self.publisher_id,
            provider_id=self.provider_id,
            headline=cleaned,
            language=raw_item.get("language", "en"),
            article_url=article_url,
            tickers=tickers,
            published_at=pub_at if isinstance(pub_at, datetime) else now_utc,
            published_at_trust=pub_trust,
            provider_received_at=raw_item.get("provider_received_at", now_utc),
            connector_received_at=now_utc,
            ingested_at=now_utc,
            ingestion_mode=mode,
            source_license_id=self.auth_reference
        )
        self.total_accepted += 1
        return event

    def health(self) -> Dict[str, Any]:
        """Returns connector health and telemetry metrics."""
        age_seconds = time.time() - self.last_heartbeat_ts
        return {
            "publisher_id": self.publisher_id,
            "provider_id": self.provider_id,
            "is_connected": self.is_connected and age_seconds < 120,
            "last_seen_seconds_ago": round(age_seconds, 1),
            "total_fetched": self.total_fetched,
            "total_accepted": self.total_accepted,
            "duplicates_dropped": self.total_rejected_duplicates,
        }

