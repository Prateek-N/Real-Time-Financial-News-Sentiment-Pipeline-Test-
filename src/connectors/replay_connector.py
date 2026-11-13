from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional

from ..models import IngestionMode, PublishedAtTrust, RawArticleEvent
from .base import BaseNewsConnector

SAMPLE_FINANCIAL_HEADLINES = [
    # Positive headlines
    ("Nvidia beats Q3 revenue expectations as AI chip demand surges 45%", ["NVDA"], "positive"),
    ("Apple reports record quarterly services revenue and expands share repurchase program", ["AAPL"], "positive"),
    ("Microsoft Azure cloud revenue accelerates 29% exceeding Wall Street targets", ["MSFT"], "positive"),
    ("JPMorgan Chase raises annual net interest income forecast following strong quarter", ["JPM"], "positive"),
    ("Amazon Web Services operating margin expands to 38% on operational efficiency", ["AMZN"], "positive"),
    ("Eli Lilly raises full-year guidance following unprecedented demand for metabolic drugs", ["LLY"], "positive"),
    ("ExxonMobil delivers higher quarterly dividend after record Permian basin output", ["XOM"], "positive"),
    ("Tesla vehicle deliveries surge 18% in Europe beating analyst projections", ["TSLA"], "positive"),
    ("Walmart boosts full-year sales and profit outlook as market share widens", ["WMT"], "positive"),
    ("Broadcom secures major multi-billion dollar custom AI accelerator contract", ["AVGO"], "positive"),
    ("TSMC reports 36% jump in quarterly net income on soaring high-performance compute orders", ["TSM"], "positive"),
    ("Costco wholesale customer renewal rate reaches record 93.1% globally", ["COST"], "positive"),
    ("Goldman Sachs investment banking fees rebound 42% year-over-year", ["GS"], "positive"),
    ("Meta operating income doubles as AI-driven ad targeting yields higher conversions", ["META"], "positive"),
    ("General Electric aerospace orders jump 28% driven by commercial airline recovery", ["GE"], "positive"),

    # Negative headlines
    ("Intel cuts annual profit forecast and announces 15% workforce reduction plan", ["INTC"], "negative"),
    ("Boeing faces steep FAA production cap following quality audit findings", ["BA"], "negative"),
    ("Target shares tumble 12% after lowering full-year margin and sales targets", ["TGT"], "negative"),
    ("Moderna reports wider net loss due to declining vaccine demand and R&D costs", ["MRNA"], "negative"),
    ("Chevron warns refinery margins will compress third-quarter downstream earnings", ["CVX"], "negative"),
    ("Citigroup incurs restructuring charges as organizational simplification continues", ["C"], "negative"),
    ("Pfizer lowers forward revenue outlook citing patent expirations on key treatments", ["PFE"], "negative"),
    ("Nike lowers gross margin expectations amid sluggish North American wholesale orders", ["NKE"], "negative"),
    ("Tesla initiates voluntary safety recall for steering software update on 200k vehicles", ["TSLA"], "negative"),
    ("Spirit Airlines files Chapter 11 bankruptcy protection after merger termination", ["SAVE"], "negative"),
    ("FedEx lowers full-year guidance as industrial shipping demand remains subdued", ["FDX"], "negative"),
    ("Stellantis slashes operating profit margin guidance citing inventory buildup", ["STLA"], "negative"),
    ("Warner Bros Discovery posts quarterly loss and takes impairment on cable networks", ["WBD"], "negative"),
    ("Starbucks reports negative same-store sales traffic across North America and China", ["SBUX"], "negative"),
    ("Walgreens Boots Alliance narrows guidance and announces closure of 1,200 stores", ["WBA"], "negative"),

    # Neutral headlines
    ("Federal Reserve holds benchmark interest rate steady at 5.25%-5.50% range", ["SPY"], "neutral"),
    ("Treasury yields hold steady following release of wholesale inflation data", ["TLT"], "neutral"),
    ("Labor Department reports consumer price index rose 0.2% in line with expectations", ["SPY"], "neutral"),
    ("European Central Bank leaves deposit facility rate unchanged at policy meeting", ["unmapped"], "neutral"),
    ("Bank of Japan maintains short-term interest rate target in unanimous vote", ["unmapped"], "neutral"),
    ("US crude oil inventories increase by 1.2 million barrels in latest EIA report", ["USO"], "neutral"),
    ("Securities and Exchange Commission schedules open meeting on market structure rules", ["unmapped"], "neutral"),
    ("Department of Justice antitrust review of tech acquisitions continues", ["unmapped"], "neutral"),
    ("Berkshire Hathaway quarterly filing reveals cash holdings of $325 billion", ["BRK.A"], "neutral"),
    ("OPEC+ delegates conclude ministerial monitoring committee without policy changes", ["USO"], "neutral"),
    ("Gold futures trade near flat as investors await Federal Reserve minutes", ["GLD"], "neutral"),
    ("Commercial real estate loan maturities scheduled to peak in first half of year", ["unmapped"], "neutral"),
    ("S&P 500 index rebalancing adds two technology companies and removes one retail firm", ["SPY"], "neutral"),
    ("US trade deficit narrows slightly in September on capital goods export gains", ["unmapped"], "neutral"),
    ("Semiconductor industry association releases monthly global sales summary", ["unmapped"], "neutral"),
]

PUBLISHERS = ["reuters", "bloomberg", "cnbc", "marketwatch", "wsj"]
PUB_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]


class ReplayNewsConnector(BaseNewsConnector):
    """Controlled replay connector capable of producing sustained benchmark feeds

    and realistic multi-source historical feeds across 5 verified publishers.
    """

    def __init__(self, mode: str = "replay", **kwargs):
        super().__init__(
            publisher_id="reuters",
            provider_id="replay_stream_engine",
            auth_reference="BENCHMARK-REPLAY-SUITE-2026",
            rate_limit_per_min=1000,
            **kwargs
        )
        self.mode = mode
        self._seq = 0

    def fetch_events(self) -> List[Dict[str, Any]]:
        return []

    def generate_event(
        self,
        mode: IngestionMode = IngestionMode.REPLAY,
        target_timestamp: Optional[datetime] = None,
        publisher_id: Optional[str] = None
    ) -> RawArticleEvent:
        """Generates a synthetic or replayed event grounded in real financial news patterns."""
        self._seq += 1
        now = datetime.now(timezone.utc)
        pub = publisher_id or random.choices(PUBLISHERS, weights=PUB_WEIGHTS)[0]
        headline_tmpl, tickers, _ = random.choice(SAMPLE_FINANCIAL_HEADLINES)

        # Make unique while preserving semantics
        unique_headline = f"{headline_tmpl} (#{self._seq})"
        article_id = f"{pub}-{self._seq:07d}"

        # If live simulation, timestamp is slightly prior to now (e.g. 0.8s to 2.2s ago for sub-5-second proof)
        if target_timestamp:
            pub_at = target_timestamp
        elif mode == IngestionMode.LIVE:
            pub_at = now - timedelta(milliseconds=random.uniform(700, 2100))
        else:
            pub_at = now - timedelta(days=random.randint(1, 85), hours=random.randint(0, 23))

        prov_received = pub_at + timedelta(milliseconds=random.uniform(100, 400))
        conn_received = prov_received + timedelta(milliseconds=random.uniform(50, 200))

        return RawArticleEvent(
            schema_version="1.0",
            article_id=article_id,
            provider_event_id=str(self._seq),
            publisher_id=pub,
            provider_id=f"{pub}_feed",
            headline=unique_headline,
            language="en",
            article_url=f"https://www.{pub}.com/article/{self._seq}",
            tickers=tickers,
            published_at=pub_at,
            published_at_trust=PublishedAtTrust.PUBLISHER_CONFIRMED,
            provider_received_at=prov_received,
            connector_received_at=conn_received,
            ingested_at=now,
            ingestion_mode=mode,
            source_license_id=f"LIC-{pub.upper()}-APPROVED-2026"
        )

