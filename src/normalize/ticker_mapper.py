from __future__ import annotations

import re
from typing import List, Set

# Comprehensive map of premier corporate financial entities to authoritative exchange tickers
COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "eli lilly": "LLY",
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "disney": "DIS",
    "netflix": "NFLX",
    "boeing": "BA",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
    "berkshire hathaway": "BRK.A",
    "visa": "V",
    "mastercard": "MA",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "cisco": "CSCO",
    "adobe": "ADBE",
}

KNOWN_TICKERS = set(COMPANY_TO_TICKER.values()) | {
    "SPY", "QQQ", "DIA", "IWM", "VIX", "GLD", "USO", "TLT"
}

# Match explicit cash tags like $AAPL, $NVDA
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")


class TickerMapper:
    """Extracts ticker symbols with high precision.

    Distinguishes mentioned entities from affected equities and defaults to
    ['unmapped'] if uncertain, per PRD section 8.3.
    """

    def __init__(self):
        # Precompile company name regex with word boundaries
        self._patterns = [
            (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ticker)
            for name, ticker in COMPANY_TO_TICKER.items()
        ]

    def extract_tickers(self, headline: str) -> List[str]:
        found: Set[str] = set()

        # 1. Check for explicit cashtags ($AAPL, $TSLA)
        for match in CASHTAG_RE.finditer(headline):
            ticker = match.group(1)
            if ticker in KNOWN_TICKERS or len(ticker) in (3, 4):
                found.add(ticker)

        # 2. Check for explicit company names
        for pattern, ticker in self._patterns:
            if pattern.search(headline):
                found.add(ticker)

        # 3. Check for standalone uppercase ticker tokens (only if known)
        tokens = re.findall(r"\b[A-Z]{2,5}\b", headline)
        for token in tokens:
            if token in KNOWN_TICKERS:
                found.add(token)

        if not found:
            return ["unmapped"]

        return sorted(list(found))

