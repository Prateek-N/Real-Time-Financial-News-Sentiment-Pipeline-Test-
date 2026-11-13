import pytest
from src.normalize.cleaner import clean_headline, normalize_text
from src.normalize.dedup import Deduplicator, compute_content_hash
from src.normalize.ticker_mapper import TickerMapper


def test_cleaner_preserves_financial_negation_and_symbols():
    raw = "BREAKING: Company did NOT meet $1.5B revenue &amp; cut guidance by -12%!"
    cleaned = clean_headline(raw)
    # Negation "NOT", currency "$", numbers "1.5B", minus "-12%", unescaped "&"
    assert "not" in cleaned.lower()
    assert "$1.5B" in cleaned
    assert "-12%" in cleaned
    assert "&" in cleaned
    assert not cleaned.startswith("BREAKING:")


def test_deduplicator_exact_and_near_duplicates():
    dedup = Deduplicator()

    # 1. First event accepted
    is_dup1, reason1, c1 = dedup.check_duplicate(
        article_id="art-1",
        publisher_id="reuters",
        headline="Exxon reports higher quarterly profit"
    )
    assert not is_dup1
    assert reason1 == "unique_accepted"

    # 2. Exact article_id duplicate rejected
    is_dup2, reason2, _ = dedup.check_duplicate(
        article_id="art-1",
        publisher_id="reuters",
        headline="Exxon reports higher quarterly profit"
    )
    assert is_dup2
    assert "exact_article_id_duplicate" in reason2

    # 3. Near-duplicate wire syndicated story (flagged cluster but not dropped if ID differs)
    is_dup3, reason3, c3 = dedup.check_duplicate(
        article_id="art-2",
        publisher_id="bloomberg",
        headline="Exxon reports higher quarterly profit"
    )
    assert not is_dup3
    assert c3 == c1  # Same cluster ID allocated


def test_ticker_mapper():
    mapper = TickerMapper()

    # Company name mapping
    assert "NVDA" in mapper.extract_tickers("Nvidia expands next-generation Blackwell architecture")
    assert "AAPL" in mapper.extract_tickers("Apple services growth accelerates in latest quarter")

    # Cashtag mapping
    assert "MSFT" in mapper.extract_tickers("Unusual options activity detected in $MSFT call contracts")

    # Ambiguous or macro headline -> unmapped
    assert mapper.extract_tickers("European central bankers discuss international interest rate trajectories") == ["unmapped"]

