import pytest
from src.ml.cache import FinbertRedisCache, compute_headline_hash


def test_cache_key_generation():
    cache = FinbertRedisCache(model_version="rev-1234", preprocess_version="v1")
    headline = "Tesla beats delivery estimates for Q3"
    key = cache.make_key(headline)

    expected_hash = compute_headline_hash(headline)
    expected_key = f"finbert:rev-1234:v1:{expected_hash}"
    assert key == expected_key


def test_cache_hit_and_miss_tracking():
    cache = FinbertRedisCache(model_version="rev-1234", preprocess_version="v1")
    headline = "Boeing receives new aircraft order"

    # Cold lookup -> Miss
    val1 = cache.get(headline)
    assert val1 is None
    assert cache.cache_misses >= 1

    # Populate cache
    score_data = {
        "label": "positive",
        "p_positive": 0.88,
        "p_negative": 0.04,
        "p_neutral": 0.08,
        "sentiment_index": 0.84
    }
    cache.set(headline, score_data)

    # Warm lookup -> Hit
    val2 = cache.get(headline)
    assert val2 is not None
    assert val2["label"] == "positive"
    assert val2["sentiment_index"] == 0.84
    assert cache.cache_hits >= 1


def test_cache_version_isolation():
    cache_v1 = FinbertRedisCache(model_version="checkpoint-A")
    cache_v2 = FinbertRedisCache(model_version="checkpoint-B")

    headline = "Alphabet announces quarterly dividend"
    score_data = {"label": "positive", "p_positive": 0.9}
    cache_v1.set(headline, score_data)

    # Version A has it
    assert cache_v1.get(headline) is not None

    # Version B must miss it (cannot reuse across different model versions per PRD 10.2)
    assert cache_v2.get(headline) is None

