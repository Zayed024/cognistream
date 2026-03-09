"""Tests for keyword extraction (no Whisper model needed)."""

import pytest

from backend.audio.whisper_runner import KeywordExtractor
from backend.db.models import TranscriptSegment


@pytest.fixture
def extractor():
    return KeywordExtractor(top_k=3)


class TestKeywordExtractor:
    def test_basic_extraction(self, extractor):
        segments = [
            TranscriptSegment(0, 5, "machine learning algorithms neural network deep learning"),
            TranscriptSegment(5, 10, "supervised learning classification regression prediction"),
        ]
        result = extractor.extract(segments)
        assert len(result) == 2
        assert len(result[0].keywords) > 0
        assert len(result[0].keywords) <= 3

    def test_empty_segments(self, extractor):
        assert extractor.extract([]) == []

    def test_stop_words_excluded(self, extractor):
        segments = [
            TranscriptSegment(0, 5, "The car is on the street and it is moving fast"),
        ]
        extractor.extract(segments)
        for kw in segments[0].keywords:
            assert kw not in {"the", "is", "on", "and", "it"}

    def test_short_words_excluded(self, extractor):
        segments = [
            TranscriptSegment(0, 5, "AI is ok but ML has a go at NLP"),
        ]
        extractor.extract(segments)
        for kw in segments[0].keywords:
            assert len(kw) >= 3

    def test_single_segment_gets_keywords(self, extractor):
        segments = [
            TranscriptSegment(0, 5, "quantum computing represents fundamental paradigm shift"),
        ]
        extractor.extract(segments)
        assert len(segments[0].keywords) > 0


class TestTokenise:
    def test_removes_punctuation(self):
        tokens = KeywordExtractor._tokenise("Hello, world! This is a test.")
        # "hello" is 5 chars and NOT a stop word, so it should be present
        assert "hello" in tokens
        assert "world" in tokens
        # Stop words should be removed
        assert "this" not in tokens
        assert "test" in tokens

    def test_removes_stop_words(self):
        tokens = KeywordExtractor._tokenise("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens


class TestComputeIdf:
    def test_idf_values(self):
        corpus = [["apple", "banana"], ["apple", "cherry"], ["cherry", "date"]]
        idf = KeywordExtractor._compute_idf(corpus)
        # "apple" appears in 2/3 docs, "banana" in 1/3
        assert idf["banana"] > idf["apple"]
        # "cherry" also appears in 2/3
        assert abs(idf["apple"] - idf["cherry"]) < 0.01

    def test_empty_corpus(self):
        assert KeywordExtractor._compute_idf([]) == {}
