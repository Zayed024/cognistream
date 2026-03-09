"""Tests for backend.db.chroma_store — ChromaDB vector store."""

import pytest

from backend.db.chroma_store import ChromaStore
from backend.db.models import FusedSegment


@pytest.fixture
def store(tmp_path):
    """Create a ChromaStore backed by a temp directory."""
    return ChromaStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        host="",  # embedded mode
    )


def _make_segment(seg_id, video_id="v1", text="test", embedding=None):
    return FusedSegment(
        id=seg_id,
        video_id=video_id,
        start_time=0.0,
        end_time=1.0,
        text=text,
        source_type="fused",
        embedding=embedding or [0.1] * 384,
    )


class TestChromaStoreWrite:
    def test_add_segments(self, store):
        segs = [_make_segment("s1"), _make_segment("s2")]
        stored = store.add_segments(segs)
        assert stored == 2

    def test_skip_segments_without_embedding(self, store):
        seg = _make_segment("s1", embedding=None)
        seg.embedding = None
        stored = store.add_segments([seg])
        assert stored == 0

    def test_upsert_overwrites(self, store):
        seg = _make_segment("s1", text="original")
        store.add_segments([seg])
        seg2 = _make_segment("s1", text="updated")
        store.add_segments([seg2])
        assert store.count() == 1

    def test_empty_list(self, store):
        assert store.add_segments([]) == 0


class TestChromaStoreRead:
    def test_query_returns_results(self, store):
        store.add_segments([_make_segment("s1")])
        results = store.query(embedding=[0.1] * 384, top_k=5)
        assert len(results) >= 1
        assert results[0]["id"] == "s1"
        assert "score" in results[0]

    def test_query_with_video_filter(self, store):
        store.add_segments([
            _make_segment("s1", video_id="v1"),
            _make_segment("s2", video_id="v2"),
        ])
        results = store.query(embedding=[0.1] * 384, top_k=10, video_id="v1")
        assert all(r["video_id"] == "v1" for r in results)

    def test_query_empty_collection(self, store):
        results = store.query(embedding=[0.1] * 384, top_k=5)
        assert results == []

    def test_get_by_video(self, store):
        store.add_segments([
            _make_segment("s1", video_id="v1"),
            _make_segment("s2", video_id="v2"),
        ])
        items = store.get_by_video("v1")
        assert len(items) == 1
        assert items[0]["id"] == "s1"

    def test_count_all(self, store):
        store.add_segments([_make_segment("s1"), _make_segment("s2")])
        assert store.count() == 2

    def test_count_by_video(self, store):
        store.add_segments([
            _make_segment("s1", video_id="v1"),
            _make_segment("s2", video_id="v2"),
        ])
        assert store.count(video_id="v1") == 1


class TestChromaStorePurge:
    def test_purge_video(self, store):
        store.add_segments([
            _make_segment("s1", video_id="v1"),
            _make_segment("s2", video_id="v1"),
            _make_segment("s3", video_id="v2"),
        ])
        deleted = store.purge_video("v1")
        assert deleted == 2
        assert store.count(video_id="v1") == 0
        assert store.count(video_id="v2") == 1

    def test_purge_nonexistent_video(self, store):
        assert store.purge_video("nonexistent") == 0


class TestChromaStoreScoring:
    def test_score_in_range(self, store):
        store.add_segments([_make_segment("s1")])
        results = store.query(embedding=[0.1] * 384, top_k=1)
        assert 0.0 <= results[0]["score"] <= 1.0


class TestBuildWhereFilter:
    def test_no_filters(self):
        assert ChromaStore._build_where_filter(None, None) is None

    def test_video_only(self):
        f = ChromaStore._build_where_filter("v1", None)
        assert f == {"video_id": {"$eq": "v1"}}

    def test_source_only(self):
        f = ChromaStore._build_where_filter(None, "visual")
        assert f == {"source_type": {"$eq": "visual"}}

    def test_both_filters(self):
        f = ChromaStore._build_where_filter("v1", "audio")
        assert "$and" in f
        assert len(f["$and"]) == 2
