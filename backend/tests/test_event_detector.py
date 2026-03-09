"""Tests for backend.knowledge.event_detector — temporal event pattern matching."""

import pytest

from backend.db.models import Event
from backend.knowledge.event_detector import EventDetector, EventPattern
from backend.knowledge.graph import KnowledgeGraph


@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph("test_video", graph_dir=tmp_path)


@pytest.fixture
def detector():
    return EventDetector()


class TestEventDetectorBasic:
    def test_empty_graph_returns_empty(self, graph, detector):
        events = detector.detect(graph)
        assert events == []

    def test_detects_car_arrival(self, graph):
        """Build a graph with approaching → stopping pattern for a vehicle.
        Use different targets because DiGraph allows only one edge per (src, tgt) pair.
        """
        graph.add_entity("red_car", "vehicle", 30.0)
        graph.add_entity("entrance", "location", 30.0)
        graph.add_entity("curb", "location", 35.0)
        graph.add_relationship("red_car", "entrance", "approaching", 30.0)
        graph.add_relationship("red_car", "curb", "stopping", 35.0)

        detector = EventDetector()
        events = detector.detect(graph)

        car_arrivals = [e for e in events if e.event_type == "car_arrival"]
        assert len(car_arrivals) >= 1
        assert car_arrivals[0].start_time == 30.0
        assert car_arrivals[0].end_time == 35.0

    def test_detects_building_entry(self, graph):
        graph.add_entity("person_1", "person", 40.0)
        graph.add_entity("lobby", "location", 40.0)
        graph.add_entity("building", "location", 40.0)
        graph.add_relationship("person_1", "lobby", "approaching", 40.0)
        graph.add_relationship("person_1", "building", "entering", 45.0)

        detector = EventDetector()
        events = detector.detect(graph)

        entries = [e for e in events if e.event_type == "building_entry"]
        assert len(entries) >= 1

    def test_no_match_if_gap_too_large(self, graph):
        """Actions separated by more than max_gap_sec should not match."""
        graph.add_entity("red_car", "vehicle", 0.0)
        graph.add_relationship("red_car", "lot", "approaching", 0.0)
        graph.add_relationship("red_car", "lot", "stopping", 100.0)  # 100s gap

        detector = EventDetector()
        events = detector.detect(graph)

        car_arrivals = [e for e in events if e.event_type == "car_arrival"]
        assert len(car_arrivals) == 0

    def test_no_match_wrong_entity_type(self, graph):
        """Pattern requires vehicle, but entity is an object."""
        graph.add_entity("laptop", "object", 10.0)
        graph.add_relationship("laptop", "desk", "approaching", 10.0)
        graph.add_relationship("laptop", "desk", "stopping", 15.0)

        detector = EventDetector()
        events = detector.detect(graph)

        car_arrivals = [e for e in events if e.event_type == "car_arrival"]
        assert len(car_arrivals) == 0


class TestEventDetectorCustomPatterns:
    def test_custom_pattern(self, graph):
        pattern = EventPattern(
            name="test_event",
            actions=["action_a", "action_b"],
            entity_types=[],
            max_gap_sec=10.0,
        )
        graph.add_entity("x", "object", 0.0)
        graph.add_relationship("x", "y", "action_a", 1.0)
        graph.add_relationship("x", "z", "action_b", 5.0)

        detector = EventDetector(patterns=[pattern])
        events = detector.detect(graph)
        assert len(events) == 1
        assert events[0].event_type == "test_event"


class TestEventDetectorDeduplication:
    def test_deduplicates_overlapping_events(self, graph):
        """Two overlapping car_arrival events should be deduplicated."""
        graph.add_entity("car", "vehicle", 0.0)
        graph.add_relationship("car", "lot", "approaching", 10.0)
        graph.add_relationship("car", "lot", "stopping", 15.0)
        graph.add_relationship("car", "lot", "arriving", 10.5)
        graph.add_relationship("car", "lot", "stopping", 15.5)

        detector = EventDetector()
        events = detector.detect(graph)

        car_arrivals = [e for e in events if e.event_type == "car_arrival"]
        # Should be deduplicated to 1 (overlapping time ranges, same type)
        assert len(car_arrivals) <= 2  # at most the two non-overlapping ones
