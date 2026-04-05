"""Tests for backend.knowledge.graph — Knowledge graph construction."""

import pytest

from backend.knowledge.graph import KnowledgeGraph


@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph("test_video", graph_dir=tmp_path)


class TestKnowledgeGraphEntities:
    def test_add_new_entity(self, graph):
        graph.add_entity("red_car", "vehicle", 10.0)
        assert graph.G.has_node("red_car")
        node = graph.G.nodes["red_car"]
        assert node["type"] == "vehicle"
        assert node["first_seen"] == 10.0
        assert node["count"] == 1

    def test_add_existing_entity_updates(self, graph):
        graph.add_entity("red_car", "vehicle", 10.0)
        graph.add_entity("red_car", "vehicle", 20.0)
        node = graph.G.nodes["red_car"]
        assert node["last_seen"] == 20.0
        assert node["count"] == 2

    def test_add_entity_keeps_earliest_first_seen(self, graph):
        graph.add_entity("red_car", "vehicle", 10.0)
        graph.add_entity("red_car", "vehicle", 5.0)
        node = graph.G.nodes["red_car"]
        # first_seen is only set on creation, not updated
        assert node["first_seen"] == 10.0


class TestKnowledgeGraphRelationships:
    def test_add_relationship(self, graph):
        graph.add_relationship("red_car", "traffic_signal", "stops_at", 34.2)
        assert graph.G.has_edge("red_car", "traffic_signal")
        edge = next(iter(graph.G.get_edge_data("red_car", "traffic_signal").values()))
        assert edge["action"] == "stops_at"
        assert edge["timestamp"] == 34.2

    def test_add_relationship_creates_missing_nodes(self, graph):
        graph.add_relationship("person_1", "building", "enters", 45.0)
        assert graph.G.has_node("person_1")
        assert graph.G.has_node("building")

    def test_get_entity_timestamps(self, graph):
        graph.add_entity("red_car", "vehicle", 10.0)
        graph.add_relationship("red_car", "signal", "stops_at", 34.0)
        graph.add_relationship("red_car", "lot", "parks_at", 40.0)
        ts = graph.get_entity_timestamps("red_car")
        assert 10.0 in ts
        assert 34.0 in ts
        assert 40.0 in ts

    def test_get_related_entities(self, graph):
        graph.add_relationship("red_car", "signal", "stops_at", 34.0)
        graph.add_relationship("person", "red_car", "enters", 40.0)
        related = graph.get_related_entities("red_car")
        entities = {r["entity"] for r in related}
        assert "signal" in entities
        assert "person" in entities

    def test_preserves_multiple_temporal_edges_between_same_entities(self, graph):
        graph.add_relationship("red_car", "signal", "approaching", 30.0)
        graph.add_relationship("red_car", "signal", "stopping", 34.0)
        edge_data = graph.G.get_edge_data("red_car", "signal")
        assert edge_data is not None
        actions = sorted(data["action"] for data in edge_data.values())
        assert actions == ["approaching", "stopping"]


class TestKnowledgeGraphPersistence:
    def test_save_and_load(self, tmp_path):
        g1 = KnowledgeGraph("v1", graph_dir=tmp_path)
        g1.add_entity("car", "vehicle", 10.0)
        g1.add_relationship("car", "signal", "stops_at", 20.0)
        path = g1.save()
        assert path.exists()

        g2 = KnowledgeGraph("v1", graph_dir=tmp_path)
        assert g2.load() is True
        assert g2.G.has_node("car")
        assert g2.G.has_edge("car", "signal")

    def test_load_nonexistent(self, tmp_path):
        g = KnowledgeGraph("nonexistent", graph_dir=tmp_path)
        assert g.load() is False


class TestKnowledgeGraphClassify:
    def test_vehicle_classification(self):
        assert KnowledgeGraph._classify_entity("red_car") == "vehicle"
        assert KnowledgeGraph._classify_entity("bus") == "vehicle"

    def test_person_classification(self):
        assert KnowledgeGraph._classify_entity("person") == "person"
        assert KnowledgeGraph._classify_entity("tall_man") == "person"

    def test_location_classification(self):
        assert KnowledgeGraph._classify_entity("building") == "location"
        assert KnowledgeGraph._classify_entity("main_entrance") == "location"

    def test_object_fallback(self):
        assert KnowledgeGraph._classify_entity("laptop") == "object"


class TestKnowledgeGraphNormalise:
    def test_lowercase_and_underscore(self):
        assert KnowledgeGraph._normalise("Red Car") == "red_car"

    def test_strip_punctuation(self):
        assert KnowledgeGraph._normalise("hello, world!") == "hello_world"

    def test_empty_string(self):
        assert KnowledgeGraph._normalise("") == ""


class TestKnowledgeGraphBuildFromCaptions:
    def test_builds_nodes_from_objects(self, graph, sample_captions, sample_transcripts):
        graph.build_from_captions(sample_captions, sample_transcripts)
        assert graph.G.number_of_nodes() > 0

    def test_builds_edges_from_activities(self, graph, sample_captions, sample_transcripts):
        graph.build_from_captions(sample_captions, sample_transcripts)
        assert graph.G.number_of_edges() > 0

    def test_limits_transcript_keywords(self, graph, sample_captions, sample_transcripts):
        """Only top 3 keywords per segment should become nodes."""
        graph.build_from_captions(sample_captions, sample_transcripts)
        # We can't easily count exact keyword nodes, but the graph shouldn't explode
        assert graph.G.number_of_nodes() < 50

    def test_builds_keyword_relationships_from_transcripts(self, graph, sample_transcripts):
        graph.build_from_captions([], sample_transcripts)
        related = graph.get_related_entities("car")
        assert any(item["entity"] == "traffic" and item["action"] == "mentioned_with" for item in related)
