"""
CogniStream — Temporal Knowledge Graph

Builds a directed graph of entities and their relationships from
visual captions and transcripts.  Each video gets its own graph
serialised to GraphML on disk.

Nodes represent entities (person, vehicle, object, location).
Edges represent actions with temporal attributes.

Example:
    red_car  --stops_at[t=34.2]--> traffic_signal
    person_1 --enters[t=45.0]-->   building

The graph enables:
    - Entity-based queries ("find all segments involving person_1")
    - Relationship traversal ("what happened after the car stopped?")
    - Event detection (sequences of edges → higher-level events)

Usage:
    graph = KnowledgeGraph(video_id)
    graph.add_entity("red_car", "vehicle", timestamp=34.0)
    graph.add_relationship("red_car", "traffic_signal", "stops_at", timestamp=34.2)
    graph.save()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from itertools import combinations
from typing import Optional

import networkx as nx

from backend.config import GRAPH_DIR
from backend.db.models import VisualCaption, TranscriptSegment

logger = logging.getLogger(__name__)

# Entity types we attempt to extract
_ENTITY_TYPES = {"person", "vehicle", "object", "location"}

# Simple patterns to classify detected objects into entity types
_VEHICLE_WORDS = {"car", "truck", "bus", "van", "bicycle", "motorcycle", "vehicle", "sedan", "suv"}
_PERSON_WORDS = {"person", "man", "woman", "child", "people", "pedestrian", "worker", "student"}
_LOCATION_WORDS = {"building", "entrance", "exit", "door", "room", "street", "road", "intersection", "signal", "crosswalk"}


class KnowledgeGraph:
    """Per-video directed temporal knowledge graph."""

    def __init__(self, video_id: str, graph_dir: Path | None = None):
        self.video_id = video_id
        self._graph_dir = graph_dir or GRAPH_DIR
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._graph_dir / f"{video_id}.graphml"
        # Preserve multiple temporal relationships between the same entities.
        self.G = nx.MultiDiGraph()

    # ── build from pipeline outputs ─────────────────────────────

    def build_from_captions(
        self,
        captions: list[VisualCaption],
        transcripts: list[TranscriptSegment],
    ) -> None:
        """Extract entities and relationships from VLM outputs.

        For each caption:
            1. Add each detected object as an entity node.
            2. If an activity is described, create edges linking
               entities to the activity's implied target.
        """
        for cap in captions:
            timestamp = cap.keyframe.timestamp

            # Add entity nodes from detected objects
            for obj_name in cap.objects:
                normalised = self._normalise(obj_name)
                if not normalised:
                    continue
                entity_type = self._classify_entity(normalised)
                self.add_entity(normalised, entity_type, timestamp)

            # Extract relationships from the activity description
            if cap.activity and cap.activity.lower() != "static scene":
                self._extract_activity_edges(cap.objects, cap.activity, timestamp)

        # Extract entities from transcript keywords (top 3 per segment
        # to prevent graph bloat — a 10-minute transcript can produce
        # hundreds of keywords, most of which are noise).
        for tseg in transcripts:
            keywords = self._extract_keywords(tseg.keywords)
            for kw in keywords:
                entity_type = self._classify_entity(kw)
                self.add_entity(kw, entity_type, tseg.start_time)
            self._extract_keyword_edges(keywords, tseg.start_time)

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    # ── graph operations ────────────────────────────────────────

    def add_entity(
        self, name: str, entity_type: str, timestamp: float
    ) -> None:
        """Add or update an entity node."""
        if self.G.has_node(name):
            node = self.G.nodes[name]
            node["last_seen"] = max(node.get("last_seen", 0), timestamp)
            node["count"] = node.get("count", 1) + 1
        else:
            self.G.add_node(
                name,
                label=name,
                type=entity_type,
                first_seen=timestamp,
                last_seen=timestamp,
                count=1,
            )

    def add_relationship(
        self,
        source: str,
        target: str,
        action: str,
        timestamp: float,
        segment_id: str = "",
    ) -> None:
        """Add a directed edge representing a relationship."""
        # Ensure both nodes exist
        if not self.G.has_node(source):
            self.add_entity(source, "object", timestamp)
        if not self.G.has_node(target):
            self.add_entity(target, "object", timestamp)

        self.G.add_edge(
            source,
            target,
            action=action,
            timestamp=timestamp,
            segment_id=segment_id,
        )

    def get_entity_timestamps(self, entity: str) -> list[float]:
        """Get all timestamps where an entity appears (from edges)."""
        timestamps = []
        for source, target, data in self.G.edges(data=True):
            if entity in (source, target):
                timestamps.append(data.get("timestamp", 0.0))

        node = self.G.nodes.get(entity)
        if node:
            timestamps.append(node.get("first_seen", 0.0))
        return sorted(set(timestamps))

    def get_related_entities(self, entity: str) -> list[dict]:
        """Get all entities related to the given entity."""
        related = []
        for _, target, data in self.G.out_edges(entity, data=True):
            related.append({"entity": target, "action": data.get("action", ""), "timestamp": data.get("timestamp", 0)})
        for source, _, data in self.G.in_edges(entity, data=True):
            related.append({"entity": source, "action": data.get("action", ""), "timestamp": data.get("timestamp", 0)})
        return related

    # ── persistence ─────────────────────────────────────────────

    def save(self) -> Path:
        """Serialise the graph to GraphML."""
        nx.write_graphml(self.G, str(self._graph_path))
        logger.info("Graph saved: %s (%d nodes, %d edges)",
                     self._graph_path.name, self.G.number_of_nodes(), self.G.number_of_edges())
        return self._graph_path

    def load(self) -> bool:
        """Load a previously saved graph. Returns True if found."""
        if not self._graph_path.exists():
            return False
        self.G = nx.read_graphml(str(self._graph_path))
        logger.info("Graph loaded: %s", self._graph_path.name)
        return True

    # ── internal helpers ────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip, collapse whitespace, remove punctuation."""
        text = re.sub(r"[^\w\s]", "", text.lower().strip())
        text = re.sub(r"\s+", "_", text)
        return text

    @staticmethod
    def _classify_entity(name: str) -> str:
        """Guess entity type from name."""
        words = set(name.split("_"))
        if words & _VEHICLE_WORDS:
            return "vehicle"
        if words & _PERSON_WORDS:
            return "person"
        if words & _LOCATION_WORDS:
            return "location"
        return "object"

    def _extract_activity_edges(
        self, objects: list[str], activity: str, timestamp: float
    ) -> None:
        """Best-effort relationship extraction from activity text.

        Looks for verb-like patterns connecting known objects.
        Falls back to linking all objects to the activity as a node.
        """
        normalised_objects = [self._normalise(o) for o in objects if self._normalise(o)]
        if not normalised_objects:
            return

        # Simple heuristic: extract verbs/prepositions as edge labels
        activity_lower = activity.lower()
        action_verbs = re.findall(
            r"\b(entering|leaving|stopping|moving|walking|running|standing|sitting|"
            r"driving|parking|crossing|waiting|arriving|departing|approaching)\b",
            activity_lower,
        )

        action = action_verbs[0] if action_verbs else "involved_in"

        # Link the primary actor to every other detected entity so the
        # visual relationships remain visible instead of collapsing to one edge.
        if len(normalised_objects) >= 2:
            source = normalised_objects[0]
            for target in normalised_objects[1:]:
                self.add_relationship(source, target, action, timestamp)
        else:
            activity_node = self._normalise(activity[:30])
            if activity_node:
                self.add_relationship(
                    normalised_objects[0], activity_node,
                    action, timestamp,
                )

    def _extract_keywords(self, keywords: list[str]) -> list[str]:
        """Return up to three unique, normalised keywords."""
        extracted: list[str] = []
        seen: set[str] = set()

        for kw in keywords[:3]:
            normalised = self._normalise(kw)
            if not normalised or len(normalised) <= 2 or normalised in seen:
                continue
            extracted.append(normalised)
            seen.add(normalised)

        return extracted

    def _extract_keyword_edges(self, keywords: list[str], timestamp: float) -> None:
        """Connect keywords mentioned together in the same transcript segment."""
        if len(keywords) < 2:
            return

        for source, target in combinations(keywords, 2):
            self.add_relationship(source, target, "mentioned_with", timestamp)
