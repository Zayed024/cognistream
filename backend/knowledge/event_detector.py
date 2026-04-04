"""
CogniStream — Event Detector

Detects higher-level events from sequences of actions in the knowledge
graph.  An "event" is a temporal pattern of edges that matches a known
template.

Example patterns:
    vehicle_appears → vehicle_stops  =  "car_arrival"
    person_appears  → person_enters  =  "building_entry"
    person_running  → person_exits   =  "suspicious_departure"

The detector scans graph edges in temporal order, maintains a sliding
window, and emits Event objects when a pattern completes.

Usage:
    detector = EventDetector()
    events = detector.detect(knowledge_graph)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.db.models import Event
from backend.knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class EventPattern:
    """A sequence of actions that constitutes a named event."""
    name: str
    actions: list[str]          # ordered action sequence to match
    entity_types: list[str]     # required entity types (e.g. ["vehicle"])
    max_gap_sec: float = 30.0   # max time between consecutive actions


# ── Built-in event patterns ────────────────────────────────────

DEFAULT_PATTERNS: list[EventPattern] = [
    EventPattern(
        name="car_arrival",
        actions=["approaching", "stopping"],
        entity_types=["vehicle"],
        max_gap_sec=30.0,
    ),
    EventPattern(
        name="car_arrival",
        actions=["arriving", "stopping"],
        entity_types=["vehicle"],
        max_gap_sec=30.0,
    ),
    EventPattern(
        name="car_departure",
        actions=["moving", "departing"],
        entity_types=["vehicle"],
        max_gap_sec=30.0,
    ),
    EventPattern(
        name="building_entry",
        actions=["approaching", "entering"],
        entity_types=["person"],
        max_gap_sec=20.0,
    ),
    EventPattern(
        name="building_entry",
        actions=["walking", "entering"],
        entity_types=["person"],
        max_gap_sec=20.0,
    ),
    EventPattern(
        name="building_exit",
        actions=["leaving", "walking"],
        entity_types=["person"],
        max_gap_sec=20.0,
    ),
    EventPattern(
        name="suspicious_activity",
        actions=["running", "departing"],
        entity_types=["person"],
        max_gap_sec=15.0,
    ),
    EventPattern(
        name="pedestrian_crossing",
        actions=["waiting", "crossing"],
        entity_types=["person"],
        max_gap_sec=45.0,
    ),
]


class EventDetector:
    """Scan a knowledge graph for temporal action patterns."""

    def __init__(self, patterns: list[EventPattern] | None = None):
        self.patterns = patterns or DEFAULT_PATTERNS

    def detect(self, graph: KnowledgeGraph) -> list[Event]:
        """Detect events in the knowledge graph.

        Returns:
            List of :class:`Event` objects sorted by start_time.
        """
        if graph.G.number_of_edges() == 0:
            logger.debug("Empty graph — no events to detect.")
            return []

        # Collect all edges sorted by timestamp
        edges = self._sorted_edges(graph)

        events: list[Event] = []

        for pattern in self.patterns:
            found = self._match_pattern(edges, pattern, graph)
            events.extend(found)

        # Deduplicate overlapping events of the same type
        events = self._deduplicate(events)
        events.sort(key=lambda e: e.start_time)

        logger.info("Event detection: %d events found.", len(events))
        return events

    # ── pattern matching ────────────────────────────────────────

    def _match_pattern(
        self,
        edges: list[dict],
        pattern: EventPattern,
        graph: KnowledgeGraph,
    ) -> list[Event]:
        """Scan edges for sequences matching the pattern."""
        events: list[Event] = []
        required_actions = pattern.actions
        n_required = len(required_actions)

        for i, edge in enumerate(edges):
            if edge["action"] != required_actions[0]:
                continue

            # Check entity type
            source_type = graph.G.nodes.get(edge["source"], {}).get("type", "")
            if pattern.entity_types and source_type not in pattern.entity_types:
                continue

            # Try to match the remaining actions in sequence
            matched = [edge]
            for action_idx in range(1, n_required):
                next_match = self._find_next_action(
                    edges, i + 1,
                    required_actions[action_idx],
                    matched[-1]["timestamp"],
                    pattern.max_gap_sec,
                    edge["source"],
                )
                if next_match is None:
                    break
                matched.append(next_match)

            if len(matched) == n_required:
                entities = list({m["source"] for m in matched} | {m["target"] for m in matched})
                events.append(Event(
                    id=uuid.uuid4().hex,
                    video_id=graph.video_id,
                    event_type=pattern.name,
                    start_time=matched[0]["timestamp"],
                    end_time=matched[-1]["timestamp"],
                    description=f"{pattern.name}: {' → '.join(m['action'] for m in matched)}",
                    entities=entities,
                ))

        return events

    @staticmethod
    def _find_next_action(
        edges: list[dict],
        start_idx: int,
        action: str,
        after_time: float,
        max_gap: float,
        source_entity: str,
    ) -> dict | None:
        """Find the next edge matching the action within the time window."""
        deadline = after_time + max_gap
        for j in range(start_idx, len(edges)):
            e = edges[j]
            if e["timestamp"] > deadline:
                break
            if e["timestamp"] <= after_time:
                continue
            if e["action"] == action and e["source"] == source_entity:
                return e
        return None

    @staticmethod
    def _sorted_edges(graph: KnowledgeGraph) -> list[dict]:
        """Extract all edges as dicts, sorted by timestamp."""
        edges = []
        for source, target, data in graph.G.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "action": data.get("action", ""),
                "timestamp": float(data.get("timestamp", 0)),
                "segment_id": data.get("segment_id", ""),
            })
        edges.sort(key=lambda e: e["timestamp"])
        return edges

    @staticmethod
    def _deduplicate(events: list[Event]) -> list[Event]:
        """Remove duplicate events with overlapping time ranges."""
        if not events:
            return events

        events.sort(key=lambda e: (e.event_type, e.start_time))
        unique: list[Event] = [events[0]]

        for ev in events[1:]:
            prev = unique[-1]
            # Same type and overlapping — skip
            if (ev.event_type == prev.event_type
                    and ev.start_time <= prev.end_time):
                continue
            unique.append(ev)

        return unique
