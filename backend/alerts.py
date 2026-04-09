"""
CogniStream — Alert Rules Engine

Configurable rules that watch processed segments + events and trigger
notifications (webhooks, log entries, WebSocket events) when conditions match.

Inspired by NVIDIA Metropolis VSS 3 RTVI alerts.

Rule types:
    - object_count: trigger when object label appears N+ times in a window
    - event_match: trigger when event_type matches and confidence >= threshold
    - keyword_match: trigger when caption text contains any keyword
    - anomaly: trigger when caption.anomaly is non-empty
    - sequence: trigger when pattern A is followed by pattern B within window

Usage:
    from backend.alerts import alert_engine

    # Add a rule
    alert_engine.add_rule({
        "id": "person_in_zone",
        "name": "Person enters restricted zone",
        "type": "keyword_match",
        "keywords": ["person", "intruder"],
        "severity": "high",
    })

    # Process a segment — fires alerts if matched
    fired = alert_engine.evaluate_segment(video_id, segment)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from backend.config import DATA_DIR
from backend.webhooks import fire_webhook

logger = logging.getLogger(__name__)

ALERTS_DIR = DATA_DIR / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
RULES_PATH = ALERTS_DIR / "rules.json"
HISTORY_PATH = ALERTS_DIR / "history.jsonl"


@dataclass
class AlertRule:
    """A single alert rule configuration."""
    id: str
    name: str
    type: str  # "keyword_match", "object_count", "event_match", "anomaly", "sequence"
    severity: str = "medium"  # "low", "medium", "high", "critical"
    enabled: bool = True
    # Type-specific fields
    keywords: list[str] = field(default_factory=list)
    object_label: str = ""
    threshold: int = 1
    window_sec: float = 30.0
    event_type: str = ""
    min_confidence: float = 0.5
    pattern_a: str = ""
    pattern_b: str = ""
    # Filtering
    video_ids: list[str] = field(default_factory=list)  # empty = all videos
    # Notification config
    webhook: bool = True
    websocket: bool = True


@dataclass
class AlertEvent:
    """A fired alert event."""
    id: str
    rule_id: str
    rule_name: str
    severity: str
    video_id: str
    timestamp: str
    triggered_at_sec: float  # video time
    matched_text: str
    segment_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Default rules — sensible starting set
DEFAULT_RULES = [
    {
        "id": "anomaly_detector",
        "name": "Anomaly detected in scene",
        "type": "anomaly",
        "severity": "high",
        "enabled": True,
    },
    {
        "id": "person_alert",
        "name": "Person detected",
        "type": "keyword_match",
        "keywords": ["person", "people", "intruder", "individual"],
        "severity": "medium",
        "enabled": False,
    },
    {
        "id": "vehicle_alert",
        "name": "Vehicle detected",
        "type": "keyword_match",
        "keywords": ["car", "truck", "vehicle", "bus", "motorcycle"],
        "severity": "low",
        "enabled": False,
    },
    {
        "id": "fire_smoke_alert",
        "name": "Fire or smoke detected",
        "type": "keyword_match",
        "keywords": ["fire", "smoke", "flame", "burning"],
        "severity": "critical",
        "enabled": True,
    },
    {
        "id": "weapon_alert",
        "name": "Weapon detected",
        "type": "keyword_match",
        "keywords": ["weapon", "gun", "knife", "firearm"],
        "severity": "critical",
        "enabled": True,
    },
]


class AlertEngine:
    """Evaluates segments against configured alert rules."""

    def __init__(self):
        self._rules: dict[str, AlertRule] = {}
        self._lock = threading.Lock()
        # Sliding windows per (video_id, label) for count-based rules
        self._windows: dict[tuple[str, str], deque] = defaultdict(deque)
        self._load_rules()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Rule management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_rules(self) -> None:
        """Load rules from disk, falling back to defaults."""
        if RULES_PATH.exists():
            try:
                data = json.loads(RULES_PATH.read_text())
                for r in data:
                    rule = AlertRule(**r)
                    self._rules[rule.id] = rule
                logger.info("Loaded %d alert rules from disk", len(self._rules))
                return
            except Exception as exc:
                logger.warning("Failed to load alert rules: %s — using defaults", exc)

        for r in DEFAULT_RULES:
            rule = AlertRule(**r)
            self._rules[rule.id] = rule
        self._save_rules()
        logger.info("Initialized %d default alert rules", len(self._rules))

    def _save_rules(self) -> None:
        """Persist rules to disk."""
        with self._lock:
            data = [asdict(r) for r in self._rules.values()]
        RULES_PATH.write_text(json.dumps(data, indent=2))

    def add_rule(self, rule_data: dict) -> AlertRule:
        """Add or update a rule."""
        if "id" not in rule_data:
            rule_data["id"] = uuid.uuid4().hex[:12]
        rule = AlertRule(**rule_data)
        with self._lock:
            self._rules[rule.id] = rule
        self._save_rules()
        logger.info("Added alert rule: %s (%s)", rule.name, rule.type)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._save_rules()
                return True
        return False

    def list_rules(self) -> list[AlertRule]:
        with self._lock:
            return list(self._rules.values())

    def get_rule(self, rule_id: str) -> AlertRule | None:
        with self._lock:
            return self._rules.get(rule_id)

    def update_rule(self, rule_id: str, updates: dict) -> AlertRule | None:
        with self._lock:
            existing = self._rules.get(rule_id)
            if not existing:
                return None
            current = asdict(existing)
            current.update(updates)
            current["id"] = rule_id
            updated = AlertRule(**current)
            self._rules[rule_id] = updated
        self._save_rules()
        return updated

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Evaluation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def evaluate_segment(
        self,
        video_id: str,
        segment: dict,
    ) -> list[AlertEvent]:
        """Check a segment against all enabled rules. Fires matched alerts.

        Args:
            video_id: The video this segment belongs to.
            segment: dict with at least: text, start_time, end_time, source_type, id.
                     Optional: anomaly, objects, event_type, score.

        Returns:
            List of AlertEvents that fired.
        """
        fired = []
        rules = self.list_rules()

        for rule in rules:
            if not rule.enabled:
                continue
            if rule.video_ids and video_id not in rule.video_ids:
                continue

            event = None
            if rule.type == "keyword_match":
                event = self._eval_keyword(rule, video_id, segment)
            elif rule.type == "anomaly":
                event = self._eval_anomaly(rule, video_id, segment)
            elif rule.type == "object_count":
                event = self._eval_object_count(rule, video_id, segment)
            elif rule.type == "event_match":
                event = self._eval_event_match(rule, video_id, segment)
            # sequence is more complex, kept for future

            if event:
                fired.append(event)
                self._dispatch(event, rule)

        return fired

    def _eval_keyword(self, rule: AlertRule, video_id: str, segment: dict) -> AlertEvent | None:
        text = (segment.get("text") or "").lower()
        for kw in rule.keywords:
            if kw.lower() in text:
                return self._make_event(rule, video_id, segment, matched=kw)
        return None

    def _eval_anomaly(self, rule: AlertRule, video_id: str, segment: dict) -> AlertEvent | None:
        text = segment.get("text") or ""
        anomaly = segment.get("anomaly")
        # Check explicit anomaly field
        if anomaly and str(anomaly).lower() not in ("none", "no", "n/a", ""):
            return self._make_event(rule, video_id, segment, matched=str(anomaly))
        # Check ANOMALY: line in caption text
        if "anomaly:" in text.lower():
            for line in text.split("\n"):
                line_lower = line.strip().lower()
                if line_lower.startswith("anomaly:") or line_lower.startswith("**anomaly:**"):
                    val = line.split(":", 1)[1].strip().strip("*").lower()
                    if val and val not in ("none", "no", "n/a", "nothing unusual"):
                        return self._make_event(rule, video_id, segment, matched=val)
        return None

    def _eval_object_count(self, rule: AlertRule, video_id: str, segment: dict) -> AlertEvent | None:
        if not rule.object_label:
            return None
        text = (segment.get("text") or "").lower()
        if rule.object_label.lower() not in text:
            return None

        # Track in sliding window
        key = (video_id, rule.object_label)
        now = time.monotonic()
        window = self._windows[key]
        window.append(now)
        # Prune old entries
        cutoff = now - rule.window_sec
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= rule.threshold:
            return self._make_event(
                rule, video_id, segment,
                matched=f"{rule.object_label} x{len(window)} in {rule.window_sec}s",
            )
        return None

    def _eval_event_match(self, rule: AlertRule, video_id: str, segment: dict) -> AlertEvent | None:
        if segment.get("source_type") != "event":
            return None
        event_type = segment.get("event_type") or ""
        if rule.event_type and event_type != rule.event_type:
            return None
        score = segment.get("score", 1.0)
        if score < rule.min_confidence:
            return None
        return self._make_event(rule, video_id, segment, matched=event_type)

    def _make_event(
        self,
        rule: AlertRule,
        video_id: str,
        segment: dict,
        matched: str,
    ) -> AlertEvent:
        return AlertEvent(
            id=uuid.uuid4().hex,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            video_id=video_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            triggered_at_sec=segment.get("start_time", 0),
            matched_text=matched,
            segment_id=segment.get("id", ""),
            metadata={
                "text_preview": (segment.get("text") or "")[:200],
                "source_type": segment.get("source_type", ""),
            },
        )

    def _dispatch(self, event: AlertEvent, rule: AlertRule) -> None:
        """Persist + fire notifications for an alert event."""
        # Append to history
        try:
            with open(HISTORY_PATH, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as exc:
            logger.debug("Failed to write alert history: %s", exc)

        logger.warning(
            "ALERT [%s] %s — video=%s @ %.1fs — matched: %s",
            event.severity.upper(), event.rule_name,
            event.video_id, event.triggered_at_sec, event.matched_text,
        )

        # Webhook notification
        if rule.webhook:
            fire_webhook("alert_triggered", asdict(event))

    def history(self, limit: int = 100, video_id: str | None = None) -> list[dict]:
        """Return recent alert history."""
        if not HISTORY_PATH.exists():
            return []
        events = []
        with open(HISTORY_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if video_id and e.get("video_id") != video_id:
                        continue
                    events.append(e)
                except json.JSONDecodeError:
                    continue
        return events[-limit:][::-1]  # newest first


# Module-level singleton
alert_engine = AlertEngine()
