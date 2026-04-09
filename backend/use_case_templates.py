"""
CogniStream — Use Case Templates

Pre-configured rule sets, object labels, and report templates for common
video analytics scenarios. User picks a template → auto-configures alerts,
detection labels, and the default report style.

Inspired by NVIDIA Metropolis VSS 3 industry-specific examples
(smart city, warehouse, retail).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UseCaseTemplate:
    id: str
    name: str
    description: str
    detection_labels: list[str] = field(default_factory=list)
    alert_rules: list[dict] = field(default_factory=list)
    default_report_template: str = "executive"
    chunk_sec: int = 15  # default live chunk size for this use case
    suggested_queries: list[str] = field(default_factory=list)


TEMPLATES: dict[str, UseCaseTemplate] = {
    "general": UseCaseTemplate(
        id="general",
        name="General Purpose",
        description="Default settings for any video content",
        detection_labels=["person", "vehicle", "building", "animal", "object"],
        alert_rules=[
            {
                "id": "general_anomaly",
                "name": "Anomaly detected",
                "type": "anomaly",
                "severity": "medium",
                "enabled": True,
            },
        ],
        default_report_template="executive",
        chunk_sec=15,
        suggested_queries=[
            "what is happening in the video",
            "show me the main scene",
            "find any unusual activity",
        ],
    ),

    "surveillance": UseCaseTemplate(
        id="surveillance",
        name="Security Surveillance",
        description="CCTV/security footage with intrusion and anomaly detection",
        detection_labels=[
            "person", "intruder", "weapon", "vehicle", "package",
            "door", "window", "fence", "fire", "smoke",
        ],
        alert_rules=[
            {
                "id": "surveillance_intrusion",
                "name": "Person detected",
                "type": "keyword_match",
                "keywords": ["person", "individual", "intruder", "trespasser"],
                "severity": "high",
                "enabled": True,
            },
            {
                "id": "surveillance_weapon",
                "name": "Weapon detected",
                "type": "keyword_match",
                "keywords": ["weapon", "gun", "knife", "firearm"],
                "severity": "critical",
                "enabled": True,
            },
            {
                "id": "surveillance_fire",
                "name": "Fire or smoke",
                "type": "keyword_match",
                "keywords": ["fire", "smoke", "flame", "burning"],
                "severity": "critical",
                "enabled": True,
            },
            {
                "id": "surveillance_anomaly",
                "name": "Suspicious activity",
                "type": "anomaly",
                "severity": "high",
                "enabled": True,
            },
        ],
        default_report_template="incident",
        chunk_sec=5,  # tight latency for security
        suggested_queries=[
            "when did someone enter the building",
            "find any suspicious activity",
            "show me people approaching the entrance",
            "did anyone leave a package",
        ],
    ),

    "smart_city": UseCaseTemplate(
        id="smart_city",
        name="Smart City / Traffic",
        description="Traffic cameras, intersections, urban monitoring",
        detection_labels=[
            "car", "truck", "bus", "motorcycle", "bicycle",
            "pedestrian", "traffic light", "sign", "road",
        ],
        alert_rules=[
            {
                "id": "city_accident",
                "name": "Possible accident",
                "type": "keyword_match",
                "keywords": ["accident", "collision", "crash", "stalled"],
                "severity": "high",
                "enabled": True,
            },
            {
                "id": "city_emergency",
                "name": "Emergency vehicle",
                "type": "keyword_match",
                "keywords": ["ambulance", "police", "fire truck", "siren"],
                "severity": "high",
                "enabled": True,
            },
            {
                "id": "city_congestion",
                "name": "Heavy traffic",
                "type": "object_count",
                "object_label": "car",
                "threshold": 10,
                "window_sec": 30.0,
                "severity": "low",
                "enabled": False,
            },
        ],
        default_report_template="activity",
        chunk_sec=10,
        suggested_queries=[
            "when did traffic get heavy",
            "show me pedestrian crossings",
            "find any accidents or incidents",
            "what color was the car at the intersection",
        ],
    ),

    "warehouse": UseCaseTemplate(
        id="warehouse",
        name="Warehouse Operations",
        description="Forklift tracking, inventory zones, worker safety",
        detection_labels=[
            "forklift", "worker", "pallet", "box", "shelf",
            "cart", "truck", "loading dock", "safety vest",
        ],
        alert_rules=[
            {
                "id": "warehouse_unauthorized",
                "name": "Unauthorized person in zone",
                "type": "keyword_match",
                "keywords": ["unauthorized", "intruder", "no safety vest"],
                "severity": "high",
                "enabled": True,
            },
            {
                "id": "warehouse_fall",
                "name": "Worker fall or injury",
                "type": "keyword_match",
                "keywords": ["fall", "fallen", "injury", "lying down", "collapsed"],
                "severity": "critical",
                "enabled": True,
            },
            {
                "id": "warehouse_spill",
                "name": "Spill or hazard",
                "type": "keyword_match",
                "keywords": ["spill", "leak", "broken", "hazard", "obstruction"],
                "severity": "medium",
                "enabled": True,
            },
        ],
        default_report_template="incident",
        chunk_sec=10,
        suggested_queries=[
            "when did the forklift enter zone B",
            "find any worker safety incidents",
            "show me pallet movements",
            "did anyone enter without a safety vest",
        ],
    ),

    "retail": UseCaseTemplate(
        id="retail",
        name="Retail Analytics",
        description="Store cameras, customer behavior, theft prevention",
        detection_labels=[
            "customer", "employee", "shopping cart", "shelf",
            "cash register", "exit", "product", "queue",
        ],
        alert_rules=[
            {
                "id": "retail_theft",
                "name": "Suspicious theft activity",
                "type": "keyword_match",
                "keywords": ["concealing", "stealing", "shoplifting", "hiding item"],
                "severity": "critical",
                "enabled": True,
            },
            {
                "id": "retail_queue",
                "name": "Long queue at checkout",
                "type": "object_count",
                "object_label": "customer",
                "threshold": 5,
                "window_sec": 60.0,
                "severity": "low",
                "enabled": False,
            },
        ],
        default_report_template="activity",
        chunk_sec=15,
        suggested_queries=[
            "when was the store busiest",
            "find any suspicious customer behavior",
            "show me customers near the exit",
            "how many people were at the checkout",
        ],
    ),

    "lecture": UseCaseTemplate(
        id="lecture",
        name="Educational / Lecture",
        description="Lecture videos, presentations, tutorials",
        detection_labels=[
            "presenter", "whiteboard", "screen", "slide",
            "laptop", "audience", "diagram", "text",
        ],
        alert_rules=[],  # No alerts for lectures
        default_report_template="timeline",
        chunk_sec=30,
        suggested_queries=[
            "when did the speaker discuss X",
            "find the slide about Y",
            "what topics were covered",
            "show me the diagrams in the video",
        ],
    ),
}


def list_templates() -> list[dict]:
    """Return all available templates as dicts."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "detection_labels": t.detection_labels,
            "alert_rule_count": len(t.alert_rules),
            "default_report_template": t.default_report_template,
            "chunk_sec": t.chunk_sec,
            "suggested_queries": t.suggested_queries,
        }
        for t in TEMPLATES.values()
    ]


def get_template(template_id: str) -> UseCaseTemplate | None:
    return TEMPLATES.get(template_id)


def apply_template(template_id: str) -> dict[str, Any]:
    """Apply a template — adds its alert rules to the alert engine.

    Returns a summary of what was applied.
    """
    template = get_template(template_id)
    if not template:
        return {"error": f"Unknown template: {template_id}"}

    from backend.alerts import alert_engine

    added = []
    for rule_data in template.alert_rules:
        rule = alert_engine.add_rule(rule_data.copy())
        added.append(rule.id)

    return {
        "template": template_id,
        "name": template.name,
        "rules_added": added,
        "detection_labels": template.detection_labels,
        "default_report_template": template.default_report_template,
        "chunk_sec": template.chunk_sec,
    }
