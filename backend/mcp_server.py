"""
CogniStream — Model Context Protocol (MCP) Server

Exposes CogniStream's video analytics capabilities as MCP tools so any
MCP-compatible LLM client (Claude Desktop, Cursor, Cline, etc.) can drive it.

Borrowed from NVIDIA Metropolis VSS 3 — they use MCP as the unified
agent-tool interface.

Tools exposed:
    - search_videos:        natural language search across all videos
    - search_agentic:       agentic search with query decomposition + VLM rerank
    - list_videos:          list all processed videos
    - get_video_details:    metadata + segment counts for one video
    - get_video_report:     generate an LLM summary report
    - find_similar:         find segments similar to a given one
    - list_alert_rules:     show configured alert rules
    - get_alert_history:    show recent triggered alerts
    - apply_template:       apply a use-case template (surveillance, smart_city, etc.)

Run as a stdio MCP server:
    python -m backend.mcp_server

Or expose over HTTP/SSE alongside the FastAPI app (see /mcp endpoint).

Requirements:
    pip install mcp  # optional — if missing, the server falls back to a stub
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool implementations — these wrap CogniStream internals
# Each returns a dict that's JSON-serializable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def tool_search_videos(query: str, top_k: int = 10, video_id: str | None = None) -> dict:
    """Natural language search across all processed videos."""
    from backend.db.chroma_store import ChromaStore
    from backend.fusion.multimodal_embedder import MultimodalEmbedder
    from backend.retrieval.query_engine import QueryEngine

    engine = QueryEngine(embedder=MultimodalEmbedder(), store=ChromaStore())
    results = engine.search(query=query, top_k=top_k, video_id=video_id)
    return {
        "query": query,
        "result_count": len(results),
        "results": [
            {
                "video_id": r.video_id,
                "segment_id": r.segment_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "score": r.score,
                "source_type": r.source_type,
            }
            for r in results
        ],
    }


def tool_search_agentic(query: str, top_k: int = 10, video_id: str | None = None) -> dict:
    """Agentic search: query decomposition + multi-vector + VLM rerank."""
    from backend.db.chroma_store import ChromaStore
    from backend.fusion.multimodal_embedder import MultimodalEmbedder
    from backend.retrieval.query_engine import QueryEngine

    engine = QueryEngine(embedder=MultimodalEmbedder(), store=ChromaStore())
    results = engine.search_agentic(query=query, top_k=top_k, video_id=video_id)
    return {
        "query": query,
        "mode": "agentic",
        "result_count": len(results),
        "results": [
            {
                "video_id": r.video_id,
                "segment_id": r.segment_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "score": r.score,
            }
            for r in results
        ],
    }


def tool_list_videos() -> dict:
    """List all ingested videos."""
    from backend.db.sqlite import SQLiteDB
    db = SQLiteDB()
    videos = db.list_videos()
    return {
        "count": len(videos),
        "videos": [
            {
                "video_id": v.id,
                "filename": v.filename,
                "duration_sec": v.duration_sec,
                "status": v.status.value,
                "created_at": v.created_at,
            }
            for v in videos
        ],
    }


def tool_get_video_details(video_id: str) -> dict:
    """Get metadata and processing details for a video."""
    from backend.db.sqlite import SQLiteDB
    db = SQLiteDB()
    meta = db.get_video(video_id)
    if meta is None:
        return {"error": f"Video not found: {video_id}"}
    return {
        "video_id": meta.id,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "resolution": f"{meta.width}x{meta.height}",
        "status": meta.status.value,
        "segment_count": db.segment_count(video_id),
        "event_count": db.event_count(video_id),
        "created_at": meta.created_at,
        "processed_at": meta.processed_at,
    }


def tool_generate_report(
    video_id: str,
    template: str = "executive",
    scenario: str | None = None,
    events_to_track: list[str] | None = None,
    objects_of_interest: list[str] | None = None,
) -> dict:
    """Generate an LLM-powered summary report for a video."""
    from backend.db.chroma_store import ChromaStore
    from backend.db.sqlite import SQLiteDB
    from backend.reports import report_generator

    db = SQLiteDB()
    store = ChromaStore()
    meta = db.get_video(video_id)
    if meta is None:
        return {"error": f"Video not found: {video_id}"}

    return report_generator.generate(
        video_meta={
            "video_id": meta.id,
            "filename": meta.filename,
            "duration_sec": meta.duration_sec,
        },
        segments=store.get_by_video(video_id),
        events=db.list_events(video_id),
        annotations=db.list_annotations(video_id),
        template=template,
        scenario=scenario,
        events_to_track=events_to_track,
        objects_of_interest=objects_of_interest,
    )


def tool_list_alert_rules() -> dict:
    """List all configured alert rules."""
    from dataclasses import asdict
    from backend.alerts import alert_engine
    return {"rules": [asdict(r) for r in alert_engine.list_rules()]}


def tool_get_alert_history(video_id: str | None = None, limit: int = 50) -> dict:
    """Get recent alert events."""
    from backend.alerts import alert_engine
    return {"alerts": alert_engine.history(limit=limit, video_id=video_id)}


def tool_apply_template(template_id: str) -> dict:
    """Apply a use-case template (surveillance, smart_city, warehouse, etc.)."""
    from backend.use_case_templates import apply_template
    return apply_template(template_id)


def tool_list_templates() -> dict:
    """List all available use-case templates."""
    from backend.use_case_templates import list_templates
    return {"templates": list_templates()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCP tool registry — JSON Schema definitions for each tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "search_videos": {
        "description": (
            "Natural language search across all processed videos. "
            "Returns segments matching the query with timestamps and scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "top_k": {"type": "integer", "default": 10},
                "video_id": {"type": "string", "description": "Optional: scope to one video"},
            },
            "required": ["query"],
        },
        "handler": tool_search_videos,
    },
    "search_agentic": {
        "description": (
            "Advanced search using query decomposition + multi-vector retrieval + "
            "VLM reflection rerank. Best for complex multi-part queries like "
            "'find the red car after the person enters the building'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
                "video_id": {"type": "string"},
            },
            "required": ["query"],
        },
        "handler": tool_search_agentic,
    },
    "list_videos": {
        "description": "List all ingested videos in the system.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_videos,
    },
    "get_video_details": {
        "description": "Get detailed metadata for a specific video.",
        "input_schema": {
            "type": "object",
            "properties": {"video_id": {"type": "string"}},
            "required": ["video_id"],
        },
        "handler": tool_get_video_details,
    },
    "generate_report": {
        "description": (
            "Generate an LLM-powered summary report for a video. "
            "Templates: executive, incident, timeline, activity. "
            "Optional 3-parameter contract: scenario/events_to_track/objects_of_interest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "template": {"type": "string", "default": "executive"},
                "scenario": {"type": "string"},
                "events_to_track": {"type": "array", "items": {"type": "string"}},
                "objects_of_interest": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["video_id"],
        },
        "handler": tool_generate_report,
    },
    "list_alert_rules": {
        "description": "List all configured alert rules in the alert engine.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_alert_rules,
    },
    "get_alert_history": {
        "description": "Get recent alert events that have fired.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
        "handler": tool_get_alert_history,
    },
    "list_templates": {
        "description": (
            "List all available use-case templates "
            "(general, surveillance, smart_city, warehouse, retail, lecture)."
        ),
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_templates,
    },
    "apply_template": {
        "description": (
            "Apply a use-case template — adds its alert rules to the engine. "
            "IDs: general, surveillance, smart_city, warehouse, retail, lecture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"template_id": {"type": "string"}},
            "required": ["template_id"],
        },
        "handler": tool_apply_template,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCP server (using the official `mcp` package if available)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def call_tool(name: str, args: dict | None = None) -> dict:
    """Synchronous tool dispatch — used by both stdio and HTTP modes."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    args = args or {}
    handler = TOOL_REGISTRY[name]["handler"]
    try:
        return handler(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": str(exc)}


def list_tools() -> list[dict]:
    """Return tool descriptions for MCP `list_tools` requests."""
    return [
        {
            "name": name,
            "description": meta["description"],
            "inputSchema": meta["input_schema"],
        }
        for name, meta in TOOL_REGISTRY.items()
    ]


def run_stdio_server():
    """Run an MCP-compatible stdio server.

    Uses the official `mcp` package if installed, otherwise falls back to a
    minimal JSON-RPC stdio loop that's compatible with the MCP protocol.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
        import asyncio

        server = Server("cognistream")

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            return [
                Tool(
                    name=name,
                    description=meta["description"],
                    inputSchema=meta["input_schema"],
                )
                for name, meta in TOOL_REGISTRY.items()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
            result = call_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        async def main():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(main())
        return
    except ImportError:
        logger.info("`mcp` package not installed — using minimal stdio fallback")

    # Minimal JSON-RPC stdio fallback
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        rid = req.get("id")
        if method == "tools/list":
            result = {"tools": list_tools()}
        elif method == "tools/call":
            params = req.get("params", {})
            result = call_tool(params.get("name", ""), params.get("arguments"))
        else:
            result = {"error": f"Unknown method: {method}"}

        response = {"jsonrpc": "2.0", "id": rid, "result": result}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_stdio_server()
