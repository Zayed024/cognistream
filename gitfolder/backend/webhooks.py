"""
CogniStream — Webhook Notifications

Fires HTTP POST requests to configured webhook URLs when events occur.
Runs in a background thread to avoid blocking the pipeline.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config import WEBHOOK_URLS

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
    return _client


def fire_webhook(event_type: str, data: dict[str, Any]) -> None:
    """Send an event to all configured webhook URLs (non-blocking).

    Args:
        event_type: One of "video_processed", "event_detected",
                    "live_chunk_ready", "processing_failed".
        data: Event-specific payload dict.
    """
    if not WEBHOOK_URLS:
        return

    payload = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    thread = threading.Thread(
        target=_send_all,
        args=(payload,),
        daemon=True,
    )
    thread.start()


def _send_all(payload: dict) -> None:
    """POST payload to every configured webhook URL."""
    client = _get_client()
    for url in WEBHOOK_URLS:
        try:
            resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning("Webhook %s returned %d", url, resp.status_code)
        except Exception as exc:
            logger.debug("Webhook %s failed: %s", url, exc)
