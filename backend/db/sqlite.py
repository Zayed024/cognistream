"""
CogniStream — SQLite Persistence

Stores video metadata and processing status so they survive restarts.
Uses raw sqlite3 (no ORM) for minimal dependencies and edge-friendly
memory footprint.

Tables:
    videos   — one row per ingested video (status, metadata, timestamps)
    segments — relational mirror of ChromaDB entries (for listing/counts
               without hitting the vector DB)
    events   — higher-level events detected by the knowledge graph engine

All timestamps are ISO 8601 strings.  IDs are hex UUIDs.
"""

from __future__ import annotations

import logging
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from backend.config import SQLITE_PATH
from backend.db.models import VideoMeta, VideoStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    duration_sec  REAL DEFAULT 0,
    fps           REAL DEFAULT 0,
    width         INTEGER DEFAULT 0,
    height        INTEGER DEFAULT 0,
    total_frames  INTEGER DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'UPLOADED',
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    error_message TEXT,
    bitrate_kbps  INTEGER DEFAULT 0,
    codec         TEXT DEFAULT '',
    pix_fmt       TEXT DEFAULT '',
    needs_thumbnail_preview INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    text        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    frame_path  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    event_type  TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    description TEXT,
    entities    TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    label       TEXT NOT NULL,
    note        TEXT DEFAULT '',
    color       TEXT DEFAULT '#3b82f6',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    captured_at     TEXT NOT NULL,
    success         INTEGER NOT NULL,
    elapsed_sec     REAL NOT NULL DEFAULT 0,
    segments_stored INTEGER NOT NULL DEFAULT 0,
    events_detected INTEGER NOT NULL DEFAULT 0,
    stage_timings   TEXT NOT NULL DEFAULT '{}',
    quality_metrics TEXT NOT NULL DEFAULT '{}',
    warnings        TEXT NOT NULL DEFAULT '[]',
    errors          TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL,
    text          TEXT NOT NULL,
    keywords      TEXT DEFAULT '',
    UNIQUE(video_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_segments_video     ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_events_video       ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_annotations_video  ON annotations(video_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_video_time ON benchmark_runs(video_id, captured_at DESC);
CREATE TABLE IF NOT EXISTS transcript_overlaps (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_index_start INTEGER NOT NULL,
    segment_index_end   INTEGER NOT NULL,
    start_time          REAL NOT NULL,
    end_time            REAL NOT NULL,
    text                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcript_video   ON transcript_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_overlap_video      ON transcript_overlaps(video_id);
"""

# FTS5 setup runs separately because it can fail on SQLite builds
# without the FTS5 extension compiled in.
_FTS5_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    text, content='transcript_segments', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS trfts_ai AFTER INSERT ON transcript_segments BEGIN
    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS trfts_ad AFTER DELETE ON transcript_segments BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_overlap_fts USING fts5(
    text, content='transcript_overlaps', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS tofts_ai AFTER INSERT ON transcript_overlaps BEGIN
    INSERT INTO transcript_overlap_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS tofts_ad AFTER DELETE ON transcript_overlaps BEGIN
    INSERT INTO transcript_overlap_fts(transcript_overlap_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;
"""


class SQLiteDB:
    """Thin wrapper around sqlite3 for video metadata persistence."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = str(db_path or SQLITE_PATH)
        self._initialised = False
        self._fts5_available = False

    # ── connection ──────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """Yield a sqlite3 connection with WAL mode and foreign keys."""
        if not self._initialised:
            self._init_schema()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_SCHEMA)
            # Forward-compatible column migrations for existing DBs.
            # CREATE TABLE IF NOT EXISTS won't add columns to tables that
            # already exist, so we use ALTER TABLE with duplicate-column
            # error suppression.
            for col, typ, default in [
                ("bitrate_kbps", "INTEGER", "0"),
                ("codec", "TEXT", "''"),
                ("pix_fmt", "TEXT", "''"),
                ("needs_thumbnail_preview", "INTEGER", "0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {typ} DEFAULT {default}")
                except Exception:
                    pass  # column already exists
            try:
                conn.executescript(_FTS5_SCHEMA)
                self._fts5_available = True
            except Exception as exc:
                logger.warning("FTS5 not available in this SQLite build: %s", exc)
                self._fts5_available = False
            conn.commit()
            logger.info("SQLite schema initialised: %s", self._db_path)
        finally:
            conn.close()
        self._initialised = True

    # ── video CRUD ──────────────────────────────────────────────

    def save_video(self, meta: VideoMeta) -> None:
        """Insert or replace a video record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO videos
                   (id, filename, file_path, duration_sec, fps, width, height,
                    total_frames, status, created_at, processed_at, error_message,
                    bitrate_kbps, codec, pix_fmt, needs_thumbnail_preview)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    meta.id, meta.filename, meta.file_path,
                    meta.duration_sec, meta.fps, meta.width, meta.height,
                    meta.total_frames, meta.status.value, meta.created_at,
                    meta.processed_at, meta.error_message,
                    meta.bitrate_kbps, meta.codec, meta.pix_fmt,
                    1 if meta.needs_thumbnail_preview else 0,
                ),
            )

    def get_video(self, video_id: str) -> Optional[VideoMeta]:
        """Fetch a single video by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_meta(row)

    def list_videos(self) -> list[VideoMeta]:
        """Return all videos ordered by creation time (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        processed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update video processing status."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE videos
                   SET status = ?, processed_at = ?, error_message = ?
                   WHERE id = ?""",
                (status.value, processed_at, error_message, video_id),
            )
        logger.debug("Video %s → %s", video_id, status.value)

    def delete_video(self, video_id: str) -> None:
        """Delete a video and all its associated segments and events."""
        with self._connect() as conn:
            conn.execute("DELETE FROM benchmark_runs WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        logger.info("Deleted video %s from SQLite.", video_id)

    # ── benchmark runs ─────────────────────────────────────────

    def save_benchmark_run(self, benchmark: dict) -> None:
        """Persist one benchmark run payload for historical tracking."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO benchmark_runs
                   (id, video_id, captured_at, success, elapsed_sec,
                    segments_stored, events_detected, stage_timings,
                    quality_metrics, warnings, errors)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    benchmark["id"],
                    benchmark["video_id"],
                    benchmark["captured_at"],
                    1 if benchmark.get("success") else 0,
                    float(benchmark.get("elapsed_sec", 0.0)),
                    int(benchmark.get("segments_stored", 0)),
                    int(benchmark.get("events_detected", 0)),
                    json.dumps(benchmark.get("stage_timings", {}), ensure_ascii=True),
                    json.dumps(benchmark.get("quality_metrics", {}), ensure_ascii=True),
                    json.dumps(benchmark.get("warnings", []), ensure_ascii=True),
                    json.dumps(benchmark.get("errors", []), ensure_ascii=True),
                ),
            )

    def get_latest_benchmark(self, video_id: str) -> Optional[dict]:
        """Get the most recent benchmark run for a video."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM benchmark_runs
                   WHERE video_id = ?
                   ORDER BY captured_at DESC
                   LIMIT 1""",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        return self._benchmark_row_to_dict(row)

    def list_benchmark_runs(self, video_id: str, limit: int = 20) -> list[dict]:
        """List benchmark runs (newest first) for a video."""
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM benchmark_runs
                   WHERE video_id = ?
                   ORDER BY captured_at DESC
                   LIMIT ?""",
                (video_id, safe_limit),
            ).fetchall()
        return [self._benchmark_row_to_dict(r) for r in rows]

    def segment_count(self, video_id: str) -> int:
        """Count segments for a video."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return row[0] if row else 0

    def reset_stale_processing(self) -> list[str]:
        """Reset any videos stuck in PROCESSING state back to UPLOADED.

        This handles recovery from interrupted processing (e.g., server crash).
        Returns list of video IDs that were reset.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM videos WHERE status = ?",
                (VideoStatus.PROCESSING.value,),
            ).fetchall()
            if rows:
                conn.execute(
                    "UPDATE videos SET status = ?, error_message = ? WHERE status = ?",
                    (VideoStatus.UPLOADED.value, "Reset after server restart", VideoStatus.PROCESSING.value),
                )
        reset_ids = [r[0] for r in rows]
        if reset_ids:
            logger.warning("Reset %d stale PROCESSING videos: %s", len(reset_ids), reset_ids)
        return reset_ids

    def event_count(self, video_id: str) -> int:
        """Count events for a video."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return row[0] if row else 0

    # ── events ───────────────────────────────────────────────────

    def list_events(self, video_id: str) -> list[dict]:
        """Return all events for a video, sorted by start_time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "video_id": r["video_id"],
                "event_type": r["event_type"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "description": r["description"],
                "entities": r["entities"].split(",") if r["entities"] else [],
            }
            for r in rows
        ]

    # ── annotations ──────────────────────────────────────────────

    def save_annotation(self, ann: dict) -> None:
        """Insert or replace an annotation."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO annotations
                   (id, video_id, start_time, end_time, label, note, color, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    ann["id"], ann["video_id"], ann["start_time"], ann["end_time"],
                    ann["label"], ann.get("note", ""), ann.get("color", "#3b82f6"),
                    ann["created_at"],
                ),
            )

    def list_annotations(self, video_id: str) -> list[dict]:
        """Return all annotations for a video, sorted by start_time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete a single annotation. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM annotations WHERE id = ?", (annotation_id,),
            )
        return cursor.rowcount > 0

    # ── transcripts (FTS5 speech search) ─────────────────────

    def save_transcripts(self, video_id: str, transcripts: list) -> int:
        """Persist raw Whisper transcript segments for FTS5 search.

        Args:
            video_id: The video these transcripts belong to.
            transcripts: List of TranscriptSegment dataclass instances.

        Returns:
            Number of segments stored.
        """
        if not transcripts:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO transcript_segments
                   (video_id, segment_index, start_time, end_time, text, keywords)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        video_id, i, t.start_time, t.end_time, t.text,
                        ",".join(t.keywords) if t.keywords else "",
                    )
                    for i, t in enumerate(transcripts)
                ],
            )
        return len(transcripts)

    def save_transcript_overlaps(self, video_id: str, transcripts: list) -> int:
        """Generate and store sliding 2-segment overlap windows.

        When Whisper splits a phrase across two segments, the individual
        FTS5 entries won't match a multi-word query. Overlapping windows
        (each spanning 2 consecutive segments) catch these boundary cases.
        Borrowed from Moment Search's transcript_overlaps pattern.
        """
        if len(transcripts) < 2:
            return 0
        overlaps = []
        for i in range(len(transcripts) - 1):
            a, b = transcripts[i], transcripts[i + 1]
            combined = f"{a.text} {b.text}".strip()
            if combined:
                overlaps.append((video_id, i, i + 1, a.start_time, b.end_time, combined))
        if not overlaps:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO transcript_overlaps
                   (video_id, segment_index_start, segment_index_end,
                    start_time, end_time, text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                overlaps,
            )
        return len(overlaps)

    def delete_transcripts(self, video_id: str) -> int:
        """Delete all transcript segments + overlaps for a video."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM transcript_overlaps WHERE video_id = ?",
                (video_id,),
            )
            cursor = conn.execute(
                "DELETE FROM transcript_segments WHERE video_id = ?",
                (video_id,),
            )
        return cursor.rowcount

    @staticmethod
    def _escape_fts5(query: str) -> str:
        """Escape user input for safe FTS5 MATCH."""
        return query.replace('"', '""')

    def search_transcripts(
        self,
        query: str,
        video_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Full-text search over transcript segments via FTS5.

        Returns list of dicts with video_id, start_time, end_time, text,
        snippet (highlighted match), and bm25 score.
        """
        escaped = self._escape_fts5(query.strip())
        if not escaped:
            return []

        # Ensure schema init has run (sets _fts5_available)
        if not self._initialised:
            self._init_schema()

        if not self._fts5_available:
            return []

        sql = """
            SELECT
                ts.video_id,
                ts.start_time,
                ts.end_time,
                ts.text,
                ts.keywords,
                snippet(transcript_fts, 0, '<mark>', '</mark>', '...', 24) AS snippet,
                rank AS bm25_score
            FROM transcript_fts
            JOIN transcript_segments ts ON ts.id = transcript_fts.rowid
            WHERE transcript_fts MATCH ?
        """
        params: list = [f'"{escaped}"']

        if video_id:
            sql += " AND ts.video_id = ?"
            params.append(video_id)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            results = [
                {
                    "video_id": r["video_id"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "text": r["text"],
                    "snippet": r["snippet"],
                    "score": abs(r["bm25_score"]),
                    "source_type": "speech",
                    "keywords": r["keywords"].split(",") if r["keywords"] else [],
                }
                for r in rows
            ]
            # Fallback chain (borrowed from Moment Search):
            # 1. Segment-level phrase match (above)
            # 2. Overlap window phrase match (boundary recovery)
            # 3. Any-term OR match (last resort — at least one word matches)
            if not results:
                results = self._search_transcript_overlaps(escaped, video_id, limit)
            if not results:
                results = self._search_transcripts_any_term(query.strip(), video_id, limit)
            return results
        except Exception as exc:
            logger.warning("FTS5 search failed: %s", exc)
            return []

    def _search_transcript_overlaps(
        self, escaped_query: str, video_id: str | None, limit: int,
    ) -> list[dict]:
        """Search the overlap FTS5 table as a fallback."""
        sql = """
            SELECT
                tov.video_id, tov.start_time, tov.end_time, tov.text,
                snippet(transcript_overlap_fts, 0, '<mark>', '</mark>', '...', 32) AS snippet,
                rank AS bm25_score
            FROM transcript_overlap_fts
            JOIN transcript_overlaps tov ON tov.id = transcript_overlap_fts.rowid
            WHERE transcript_overlap_fts MATCH ?
        """
        params: list = [f'"{escaped_query}"']
        if video_id:
            sql += " AND tov.video_id = ?"
            params.append(video_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "video_id": r["video_id"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "text": r["text"],
                    "snippet": r["snippet"],
                    "score": abs(r["bm25_score"]),
                    "source_type": "speech",
                    "keywords": [],
                }
                for r in rows
            ]
        except Exception:
            return []

    def _search_transcripts_any_term(
        self, query: str, video_id: str | None, limit: int,
    ) -> list[dict]:
        """Last-resort fallback: OR individual words from the query.

        If the exact phrase and overlap matches both return nothing, at
        least one individual word might match a transcript segment.
        Borrowed from Moment Search's ``searchTranscriptsFTSAnyTerms``.
        """
        words = [w for w in re.split(r'\s+', query) if len(w) >= 2]
        if not words:
            return []
        or_query = " OR ".join(f'"{self._escape_fts5(w)}"' for w in words)

        sql = """
            SELECT
                ts.video_id, ts.start_time, ts.end_time, ts.text,
                snippet(transcript_fts, 0, '<mark>', '</mark>', '...', 24) AS snippet,
                rank AS bm25_score
            FROM transcript_fts
            JOIN transcript_segments ts ON ts.id = transcript_fts.rowid
            WHERE transcript_fts MATCH ?
        """
        params: list = [or_query]
        if video_id:
            sql += " AND ts.video_id = ?"
            params.append(video_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "video_id": r["video_id"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "text": r["text"],
                    "snippet": r["snippet"],
                    "score": abs(r["bm25_score"]),
                    "source_type": "speech",
                    "keywords": [],
                }
                for r in rows
            ]
        except Exception:
            return []

    def transcript_count(self, video_id: str) -> int:
        """Count transcript segments for a video."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM transcript_segments WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return row[0] if row else 0

    # ── helpers ─────────────────────────────────────────────────

    def add_column_if_not_exists(
        self, table: str, column: str, col_type: str, default: str = "NULL",
    ) -> bool:
        """Safely add a column to an existing table.

        Returns True if the column was added, False if it already existed.
        Borrowed from Moment Search's addColumnIfNotExists pattern for
        forward-compatible schema evolution without data loss.
        """
        try:
            with self._connect() as conn:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")
            logger.info("Added column %s.%s", table, column)
            return True
        except Exception as exc:
            if "duplicate column" in str(exc).lower():
                return False
            raise

    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> VideoMeta:
        keys = row.keys() if hasattr(row, "keys") else []
        return VideoMeta(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            duration_sec=row["duration_sec"],
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            total_frames=row["total_frames"],
            status=VideoStatus(row["status"]),
            created_at=row["created_at"],
            processed_at=row["processed_at"],
            error_message=row["error_message"],
            bitrate_kbps=row["bitrate_kbps"] if "bitrate_kbps" in keys else 0,
            codec=row["codec"] if "codec" in keys else "",
            pix_fmt=row["pix_fmt"] if "pix_fmt" in keys else "",
            needs_thumbnail_preview=bool(row["needs_thumbnail_preview"]) if "needs_thumbnail_preview" in keys else False,
        )

    @staticmethod
    def _benchmark_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "video_id": row["video_id"],
            "captured_at": row["captured_at"],
            "success": bool(row["success"]),
            "elapsed_sec": row["elapsed_sec"],
            "segments_stored": row["segments_stored"],
            "events_detected": row["events_detected"],
            "stage_timings": json.loads(row["stage_timings"] or "{}"),
            "quality_metrics": json.loads(row["quality_metrics"] or "{}"),
            "warnings": json.loads(row["warnings"] or "[]"),
            "errors": json.loads(row["errors"] or "[]"),
        }
