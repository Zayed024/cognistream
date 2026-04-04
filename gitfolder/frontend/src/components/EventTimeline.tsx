import { useEffect, useState } from "react";
import { getVideoEvents, getAnnotations, createAnnotation, deleteAnnotation } from "../api/client";
import type { VideoEvent, Annotation } from "../types";

interface EventTimelineProps {
  videoId: string;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

const EVENT_COLORS: Record<string, string> = {
  car_arrival: "#3b82f6",
  car_departure: "#6366f1",
  building_entry: "#22c55e",
  building_exit: "#f59e0b",
  suspicious_activity: "#ef4444",
  pedestrian_crossing: "#06b6d4",
};

export default function EventTimeline({ videoId, duration, currentTime, onSeek }: EventTimelineProps) {
  const [events, setEvents] = useState<VideoEvent[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newLabel, setNewLabel] = useState("");
  const [newColor, setNewColor] = useState("#3b82f6");

  useEffect(() => {
    getVideoEvents(videoId).then(setEvents).catch(console.error);
    getAnnotations(videoId).then(setAnnotations).catch(console.error);
  }, [videoId]);

  const handleAddAnnotation = async () => {
    if (!newLabel.trim()) return;
    const end = Math.min(currentTime + 5, duration);
    try {
      const ann = await createAnnotation({
        video_id: videoId,
        start_time: currentTime,
        end_time: end,
        label: newLabel.trim(),
        color: newColor,
      });
      setAnnotations((prev) => [...prev, ann]);
      setNewLabel("");
      setShowAddForm(false);
    } catch (err) {
      console.error("Failed to create annotation:", err);
    }
  };

  const handleDeleteAnnotation = async (id: string) => {
    try {
      await deleteAnnotation(id);
      setAnnotations((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      console.error("Failed to delete annotation:", err);
    }
  };

  if (duration <= 0) return null;

  const toPercent = (t: number) => `${(t / duration) * 100}%`;
  const toWidth = (start: number, end: number) =>
    `${Math.max(((end - start) / duration) * 100, 0.5)}%`;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h4 style={styles.title}>Timeline</h4>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          style={styles.addBtn}
        >
          + Bookmark
        </button>
      </div>

      {/* Events track */}
      {events.length > 0 && (
        <div style={styles.track}>
          <span style={styles.trackLabel}>Events</span>
          <div style={styles.trackBar}>
            {events.map((ev) => (
              <div
                key={ev.id}
                title={`${ev.event_type}: ${ev.description}`}
                onClick={() => onSeek(ev.start_time)}
                style={{
                  ...styles.segment,
                  left: toPercent(ev.start_time),
                  width: toWidth(ev.start_time, ev.end_time),
                  background: EVENT_COLORS[ev.event_type] || "#94a3b8",
                }}
              />
            ))}
            {/* Playhead */}
            <div style={{ ...styles.playhead, left: toPercent(currentTime) }} />
          </div>
        </div>
      )}

      {/* Annotations track */}
      {annotations.length > 0 && (
        <div style={styles.track}>
          <span style={styles.trackLabel}>Bookmarks</span>
          <div style={styles.trackBar}>
            {annotations.map((ann) => (
              <div
                key={ann.id}
                title={`${ann.label}${ann.note ? ": " + ann.note : ""}`}
                onClick={() => onSeek(ann.start_time)}
                style={{
                  ...styles.segment,
                  left: toPercent(ann.start_time),
                  width: toWidth(ann.start_time, ann.end_time),
                  background: ann.color,
                }}
              />
            ))}
            <div style={{ ...styles.playhead, left: toPercent(currentTime) }} />
          </div>
        </div>
      )}

      {/* Event legend */}
      {events.length > 0 && (
        <div style={styles.legend}>
          {[...new Set(events.map((e) => e.event_type))].map((type) => (
            <span key={type} style={styles.legendItem}>
              <span
                style={{
                  ...styles.legendDot,
                  background: EVENT_COLORS[type] || "#94a3b8",
                }}
              />
              {type.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      {/* Annotation list */}
      {annotations.length > 0 && (
        <div style={styles.annotationList}>
          {annotations.map((ann) => (
            <div key={ann.id} style={styles.annotationItem}>
              <span
                style={{ ...styles.annotationDot, background: ann.color }}
              />
              <span
                style={styles.annotationLabel}
                onClick={() => onSeek(ann.start_time)}
              >
                {ann.label}
              </span>
              <span style={styles.annotationTime}>
                {formatTime(ann.start_time)}
              </span>
              <button
                onClick={() => handleDeleteAnnotation(ann.id)}
                style={styles.annotationDelete}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Add annotation form */}
      {showAddForm && (
        <div style={styles.addForm}>
          <input
            type="text"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            placeholder="Bookmark label..."
            style={styles.input}
            onKeyDown={(e) => e.key === "Enter" && handleAddAnnotation()}
          />
          <input
            type="color"
            value={newColor}
            onChange={(e) => setNewColor(e.target.value)}
            style={styles.colorPicker}
          />
          <button onClick={handleAddAnnotation} style={styles.saveBtn}>
            Save
          </button>
        </div>
      )}
    </div>
  );
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: "12px",
    padding: "12px 16px",
    background: "#fff",
    borderRadius: "10px",
    border: "1px solid #e2e8f0",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  },
  title: {
    margin: 0,
    fontSize: "13px",
    fontWeight: 700,
    color: "#475569",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  addBtn: {
    background: "none",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    padding: "4px 10px",
    fontSize: "12px",
    color: "#3b82f6",
    cursor: "pointer",
    fontWeight: 500,
  },
  track: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    marginBottom: "6px",
  },
  trackLabel: {
    fontSize: "11px",
    color: "#94a3b8",
    width: "65px",
    flexShrink: 0,
  },
  trackBar: {
    flex: 1,
    height: "16px",
    background: "#f1f5f9",
    borderRadius: "4px",
    position: "relative",
    overflow: "hidden",
  },
  segment: {
    position: "absolute",
    top: "2px",
    height: "12px",
    borderRadius: "3px",
    cursor: "pointer",
    opacity: 0.85,
    minWidth: "4px",
  },
  playhead: {
    position: "absolute",
    top: 0,
    width: "2px",
    height: "100%",
    background: "#ef4444",
    zIndex: 2,
  },
  legend: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    marginTop: "8px",
    fontSize: "11px",
    color: "#64748b",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
  },
  legendDot: {
    width: "8px",
    height: "8px",
    borderRadius: "2px",
    display: "inline-block",
  },
  annotationList: {
    marginTop: "8px",
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  annotationItem: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "12px",
    color: "#475569",
  },
  annotationDot: {
    width: "8px",
    height: "8px",
    borderRadius: "2px",
    flexShrink: 0,
  },
  annotationLabel: {
    flex: 1,
    cursor: "pointer",
  },
  annotationTime: {
    color: "#94a3b8",
    fontFamily: "monospace",
    fontSize: "11px",
  },
  annotationDelete: {
    background: "none",
    border: "none",
    color: "#94a3b8",
    cursor: "pointer",
    fontSize: "14px",
    padding: "0 2px",
  },
  addForm: {
    display: "flex",
    gap: "8px",
    marginTop: "8px",
    alignItems: "center",
  },
  input: {
    flex: 1,
    padding: "6px 10px",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    fontSize: "13px",
    outline: "none",
  },
  colorPicker: {
    width: "32px",
    height: "32px",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    cursor: "pointer",
    padding: "2px",
  },
  saveBtn: {
    padding: "6px 14px",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    fontSize: "13px",
    fontWeight: 500,
    cursor: "pointer",
  },
};
