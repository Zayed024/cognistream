import type { SearchResult } from "../types";

interface TimelineMarkersProps {
  results: SearchResult[];
  duration: number;
  currentTime: number;
  activeIndex: number | null;
  onMarkerClick: (index: number) => void;
}

export default function TimelineMarkers({
  results,
  duration,
  currentTime,
  activeIndex,
  onMarkerClick,
}: TimelineMarkersProps) {
  if (duration <= 0 || results.length === 0) return null;

  return (
    <div style={styles.container}>
      <div style={styles.label}>Timeline</div>

      <div style={styles.track}>
        {/* Playhead */}
        <div
          style={{
            ...styles.playhead,
            left: `${(currentTime / duration) * 100}%`,
          }}
        />

        {/* Result markers */}
        {results.map((result, idx) => {
          const left = (result.start_time / duration) * 100;
          const isActive = idx === activeIndex;
          const opacity = 0.4 + result.score * 0.6; // higher score = more opaque

          return (
            <button
              key={result.segment_id}
              onClick={() => onMarkerClick(idx)}
              title={`${formatTime(result.start_time)} — ${result.text.slice(0, 60)}...`}
              style={{
                ...styles.marker,
                left: `${left}%`,
                opacity,
                background: isActive ? "#f59e0b" : sourceColor(result.source_type),
                transform: isActive ? "translateX(-50%) scale(1.4)" : "translateX(-50%) scale(1)",
                zIndex: isActive ? 10 : 1,
              }}
            />
          );
        })}
      </div>

      {/* Legend */}
      <div style={styles.legend}>
        <LegendItem color="#3b82f6" label="Fused" />
        <LegendItem color="#8b5cf6" label="Visual" />
        <LegendItem color="#10b981" label="Audio" />
        <LegendItem color="#ef4444" label="Event" />
      </div>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div style={styles.legendItem}>
      <span style={{ ...styles.legendDot, background: color }} />
      <span style={styles.legendLabel}>{label}</span>
    </div>
  );
}

function sourceColor(source: string): string {
  switch (source) {
    case "visual": return "#8b5cf6";
    case "audio":  return "#10b981";
    case "event":  return "#ef4444";
    default:       return "#3b82f6"; // fused
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "12px 0",
  },
  label: {
    fontSize: "12px",
    fontWeight: 600,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "8px",
  },
  track: {
    position: "relative",
    height: "28px",
    background: "#f1f5f9",
    borderRadius: "6px",
    border: "1px solid #e2e8f0",
  },
  playhead: {
    position: "absolute",
    top: 0,
    width: "2px",
    height: "100%",
    background: "#ef4444",
    zIndex: 20,
    transition: "left 0.1s linear",
  },
  marker: {
    position: "absolute",
    top: "50%",
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    border: "2px solid #fff",
    cursor: "pointer",
    marginTop: "-5px",
    padding: 0,
    transition: "transform 0.15s, background 0.15s",
    boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
  },
  legend: {
    display: "flex",
    gap: "16px",
    marginTop: "8px",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
  },
  legendDot: {
    display: "inline-block",
    width: "8px",
    height: "8px",
    borderRadius: "50%",
  },
  legendLabel: {
    fontSize: "11px",
    color: "#64748b",
  },
};
