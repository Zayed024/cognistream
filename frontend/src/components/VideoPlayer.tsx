import type { SearchResult } from "../types";

interface VideoPlayerProps {
  videoUrl: string | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  onTimeUpdate: () => void;
  onLoadedMetadata: () => void;
  togglePlay: () => void;
  activeResult: SearchResult | null;
}

export default function VideoPlayer({
  videoUrl,
  videoRef,
  currentTime,
  duration,
  isPlaying,
  onTimeUpdate,
  onLoadedMetadata,
  togglePlay,
  activeResult,
}: VideoPlayerProps) {
  if (!videoUrl) {
    return (
      <div style={styles.placeholder}>
        <svg style={styles.placeholderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <polygon points="5,3 19,12 5,21" />
        </svg>
        <p style={styles.placeholderText}>Select a video to begin</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <video
        ref={videoRef}
        src={videoUrl}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
        style={styles.video}
        onClick={togglePlay}
      />

      {/* Transport bar */}
      <div style={styles.transport}>
        <button onClick={togglePlay} style={styles.playBtn}>
          {isPlaying ? "\u23F8" : "\u25B6"}
        </button>

        <div style={styles.progressTrack}>
          <div
            style={{
              ...styles.progressFill,
              width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%",
            }}
          />
          {/* Active result indicator on the progress bar */}
          {activeResult && duration > 0 && (
            <div
              style={{
                ...styles.activeMarker,
                left: `${(activeResult.start_time / duration) * 100}%`,
                width: `${(Math.max(activeResult.end_time - activeResult.start_time, 1) / duration) * 100}%`,
              }}
            />
          )}
        </div>

        <span style={styles.timecode}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    borderRadius: "12px",
    overflow: "hidden",
    background: "#000",
    border: "1px solid #e2e8f0",
  },
  video: {
    width: "100%",
    display: "block",
    cursor: "pointer",
    maxHeight: "450px",
    objectFit: "contain",
    background: "#000",
  },
  transport: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "10px 16px",
    background: "#1e293b",
  },
  playBtn: {
    background: "none",
    border: "none",
    color: "#fff",
    fontSize: "18px",
    cursor: "pointer",
    padding: "0 4px",
    lineHeight: 1,
  },
  progressTrack: {
    flex: 1,
    height: "6px",
    background: "#334155",
    borderRadius: "3px",
    position: "relative",
    overflow: "hidden",
  },
  progressFill: {
    position: "absolute",
    top: 0,
    left: 0,
    height: "100%",
    background: "#3b82f6",
    borderRadius: "3px",
    transition: "width 0.1s linear",
  },
  activeMarker: {
    position: "absolute",
    top: 0,
    height: "100%",
    background: "rgba(251, 191, 36, 0.6)",
    borderRadius: "3px",
    minWidth: "4px",
  },
  timecode: {
    color: "#94a3b8",
    fontSize: "13px",
    fontFamily: "monospace",
    whiteSpace: "nowrap",
    minWidth: "90px",
    textAlign: "right",
  },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "350px",
    background: "#f1f5f9",
    borderRadius: "12px",
    border: "2px dashed #cbd5e1",
    gap: "16px",
  },
  placeholderIcon: {
    width: "48px",
    height: "48px",
    color: "#94a3b8",
  },
  placeholderText: {
    color: "#64748b",
    fontSize: "15px",
  },
};
