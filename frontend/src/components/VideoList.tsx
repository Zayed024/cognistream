import { useEffect, useState, useCallback, useRef } from "react";
import type { VideoMeta } from "../types";
import { listVideos, processVideo, deleteVideo, subscribeToProgress, type VideoProgress } from "../api/client";

interface VideoListProps {
  onSelectVideo: (video: VideoMeta) => void;
  onUploadClick: () => void;
  refreshTrigger?: number;
}

export default function VideoList({ onSelectVideo, onUploadClick, refreshTrigger }: VideoListProps) {
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set());
  const [progressMap, setProgressMap] = useState<Map<string, VideoProgress>>(new Map());

  // Track active SSE subscriptions
  const subscriptionsRef = useRef<Map<string, () => void>>(new Map());

  const fetchVideos = useCallback(async () => {
    try {
      const data = await listVideos();
      setVideos(data);
      setError(null);

      // Update processingIds based on actual status
      const currentlyProcessing = new Set(
        data.filter((v) => v.status === "PROCESSING").map((v) => v.video_id)
      );
      setProcessingIds(currentlyProcessing);
    } catch (err) {
      setError("Failed to load videos");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Subscribe to SSE for processing videos
  useEffect(() => {
    const processingVideos = videos.filter((v) => v.status === "PROCESSING");
    const currentSubs = subscriptionsRef.current;

    // Subscribe to new processing videos
    for (const video of processingVideos) {
      if (!currentSubs.has(video.video_id)) {
        const unsubscribe = subscribeToProgress(
          video.video_id,
          (progress) => {
            setProgressMap((prev) => {
              const next = new Map(prev);
              next.set(video.video_id, progress);
              return next;
            });

            // When done, refresh the video list
            if (progress.done) {
              currentSubs.delete(video.video_id);
              fetchVideos();
            }
          },
          () => {
            // On error, remove subscription and refresh
            currentSubs.delete(video.video_id);
            fetchVideos();
          }
        );
        currentSubs.set(video.video_id, unsubscribe);
      }
    }

    // Cleanup subscriptions for videos no longer processing
    for (const [videoId, unsubscribe] of currentSubs) {
      const stillProcessing = processingVideos.some((v) => v.video_id === videoId);
      if (!stillProcessing) {
        unsubscribe();
        currentSubs.delete(videoId);
      }
    }
  }, [videos, fetchVideos]);

  // Cleanup all subscriptions on unmount
  useEffect(() => {
    return () => {
      for (const unsubscribe of subscriptionsRef.current.values()) {
        unsubscribe();
      }
      subscriptionsRef.current.clear();
    };
  }, []);

  useEffect(() => {
    fetchVideos();
    // Poll for status updates every 10 seconds (reduced from 5s since we have SSE)
    const interval = setInterval(fetchVideos, 10000);
    return () => clearInterval(interval);
  }, [fetchVideos, refreshTrigger]);

  const handleProcess = async (videoId: string) => {
    try {
      setProcessingIds((prev) => new Set(prev).add(videoId));
      await processVideo(videoId);
      // Update local state immediately
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === videoId ? { ...v, status: "PROCESSING" as const } : v
        )
      );
    } catch (err) {
      console.error("Failed to start processing:", err);
      setProcessingIds((prev) => {
        const next = new Set(prev);
        next.delete(videoId);
        return next;
      });
    }
  };

  const handleDelete = async (videoId: string, filename: string) => {
    if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
    try {
      await deleteVideo(videoId);
      setVideos((prev) => prev.filter((v) => v.video_id !== videoId));
    } catch (err) {
      console.error("Failed to delete video:", err);
    }
  };

  if (loading) {
    return (
      <div style={styles.centered}>
        <div style={styles.spinner} />
        <p style={styles.loadingText}>Loading videos...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.centered}>
        <p style={styles.errorText}>{error}</p>
        <button onClick={fetchVideos} style={styles.retryBtn}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Videos</h2>
        <button onClick={onUploadClick} style={styles.uploadBtn}>
          + Upload Video
        </button>
      </div>

      {videos.length === 0 ? (
        <div style={styles.empty}>
          <svg style={styles.emptyIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="4" width="20" height="16" rx="2" />
            <path d="M10 9l5 3-5 3V9z" />
          </svg>
          <p style={styles.emptyText}>No videos yet</p>
          <p style={styles.emptySubtext}>Upload a video to get started</p>
        </div>
      ) : (
        <div style={styles.grid}>
          {videos.map((video) => (
            <VideoCard
              key={video.video_id}
              video={video}
              isProcessing={processingIds.has(video.video_id) || video.status === "PROCESSING"}
              progress={progressMap.get(video.video_id)}
              onSelect={() => onSelectVideo(video)}
              onProcess={() => handleProcess(video.video_id)}
              onDelete={() => handleDelete(video.video_id, video.filename)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Video Card ──────────────────────────────────────────────── */

function VideoCard({
  video,
  isProcessing,
  progress,
  onSelect,
  onProcess,
  onDelete,
}: {
  video: VideoMeta;
  isProcessing: boolean;
  progress?: VideoProgress;
  onSelect: () => void;
  onProcess: () => void;
  onDelete: () => void;
}) {
  const isReady = video.status === "PROCESSED";
  const isFailed = video.status === "FAILED";
  const canProcess = video.status === "UPLOADED" || isFailed;

  return (
    <div style={styles.card}>
      {/* Thumbnail placeholder */}
      <div
        style={{
          ...styles.thumbnail,
          cursor: isReady ? "pointer" : "default",
        }}
        onClick={isReady ? onSelect : undefined}
      >
        <svg style={styles.thumbnailIcon} viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5,3 19,12 5,21" />
        </svg>
        {isProcessing && (
          <div style={styles.processingOverlay}>
            <div style={styles.progressContainer}>
              <div style={styles.progressTrack}>
                <div
                  style={{
                    ...styles.progressFill,
                    width: `${progress?.percent ?? 0}%`,
                  }}
                />
              </div>
              <span style={styles.progressPercent}>{progress?.percent ?? 0}%</span>
            </div>
            <span style={styles.processingText}>
              {progress?.stage || "Starting..."}
            </span>
          </div>
        )}
      </div>

      {/* Info */}
      <div style={styles.cardBody}>
        <h3 style={styles.cardTitle} title={video.filename}>
          {video.filename}
        </h3>
        <div style={styles.cardMeta}>
          <span>{formatDuration(video.duration_sec)}</span>
          <span style={styles.dot}>•</span>
          <StatusBadge status={video.status} />
        </div>
      </div>

      {/* Actions */}
      <div style={styles.cardActions}>
        {isReady && (
          <button onClick={onSelect} style={styles.actionBtn} title="Search this video">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
            Search
          </button>
        )}
        {canProcess && (
          <button
            onClick={onProcess}
            style={{ ...styles.actionBtn, ...styles.processBtn }}
            disabled={isProcessing}
          >
            {isProcessing ? "Starting..." : isFailed ? "Retry" : "Process"}
          </button>
        )}
        <button onClick={onDelete} style={styles.deleteBtn} title="Delete video">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/* ── Status Badge ────────────────────────────────────────────── */

function StatusBadge({ status }: { status: VideoMeta["status"] }) {
  const config = {
    UPLOADED: { bg: "#fef3c7", color: "#92400e", label: "Uploaded" },
    PROCESSING: { bg: "#dbeafe", color: "#1e40af", label: "Processing" },
    PROCESSED: { bg: "#d1fae5", color: "#065f46", label: "Ready" },
    FAILED: { bg: "#fee2e2", color: "#991b1b", label: "Failed" },
  }[status];

  return (
    <span
      style={{
        ...styles.badge,
        background: config.bg,
        color: config.color,
      }}
    >
      {config.label}
    </span>
  );
}

/* ── Helpers ─────────────────────────────────────────────────── */

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/* ── Styles ──────────────────────────────────────────────────── */

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "0",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "24px",
  },
  title: {
    fontSize: "20px",
    fontWeight: 700,
    margin: 0,
    color: "#1e293b",
  },
  uploadBtn: {
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    padding: "10px 20px",
    fontSize: "14px",
    fontWeight: 600,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: "20px",
  },
  card: {
    background: "#fff",
    borderRadius: "12px",
    border: "1px solid #e2e8f0",
    overflow: "hidden",
    transition: "box-shadow 0.2s",
  },
  thumbnail: {
    width: "100%",
    aspectRatio: "16/9",
    background: "#1e293b",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  thumbnailIcon: {
    width: "48px",
    height: "48px",
    color: "rgba(255,255,255,0.3)",
  },
  processingOverlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(0,0,0,0.7)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "8px",
  },
  processingText: {
    color: "#fff",
    fontSize: "13px",
    fontWeight: 500,
  },
  cardBody: {
    padding: "12px 16px 8px",
  },
  cardTitle: {
    fontSize: "14px",
    fontWeight: 600,
    margin: "0 0 6px 0",
    color: "#1e293b",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  cardMeta: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "12px",
    color: "#64748b",
  },
  dot: {
    color: "#cbd5e1",
  },
  badge: {
    padding: "2px 8px",
    borderRadius: "4px",
    fontSize: "11px",
    fontWeight: 600,
    textTransform: "uppercase",
  },
  cardActions: {
    display: "flex",
    gap: "8px",
    padding: "8px 16px 12px",
    borderTop: "1px solid #f1f5f9",
  },
  actionBtn: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    padding: "8px 12px",
    fontSize: "13px",
    fontWeight: 500,
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    cursor: "pointer",
    color: "#475569",
  },
  processBtn: {
    background: "#3b82f6",
    color: "#fff",
    border: "none",
  },
  deleteBtn: {
    padding: "8px 10px",
    background: "transparent",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    cursor: "pointer",
    color: "#94a3b8",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  centered: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "80px 20px",
  },
  spinner: {
    width: "32px",
    height: "32px",
    border: "3px solid #e2e8f0",
    borderTopColor: "#3b82f6",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  spinnerSmall: {
    width: "20px",
    height: "20px",
    border: "2px solid rgba(255,255,255,0.3)",
    borderTopColor: "#fff",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  loadingText: {
    marginTop: "12px",
    color: "#64748b",
    fontSize: "14px",
  },
  errorText: {
    color: "#dc2626",
    fontSize: "14px",
    marginBottom: "12px",
  },
  retryBtn: {
    padding: "8px 16px",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "14px",
  },
  empty: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "60px 20px",
    background: "#f8fafc",
    borderRadius: "12px",
    border: "2px dashed #e2e8f0",
  },
  emptyIcon: {
    width: "48px",
    height: "48px",
    color: "#94a3b8",
    marginBottom: "12px",
  },
  emptyText: {
    fontSize: "16px",
    fontWeight: 600,
    color: "#475569",
    margin: "0 0 4px 0",
  },
  emptySubtext: {
    fontSize: "14px",
    color: "#94a3b8",
    margin: 0,
  },
  progressContainer: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    width: "80%",
  },
  progressTrack: {
    flex: 1,
    height: "6px",
    background: "rgba(255,255,255,0.2)",
    borderRadius: "3px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#3b82f6",
    borderRadius: "3px",
    transition: "width 0.3s ease",
  },
  progressPercent: {
    color: "#fff",
    fontSize: "14px",
    fontWeight: 600,
    minWidth: "36px",
  },
};
