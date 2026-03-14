import { useCallback, useEffect, useRef, useState } from "react";
import {
  startLiveFeed,
  stopLiveFeed,
  getLiveFeedStatus,
  connectLiveWebSocket,
  type LiveFeedInfo,
  type LiveWsEvent,
} from "../api/client";

interface LiveViewProps {
  onBack: () => void;
}

interface LiveLogEntry {
  id: number;
  time: string;
  type: string;
  message: string;
}

let logCounter = 0;

export default function LiveView({ onBack }: LiveViewProps) {
  const [url, setUrl] = useState("");
  const [feedId, setFeedId] = useState("");
  const [chunkSec, setChunkSec] = useState(15);
  const [feeds, setFeeds] = useState<LiveFeedInfo[]>([]);
  const [activeFeed, setActiveFeed] = useState<string | null>(null);
  const [log, setLog] = useState<LiveLogEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<
    { text: string; start_time: number; score: number }[]
  >([]);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const wsRef = useRef<{ send: (msg: Record<string, unknown>) => void; close: () => void } | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Poll feed status every 5s
  useEffect(() => {
    const poll = () => {
      getLiveFeedStatus()
        .then(setFeeds)
        .catch(() => {});
    };
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  const addLog = useCallback((type: string, message: string) => {
    setLog((prev) => {
      const entry: LiveLogEntry = {
        id: ++logCounter,
        time: new Date().toLocaleTimeString(),
        type,
        message,
      };
      const next = [...prev, entry];
      return next.length > 200 ? next.slice(-100) : next;
    });
  }, []);

  const handleStart = async () => {
    if (!url.trim() || !feedId.trim()) {
      setError("URL and Feed ID are required.");
      return;
    }
    setError("");
    setStarting(true);
    try {
      await startLiveFeed(url.trim(), feedId.trim(), chunkSec);
      addLog("info", `Started feed "${feedId}" from ${url}`);
      connectToFeed(feedId.trim());
      const updated = await getLiveFeedStatus();
      setFeeds(updated);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to start feed";
      setError(msg);
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async (vid: string) => {
    try {
      await stopLiveFeed(vid);
      addLog("info", `Stopped feed "${vid}"`);
      if (activeFeed === vid) {
        wsRef.current?.close();
        wsRef.current = null;
        setActiveFeed(null);
      }
      const updated = await getLiveFeedStatus();
      setFeeds(updated);
    } catch {
      addLog("error", `Failed to stop feed "${vid}"`);
    }
  };

  const connectToFeed = (vid: string) => {
    // Close existing connection
    wsRef.current?.close();

    setActiveFeed(vid);
    setLog([]);
    setSearchResults([]);
    addLog("info", `Connecting to live feed "${vid}"...`);

    const conn = connectLiveWebSocket(
      vid,
      (event: LiveWsEvent) => {
        switch (event.event_type) {
          case "status_change":
            addLog("status", `State: ${event.data.state as string}`);
            break;
          case "chunk_ready":
            addLog(
              "chunk",
              `Chunk ${event.data.chunk_index as number}: ${event.data.segments_stored as number} segments (${event.data.elapsed_sec as number}s)`
            );
            break;
          case "event_detected":
            addLog(
              "event",
              `${event.data.event_type as string}: ${event.data.description as string}`
            );
            break;
          case "search_results": {
            const results = event.data.results as {
              text: string;
              start_time: number;
              score: number;
            }[];
            setSearchResults(results);
            addLog("search", `${results.length} results for "${event.data.query as string}"`);
            break;
          }
          case "error":
            addLog("error", event.data.message as string);
            break;
          default:
            addLog("info", `${event.event_type}: ${JSON.stringify(event.data)}`);
        }
      },
      () => {
        addLog("info", "WebSocket disconnected");
      }
    );
    wsRef.current = conn;
  };

  const handleLiveSearch = () => {
    if (!searchQuery.trim() || !wsRef.current) return;
    wsRef.current.send({ action: "search", query: searchQuery.trim() });
    addLog("search", `Searching: "${searchQuery}"`);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const stateColor = (state: string) => {
    switch (state) {
      case "running": return "#22c55e";
      case "connecting": case "reconnecting": return "#eab308";
      case "error": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const logColor = (type: string) => {
    switch (type) {
      case "chunk": return "#3b82f6";
      case "event": return "#f59e0b";
      case "error": return "#ef4444";
      case "search": return "#a855f7";
      case "status": return "#22c55e";
      default: return "#9ca3af";
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <button
          onClick={onBack}
          style={{
            padding: "8px 16px",
            background: "#374151",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          Back
        </button>
        <h2 style={{ margin: 0, fontSize: 20 }}>Live Video Feeds</h2>
        <span
          style={{
            marginLeft: "auto",
            padding: "4px 12px",
            background: "#1e293b",
            borderRadius: 12,
            fontSize: 13,
            color: "#94a3b8",
          }}
        >
          {feeds.filter((f) => f.state === "running").length} active
        </span>
      </div>

      {/* Start new feed form */}
      <div
        style={{
          background: "#1e293b",
          borderRadius: 8,
          padding: 20,
          marginBottom: 20,
        }}
      >
        <h3 style={{ margin: "0 0 12px", fontSize: 15, color: "#94a3b8" }}>
          Connect to Stream
        </h3>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="rtsp://camera:554/stream or 0 for webcam"
            style={{
              flex: "2 1 250px",
              padding: "8px 12px",
              background: "#0f172a",
              border: "1px solid #334155",
              borderRadius: 6,
              color: "#e2e8f0",
              fontSize: 14,
            }}
          />
          <input
            type="text"
            value={feedId}
            onChange={(e) => setFeedId(e.target.value)}
            placeholder="Feed ID (e.g. cam01)"
            style={{
              flex: "1 1 120px",
              padding: "8px 12px",
              background: "#0f172a",
              border: "1px solid #334155",
              borderRadius: 6,
              color: "#e2e8f0",
              fontSize: 14,
            }}
          />
          <input
            type="number"
            value={chunkSec}
            onChange={(e) => setChunkSec(Number(e.target.value))}
            min={5}
            max={120}
            title="Chunk duration (seconds)"
            style={{
              width: 70,
              padding: "8px 12px",
              background: "#0f172a",
              border: "1px solid #334155",
              borderRadius: 6,
              color: "#e2e8f0",
              fontSize: 14,
              textAlign: "center",
            }}
          />
          <button
            onClick={handleStart}
            disabled={starting}
            style={{
              padding: "8px 20px",
              background: starting ? "#334155" : "#3b82f6",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: starting ? "default" : "pointer",
              fontWeight: 600,
            }}
          >
            {starting ? "Starting..." : "Start Feed"}
          </button>
        </div>
        {error && (
          <p style={{ color: "#ef4444", fontSize: 13, marginTop: 8 }}>{error}</p>
        )}
      </div>

      {/* Active feeds list */}
      {feeds.length > 0 && (
        <div
          style={{
            background: "#1e293b",
            borderRadius: 8,
            padding: 16,
            marginBottom: 20,
          }}
        >
          <h3 style={{ margin: "0 0 12px", fontSize: 15, color: "#94a3b8" }}>
            Active Feeds
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {feeds.map((f) => (
              <div
                key={f.video_id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "10px 14px",
                  background:
                    activeFeed === f.video_id ? "#1e3a5f" : "#0f172a",
                  borderRadius: 6,
                  border:
                    activeFeed === f.video_id
                      ? "1px solid #3b82f6"
                      : "1px solid transparent",
                  cursor: "pointer",
                }}
                onClick={() => connectToFeed(f.video_id)}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: stateColor(f.state),
                    flexShrink: 0,
                  }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontWeight: 600,
                      fontSize: 14,
                      color: "#e2e8f0",
                    }}
                  >
                    {f.video_id}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#64748b",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {f.url} | {f.chunks_processed} chunks | {f.total_segments}{" "}
                    segments | {f.fps.toFixed(0)} fps
                  </div>
                </div>
                <span
                  style={{
                    fontSize: 12,
                    color: stateColor(f.state),
                    fontWeight: 600,
                    textTransform: "uppercase",
                  }}
                >
                  {f.state}
                </span>
                {f.state === "running" && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleStop(f.video_id);
                    }}
                    style={{
                      padding: "4px 12px",
                      background: "#dc2626",
                      color: "#fff",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontSize: 12,
                    }}
                  >
                    Stop
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live log + search panel */}
      {activeFeed && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 16 }}>
          {/* Event log */}
          <div
            style={{
              background: "#0f172a",
              borderRadius: 8,
              padding: 16,
              maxHeight: 500,
              overflowY: "auto",
              fontFamily: "monospace",
              fontSize: 13,
            }}
          >
            <h3
              style={{
                margin: "0 0 10px",
                fontSize: 14,
                color: "#94a3b8",
                position: "sticky",
                top: 0,
                background: "#0f172a",
                padding: "4px 0",
              }}
            >
              Live Events — {activeFeed}
            </h3>
            {log.map((entry) => (
              <div
                key={entry.id}
                style={{
                  marginBottom: 4,
                  lineHeight: 1.5,
                  display: "flex",
                  gap: 8,
                }}
              >
                <span style={{ color: "#475569", flexShrink: 0 }}>
                  {entry.time}
                </span>
                <span
                  style={{
                    color: logColor(entry.type),
                    fontWeight: 600,
                    flexShrink: 0,
                    width: 56,
                    textAlign: "right",
                  }}
                >
                  [{entry.type}]
                </span>
                <span style={{ color: "#cbd5e1" }}>{entry.message}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>

          {/* Live search panel */}
          <div
            style={{
              background: "#1e293b",
              borderRadius: 8,
              padding: 16,
              display: "flex",
              flexDirection: "column",
              maxHeight: 500,
            }}
          >
            <h3
              style={{
                margin: "0 0 10px",
                fontSize: 14,
                color: "#94a3b8",
              }}
            >
              Live Search
            </h3>
            <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLiveSearch()}
                placeholder="Search indexed segments..."
                style={{
                  flex: 1,
                  padding: "8px 10px",
                  background: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 6,
                  color: "#e2e8f0",
                  fontSize: 13,
                }}
              />
              <button
                onClick={handleLiveSearch}
                style={{
                  padding: "8px 14px",
                  background: "#7c3aed",
                  color: "#fff",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                Search
              </button>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {searchResults.length === 0 ? (
                <p style={{ color: "#475569", fontSize: 13, textAlign: "center" }}>
                  Search results from live indexed segments will appear here
                </p>
              ) : (
                searchResults.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      padding: "8px 10px",
                      background: "#0f172a",
                      borderRadius: 6,
                      marginBottom: 6,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 12,
                        color: "#3b82f6",
                        marginBottom: 4,
                      }}
                    >
                      {r.start_time.toFixed(1)}s | score: {r.score.toFixed(3)}
                    </div>
                    <div
                      style={{
                        fontSize: 13,
                        color: "#e2e8f0",
                        lineHeight: 1.4,
                      }}
                    >
                      {r.text.length > 150
                        ? r.text.substring(0, 150) + "..."
                        : r.text}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
