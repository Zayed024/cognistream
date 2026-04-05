import { useEffect, useState } from "react";
import { getStats, type Stats } from "../api/client";

export default function StatsPanel() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    const interval = setInterval(() => {
      getStats().then(setStats).catch(() => {});
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  if (!stats) return null;

  const cards = [
    { label: "Videos", value: stats.videos.total, sub: stats.videos.total_duration_human },
    { label: "Segments", value: stats.segments.total, sub: "indexed" },
    { label: "Live Feeds", value: stats.live_feeds.active, sub: `of ${stats.live_feeds.total}` },
    { label: "Mode", value: stats.config.pipeline_mode, sub: stats.config.nvidia_cloud ? "NVIDIA" : "local" },
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
      gap: 12,
      marginBottom: 20,
    }}>
      {cards.map((c) => (
        <div key={c.label} style={{
          background: "#f8fafc",
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: "14px 16px",
          textAlign: "center",
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#1e293b" }}>{c.value}</div>
          <div style={{ fontSize: 13, color: "#64748b", fontWeight: 500 }}>{c.label}</div>
          <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>{c.sub}</div>
        </div>
      ))}
    </div>
  );
}
