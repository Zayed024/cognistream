import { useEffect, useState, useRef, useCallback } from "react";
import { getVideoGraph } from "../api/client";
import type { GraphData, GraphNode, GraphEdge } from "../types";

interface KnowledgeGraphProps {
  videoId: string;
  onClose: () => void;
}

const TYPE_COLORS: Record<string, string> = {
  person: "#ef4444",
  vehicle: "#3b82f6",
  location: "#22c55e",
  object: "#a855f7",
};

interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export default function KnowledgeGraph({ videoId, onClose }: KnowledgeGraphProps) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const animRef = useRef<number>(0);
  const dragRef = useRef<{ nodeId: string; offsetX: number; offsetY: number } | null>(null);

  useEffect(() => {
    getVideoGraph(videoId)
      .then((data) => {
        setGraph(data);
        // Initialize node positions in a circle
        const nodes: SimNode[] = data.nodes.map((n, i) => {
          const angle = (2 * Math.PI * i) / data.nodes.length;
          const r = 150;
          return {
            ...n,
            x: 300 + r * Math.cos(angle),
            y: 250 + r * Math.sin(angle),
            vx: 0,
            vy: 0,
          };
        });
        nodesRef.current = nodes;
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [videoId]);

  // Force-directed simulation
  useEffect(() => {
    if (!graph || graph.nodes.length === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    const tick = () => {
      const nodes = nodesRef.current;
      const edges = graph.edges;

      // Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
          const force = 2000 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          nodes[i].vx -= fx;
          nodes[i].vy -= fy;
          nodes[j].vx += fx;
          nodes[j].vy += fy;
        }
      }

      // Attraction along edges
      const nodeMap = new Map(nodes.map((n) => [n.id, n]));
      for (const edge of edges) {
        const src = nodeMap.get(edge.source);
        const tgt = nodeMap.get(edge.target);
        if (!src || !tgt) continue;
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = (dist - 100) * 0.01;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        src.vx += fx;
        src.vy += fy;
        tgt.vx -= fx;
        tgt.vy -= fy;
      }

      // Center gravity
      for (const node of nodes) {
        node.vx += (W / 2 - node.x) * 0.001;
        node.vy += (H / 2 - node.y) * 0.001;
      }

      // Apply velocity with damping
      for (const node of nodes) {
        if (dragRef.current?.nodeId === node.id) continue;
        node.vx *= 0.85;
        node.vy *= 0.85;
        node.x += node.vx;
        node.y += node.vy;
        node.x = Math.max(30, Math.min(W - 30, node.x));
        node.y = Math.max(30, Math.min(H - 30, node.y));
      }

      // Draw
      ctx.clearRect(0, 0, W, H);

      // Edges
      for (const edge of edges) {
        const src = nodeMap.get(edge.source);
        const tgt = nodeMap.get(edge.target);
        if (!src || !tgt) continue;
        const isHighlighted = hoveredNode === edge.source || hoveredNode === edge.target;
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = isHighlighted ? "#475569" : "#e2e8f0";
        ctx.lineWidth = isHighlighted ? 2 : 1;
        ctx.stroke();

        // Edge label
        if (isHighlighted && edge.action) {
          const mx = (src.x + tgt.x) / 2;
          const my = (src.y + tgt.y) / 2;
          ctx.font = "10px sans-serif";
          ctx.fillStyle = "#64748b";
          ctx.textAlign = "center";
          ctx.fillText(edge.action, mx, my - 4);
        }
      }

      // Nodes
      for (const node of nodes) {
        const color = TYPE_COLORS[node.type] || "#94a3b8";
        const radius = Math.min(8 + node.count * 2, 20);
        const isHovered = hoveredNode === node.id;

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isHovered ? color : color + "cc";
        ctx.fill();
        if (isHovered) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        // Label
        ctx.font = `${isHovered ? "bold " : ""}11px sans-serif`;
        ctx.fillStyle = "#1e293b";
        ctx.textAlign = "center";
        ctx.fillText(node.label, node.x, node.y + radius + 14);
      }

      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [graph, hoveredNode]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (dragRef.current) {
      const node = nodesRef.current.find((n) => n.id === dragRef.current!.nodeId);
      if (node) {
        node.x = mx - dragRef.current.offsetX;
        node.y = my - dragRef.current.offsetY;
        node.vx = 0;
        node.vy = 0;
      }
      return;
    }

    let found: string | null = null;
    for (const node of nodesRef.current) {
      const r = Math.min(8 + node.count * 2, 20);
      const dx = mx - node.x;
      const dy = my - node.y;
      if (dx * dx + dy * dy < r * r) {
        found = node.id;
        break;
      }
    }
    setHoveredNode(found);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    for (const node of nodesRef.current) {
      const r = Math.min(8 + node.count * 2, 20);
      const dx = mx - node.x;
      const dy = my - node.y;
      if (dx * dx + dy * dy < r * r) {
        dragRef.current = { nodeId: node.id, offsetX: dx, offsetY: dy };
        break;
      }
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  if (loading) return <div style={styles.loading}>Loading graph...</div>;
  if (!graph || graph.nodes.length === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <h3 style={styles.title}>Knowledge Graph</h3>
          <button onClick={onClose} style={styles.closeBtn}>×</button>
        </div>
        <p style={styles.empty}>No entities detected in this video.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Knowledge Graph</h3>
        <div style={styles.legend}>
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <span key={type} style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: color }} />
              {type}
            </span>
          ))}
        </div>
        <button onClick={onClose} style={styles.closeBtn}>×</button>
      </div>
      <canvas
        ref={canvasRef}
        width={600}
        height={500}
        style={styles.canvas}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: "#fff",
    borderRadius: "12px",
    border: "1px solid #e2e8f0",
    padding: "16px",
    marginBottom: "16px",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    marginBottom: "12px",
  },
  title: {
    margin: 0,
    fontSize: "16px",
    fontWeight: 700,
    color: "#1e293b",
  },
  legend: {
    display: "flex",
    gap: "12px",
    flex: 1,
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
    borderRadius: "50%",
    display: "inline-block",
  },
  closeBtn: {
    background: "none",
    border: "none",
    fontSize: "20px",
    color: "#94a3b8",
    cursor: "pointer",
    padding: "0 4px",
    lineHeight: 1,
  },
  canvas: {
    width: "100%",
    height: "auto",
    borderRadius: "8px",
    background: "#f8fafc",
    cursor: "grab",
  },
  loading: {
    padding: "40px",
    textAlign: "center",
    color: "#64748b",
  },
  empty: {
    textAlign: "center",
    color: "#94a3b8",
    padding: "40px",
  },
};
