import { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { getVideoGraph } from "../api/client";
import type { GraphData, GraphEdge, GraphNode } from "../types";

interface KnowledgeGraph3DProps {
  videoId: string;
  onClose: () => void;
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
}

interface Node3D extends GraphNode {
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface Link3D extends GraphEdge {
  source: string;
  target: string;
}

const TYPE_COLORS: Record<string, string> = {
  person: "#ef4444",
  vehicle: "#3b82f6",
  location: "#22c55e",
  object: "#a855f7",
};

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 320,
    background: "#f8fafc",
    borderLeft: "1px solid #e2e8f0",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  sidebarCard: {
    borderRadius: 14,
    border: "1px solid #e2e8f0",
    background: "#f8fafc",
    padding: 14,
  },
  sidebarHeadingRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
    marginBottom: 8,
  },
  sidebarLabel: {
    fontSize: 12,
    fontWeight: 700,
    color: "#475569",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  countBadge: {
    fontSize: 12,
    fontWeight: 700,
    color: "#1d4ed8",
    background: "#dbeafe",
    borderRadius: 999,
    padding: "4px 8px",
  },
  relationshipItem: {
    display: "flex",
    gap: 10,
    alignItems: "flex-start",
    border: "1px solid #dbe3f0",
    borderRadius: 12,
    background: "#ffffff",
    cursor: "pointer",
    padding: "10px 12px",
    textAlign: "left",
    width: "100%",
  },
  relationshipTime: {
    minWidth: 52,
    fontSize: 12,
    fontFamily: "monospace",
    color: "#1d4ed8",
    paddingTop: 2,
  },
  relationshipAction: {
    fontSize: 12,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  relationshipTarget: {
    fontSize: 14,
    fontWeight: 600,
    color: "#0f172a",
  },
  emptySidebar: {
    margin: 0,
    fontSize: 13,
    color: "#64748b",
    lineHeight: 1.5,
    padding: 16,
  },
};

export default function KnowledgeGraph3D({
  videoId,
  onClose,
  selectedNodeId,
  onNodeSelect,
}: KnowledgeGraph3DProps) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load graph data
  useEffect(() => {
    let isMounted = true;
    setLoading(true);

    getVideoGraph(videoId)
      .then((data) => {
        if (!isMounted) return;
        setGraph(data);
      })
      .catch((error) => {
        console.error("Failed to load knowledge graph:", error);
        if (!isMounted) return;
        setGraph({ nodes: [], edges: [] });
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [videoId]);

  // Transform data for 3D graph
  const graphData = useCallback(() => {
    if (!graph) return { nodes: [], links: [] };

    const nodes: Node3D[] = graph.nodes.map((node) => ({
      ...node,
    }));

    const links: Link3D[] = graph.edges.map((edge) => ({
      ...edge,
      source: edge.source,
      target: edge.target,
    }));

    return { nodes, links };
  }, [graph]);

  const handleNodeClick = useCallback(
    (node: any) => {
      onNodeSelect(node.id);
    },
    [onNodeSelect]
  );

  const handleNodeRightClick = useCallback(() => {
    onNodeSelect(null);
  }, [onNodeSelect]);

  const getNodeColor = useCallback((node: Node3D): string => {
    return TYPE_COLORS[node.type] || "#6b7280";
  }, []);

  const getNodeSize = useCallback((_node: Node3D): number => {
    // Scale based on confidence/importance if available
    return 6;
  }, []);

  const getLinkColor = useCallback((link: Link3D): string => {
    // If either node is selected, highlight connected edges
    if (selectedNodeId && (link.source === selectedNodeId || link.target === selectedNodeId)) {
      return "#fbbf24";
    }
    return "#d1d5db";
  }, [selectedNodeId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-gray-600">Loading 3D graph...</div>
      </div>
    );
  }

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-gray-500">No graph data available</div>
      </div>
    );
  }

  const data = graphData();
  const selectedNode = selectedNodeId ? graph.nodes.find((n) => n.id === selectedNodeId) : null;
  const connectedEdges = selectedNodeId
    ? graph.edges.filter((e) => e.source === selectedNodeId || e.target === selectedNodeId)
    : [];

  const relationshipItems = connectedEdges.map((edge) => {
    const otherNodeId = edge.source === selectedNodeId ? edge.target : edge.source;
    return {
      ...edge,
      id: `${edge.source}-${edge.target}-${edge.timestamp}`,
      otherNodeId,
    };
  });

  return (
    <div style={{ display: "flex", width: "100%", height: "100%", background: "#0f1724" }}>
      {/* 3D Graph Container */}
      <div ref={containerRef} style={{ flex: 1, position: "relative" }}>
        <ForceGraph3D
          ref={graphRef}
          graphData={data}
          nodeColor={getNodeColor}
          nodeVal={getNodeSize}
          nodeLabel={(node: Node3D) => `${node.label} (${node.type})`}
          linkColor={getLinkColor}
          linkWidth={1.5}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onLinkClick={() => {}}
          warmupTicks={100}
          cooldownTicks={100}
          dagMode={undefined}
          width={containerRef.current?.clientWidth || 800}
          height={containerRef.current?.clientHeight || 600}
          backgroundColor="#111827"
          nodeThreeObject={(node: any) => {
            // Composite object: colored sphere + sprite label
            const colorHex = getNodeColor(node as Node3D) || "#6b7280";
            const size = getNodeSize(node as Node3D) || 6;

            const group = new THREE.Group();

            const sphere = new THREE.Mesh(
              new THREE.SphereGeometry(Math.max(1, size * 0.6), 8, 8),
              new THREE.MeshBasicMaterial({ color: new THREE.Color(colorHex) })
            );
            group.add(sphere);

            const texture = createCanvasTexture((node as Node3D).label || "");
            if (texture) {
              const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
              sprite.scale.set(20, 8, 1);
              sprite.position.set(0, Math.max(2, size * 0.9), 0);
              group.add(sprite);
            }

            return group;
          }}
        />

        {/* Controls Overlay */}
        <div className="absolute top-4 right-4 flex gap-2 z-10">
          <button
            onClick={() => {
              if (graphRef.current) {
                graphRef.current.zoomToFit(400);
              }
            }}
            className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
            title="Reset Camera"
          >
            ⟲ Reset
          </button>
          <button
            onClick={onClose}
            className="px-3 py-1 bg-gray-700 text-white rounded text-xs hover:bg-gray-600"
            title="Close 3D Graph"
          >
            ✕
          </button>
        </div>

        {/* Info Overlay */}
        <div className="absolute bottom-4 left-4 text-xs text-gray-300 pointer-events-none">
          <div>Nodes: {data.nodes.length}</div>
          <div>Links: {data.links.length}</div>
          <div className="mt-2 text-gray-400">
            <div>Click: Select • Right-click: Deselect</div>
            <div>Mouse: Rotate/Zoom/Pan</div>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 12, flex: 1, overflow: "hidden" }}>
          <div style={styles.sidebarCard}>
            <div style={styles.sidebarHeadingRow}>
              <span style={styles.sidebarLabel}>Focus</span>
              {selectedNode && (
                <button
                  onClick={() => selectedNodeId && onNodeSelect(null)}
                  style={{
                    border: "1px solid #cbd5e1",
                    borderRadius: 999,
                    background: "#ffffff",
                    color: "#334155",
                    fontSize: 12,
                    padding: "4px 10px",
                    cursor: "pointer",
                  }}
                >
                  Clear
                </button>
              )}
            </div>

            {selectedNode ? (
              <>
                <h4 style={{ margin: "0 0 6px", fontSize: 18, color: "#0f172a" }}>{selectedNode.label}</h4>
                <p style={{ margin: 0, fontSize: 13, color: "#64748b", lineHeight: 1.5 }}>
                  {selectedNode.type} • seen {selectedNode.count} times • {formatTime(selectedNode.first_seen)} to {" "}
                  {formatTime(selectedNode.last_seen)}
                </p>
              </>
            ) : (
              <p style={{ margin: 0, fontSize: 13, color: "#64748b", lineHeight: 1.5 }}>
                Select an entity to highlight its incoming and outgoing relationships.
              </p>
            )}
          </div>

          <div style={{ ...styles.sidebarCard, flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div style={styles.sidebarHeadingRow}>
              <span style={styles.sidebarLabel}>
                {selectedNode ? "Connected relationships" : "Relationship preview"}
              </span>
              <span style={styles.countBadge}>{relationshipItems.length}</span>
            </div>

            {relationshipItems.length === 0 ? (
              <p style={styles.emptySidebar}>No explicit relationships were detected for this graph yet.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8, overflow: "auto" }}>
                {relationshipItems.slice(0, 12).map((item) => {
                  const otherNode = graph.nodes.find((n) => n.id === item.otherNodeId);
                  return (
                    <button
                      key={item.id}
                      onClick={() => onNodeSelect(item.otherNodeId)}
                      style={styles.relationshipItem}
                    >
                      <span style={styles.relationshipTime}>{formatTime(item.timestamp)}</span>
                      <span style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
                        <span style={styles.relationshipAction}>{item.action}</span>
                        <span style={styles.relationshipTarget}>{otherNode?.label || item.otherNodeId}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function createCanvasTexture(text: string): any {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "rgba(0, 0, 0, 0)";
  ctx.fillRect(0, 0, 128, 128);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 18px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text.substring(0, 15), 64, 64);

  return new THREE.CanvasTexture(canvas);
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
