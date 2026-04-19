/** Mirrors backend SearchResult dataclass */
export interface SearchResult {
  video_id: string;
  segment_id: string;
  start_time: number;
  end_time: number;
  text: string;
  source_type: "visual" | "audio" | "fused" | "event" | "speech";
  score: number;
  event_type?: string;
  frame_url?: string;
  speech_snippet?: string;
  related_count?: number;
}

export interface SearchResponse {
  query: string;
  result_count: number;
  results: SearchResult[];
}

export interface VideoMeta {
  video_id: string;
  filename: string;
  duration_sec: number;
  fps?: number;
  resolution?: string;
  status: "UPLOADED" | "PROCESSING" | "PROCESSED" | "FAILED";
  segment_count?: number;
  event_count?: number;
  created_at: string;
  processed_at?: string;
}

export interface VideoListResponse {
  videos: VideoMeta[];
}

export type SearchMode = "hybrid" | "visual" | "speech";

export interface SearchRequest {
  query: string;
  video_id?: string;
  top_k?: number;
  source_filter?: string;
  search_mode?: SearchMode;
  min_score?: number;
  agentic?: boolean;
}

/** Knowledge graph types */
export interface GraphNode {
  id: string;
  label: string;
  type: "person" | "vehicle" | "location" | "object";
  count: number;
  first_seen: number;
  last_seen: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  action: string;
  timestamp: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** Event timeline types */
export interface VideoEvent {
  id: string;
  video_id: string;
  event_type: string;
  start_time: number;
  end_time: number;
  description: string;
  entities: string[];
}

/** Annotation types */
export interface Annotation {
  id: string;
  video_id: string;
  start_time: number;
  end_time: number;
  label: string;
  note: string;
  color: string;
  created_at: string;
}
