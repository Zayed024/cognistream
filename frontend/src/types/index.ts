/** Mirrors backend SearchResult dataclass */
export interface SearchResult {
  video_id: string;
  segment_id: string;
  start_time: number;
  end_time: number;
  text: string;
  source_type: "visual" | "audio" | "fused" | "event";
  score: number;
  event_type?: string;
  frame_url?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface VideoMeta {
  video_id: string;
  filename: string;
  duration_sec: number;
  fps: number;
  resolution: string;
  status: "UPLOADED" | "PROCESSING" | "PROCESSED" | "FAILED";
  segment_count: number;
  event_count: number;
  created_at: string;
  processed_at?: string;
}

export interface SearchRequest {
  query: string;
  video_id?: string;
  top_k?: number;
  source_filter?: string;
}
