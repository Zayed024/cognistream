import axios from "axios";
import type {
  SearchRequest, SearchResponse, VideoMeta, VideoListResponse,
  GraphData, VideoEvent, Annotation, SearchResult,
} from "../types";

const api = axios.create({
  baseURL: "/api",
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ── Search ──────────────────────────────────────────────────

export async function searchVideos(
  request: SearchRequest
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>("/search", request);
  return data;
}

// ── Video CRUD ──────────────────────────────────────────────

export async function getVideoMeta(videoId: string): Promise<VideoMeta> {
  const { data } = await api.get<VideoMeta>(`/video/${videoId}`);
  return data;
}

export function getVideoStreamUrl(videoId: string): string {
  return `/api/video/${videoId}/stream`;
}

export function getFrameUrl(videoId: string, frameName: string): string {
  return `/api/video/${videoId}/frame/${frameName}`;
}

export function getThumbnailUrl(videoId: string): string {
  return `/api/video/${videoId}/thumbnail`;
}

export async function listVideos(): Promise<VideoMeta[]> {
  const { data } = await api.get<VideoListResponse>("/videos");
  return data.videos;
}

export async function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<{ video_id: string; filename: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post("/ingest-video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 600000,
    onUploadProgress: (event) => {
      if (event.total && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    },
  });
  return data;
}

export async function processVideo(videoId: string): Promise<void> {
  await api.post("/process-video", { video_id: videoId });
}

export async function deleteVideo(videoId: string): Promise<void> {
  await api.delete(`/video/${videoId}`);
}

// ── Batch processing ────────────────────────────────────────

export async function processBatch(videoIds: string[]): Promise<{
  queued: string[];
  queue_size: number;
  errors: { video_id: string; error: string }[];
}> {
  const { data } = await api.post("/process-batch", { video_ids: videoIds });
  return data;
}

export async function getProcessQueue(): Promise<{
  queue: string[];
  queue_size: number;
  is_busy: boolean;
}> {
  const { data } = await api.get("/process-queue");
  return data;
}

// ── Progress SSE ────────────────────────────────────────────

export interface VideoProgress {
  video_id: string;
  stage: string;
  stage_number: number;
  total_stages: number;
  percent: number;
  elapsed_sec?: number;
  done?: boolean;
  error?: boolean;
}

export async function getVideoProgress(videoId: string): Promise<VideoProgress> {
  const { data } = await api.get<VideoProgress>(`/video/${videoId}/progress`);
  return data;
}

export function subscribeToProgress(
  videoId: string,
  onProgress: (progress: VideoProgress) => void,
  onError?: (error: Event) => void
): () => void {
  const eventSource = new EventSource(`/api/video/${videoId}/progress/stream`);

  eventSource.onmessage = (event) => {
    try {
      const progress = JSON.parse(event.data) as VideoProgress;
      onProgress(progress);
      if (progress.done) {
        eventSource.close();
      }
    } catch (e) {
      console.error("Failed to parse progress event:", e);
    }
  };

  eventSource.onerror = (error) => {
    onError?.(error);
    eventSource.close();
  };

  return () => eventSource.close();
}

// ── Find similar ────────────────────────────────────────────

export async function findSimilar(
  segmentId: string,
  topK = 10,
  videoId?: string
): Promise<SearchResult[]> {
  const { data } = await api.post<{ results: SearchResult[] }>("/similar", {
    segment_id: segmentId,
    top_k: topK,
    video_id: videoId,
  });
  return data.results;
}

// ── Clip export ─────────────────────────────────────────────

export function getClipExportUrl(
  videoId: string,
  startTime: number,
  endTime: number
): string {
  // We'll POST to get the clip, but for direct download we need a form submit
  return `/api/video/${videoId}/clip`;
}

export async function exportClip(
  videoId: string,
  startTime: number,
  endTime: number
): Promise<Blob> {
  const { data } = await api.post(
    `/video/${videoId}/clip`,
    { start_time: startTime, end_time: endTime },
    { responseType: "blob", timeout: 120000 }
  );
  return data;
}

// ── Knowledge graph ─────────────────────────────────────────

export async function getVideoGraph(videoId: string): Promise<GraphData> {
  const { data } = await api.get<GraphData>(`/video/${videoId}/graph`);
  return data;
}

// ── Events ──────────────────────────────────────────────────

export async function getVideoEvents(
  videoId: string
): Promise<VideoEvent[]> {
  const { data } = await api.get<{ events: VideoEvent[] }>(
    `/video/${videoId}/events`
  );
  return data.events;
}

// ── Annotations ─────────────────────────────────────────────

export async function getAnnotations(videoId: string): Promise<Annotation[]> {
  const { data } = await api.get<{ annotations: Annotation[] }>(
    `/video/${videoId}/annotations`
  );
  return data.annotations;
}

export async function createAnnotation(ann: {
  video_id: string;
  start_time: number;
  end_time: number;
  label: string;
  note?: string;
  color?: string;
}): Promise<Annotation> {
  const { data } = await api.post<Annotation>("/annotations", ann);
  return data;
}

export async function deleteAnnotation(annotationId: string): Promise<void> {
  await api.delete(`/annotations/${annotationId}`);
}

// ── Live feeds ──────────────────────────────────────────────

export interface LiveFeedInfo {
  video_id: string;
  url: string;
  state: string;
  chunks_processed: number;
  total_segments: number;
  fps: number;
  started_at: string;
  last_chunk_at: string;
  error: string;
}

export interface LiveWsEvent {
  video_id: string;
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export async function startLiveFeed(
  url: string,
  videoId: string,
  chunkSec = 15
): Promise<{ video_id: string; state: string; message: string }> {
  const { data } = await api.post("/live/start", {
    url,
    video_id: videoId,
    chunk_sec: chunkSec,
  });
  return data;
}

export async function stopLiveFeed(
  videoId: string
): Promise<{ video_id: string; message: string }> {
  const { data } = await api.post("/live/stop", { video_id: videoId });
  return data;
}

export async function getLiveFeedStatus(): Promise<LiveFeedInfo[]> {
  const { data } = await api.get<{ feeds: LiveFeedInfo[] }>("/live/status");
  return data.feeds;
}

export function connectLiveWebSocket(
  videoId: string,
  onEvent: (event: LiveWsEvent) => void,
  onClose?: () => void
): { send: (msg: Record<string, unknown>) => void; close: () => void } {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/live/${videoId}`);

  ws.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as LiveWsEvent;
      if (parsed.event_type !== "ping") {
        onEvent(parsed);
      }
    } catch {
      // ignore parse errors
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();

  return {
    send: (msg) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
      }
    },
    close: () => ws.close(),
  };
}

// ── Browser camera (phone / screen share) ───────────────────

export async function uploadBrowserChunk(
  videoId: string,
  chunkIndex: number,
  chunkStart: number,
  blob: Blob
): Promise<{ segments_stored: number; keyframes_extracted: number }> {
  const formData = new FormData();
  formData.append("file", blob, `chunk_${chunkIndex}.webm`);

  const { data } = await api.post(
    `/live/browser-chunk?video_id=${encodeURIComponent(videoId)}&chunk_index=${chunkIndex}&chunk_start=${chunkStart}`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" }, timeout: 120000 }
  );
  return data;
}

export async function stopBrowserFeed(
  videoId: string
): Promise<{ total_chunks: number; message: string }> {
  const { data } = await api.post(
    `/live/browser-stop?video_id=${encodeURIComponent(videoId)}`
  );
  return data;
}
