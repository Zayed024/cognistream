import axios from "axios";
import type { SearchRequest, SearchResponse, VideoMeta, VideoListResponse } from "../types";

const api = axios.create({
  baseURL: "/api",
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

export async function searchVideos(
  request: SearchRequest
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>("/search", request);
  return data;
}

export async function getVideoMeta(videoId: string): Promise<VideoMeta> {
  const { data } = await api.get<VideoMeta>(`/video/${videoId}`);
  return data;
}

/** Returns the streaming URL for the HTML5 video element */
export function getVideoStreamUrl(videoId: string): string {
  return `/api/video/${videoId}/stream`;
}

/** Returns the URL for a keyframe thumbnail */
export function getFrameUrl(videoId: string, frameName: string): string {
  return `/api/video/${videoId}/frame/${frameName}`;
}

/** List all videos in the database */
export async function listVideos(): Promise<VideoMeta[]> {
  const { data } = await api.get<VideoListResponse>("/videos");
  return data.videos;
}

/** Upload a video file */
export async function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<{ video_id: string; filename: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post("/ingest-video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 600000, // 10 min for large uploads
    onUploadProgress: (event) => {
      if (event.total && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    },
  });
  return data;
}

/** Trigger video processing pipeline */
export async function processVideo(videoId: string): Promise<void> {
  await api.post("/process-video", { video_id: videoId });
}

/** Delete a video and all its data */
export async function deleteVideo(videoId: string): Promise<void> {
  await api.delete(`/video/${videoId}`);
}

/** Video processing progress */
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

/** Get processing progress for a video */
export async function getVideoProgress(videoId: string): Promise<VideoProgress> {
  const { data } = await api.get<VideoProgress>(`/video/${videoId}/progress`);
  return data;
}

/** Subscribe to progress updates via Server-Sent Events */
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

  // Return cleanup function
  return () => eventSource.close();
}
