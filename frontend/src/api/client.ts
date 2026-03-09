import axios from "axios";
import type { SearchRequest, SearchResponse, VideoMeta } from "../types";

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
