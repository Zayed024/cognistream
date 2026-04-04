import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { createRef } from "react";
import VideoPlayer from "../VideoPlayer";

describe("VideoPlayer", () => {
  const defaultProps = {
    videoUrl: null,
    videoRef: createRef<HTMLVideoElement>(),
    currentTime: 0,
    duration: 0,
    isPlaying: false,
    onTimeUpdate: vi.fn(),
    onLoadedMetadata: vi.fn(),
    togglePlay: vi.fn(),
    activeResult: null,
  };

  it("shows placeholder when no video URL", () => {
    render(<VideoPlayer {...defaultProps} />);
    expect(screen.getByText(/select a video/i)).toBeInTheDocument();
  });

  it("renders video element when URL provided", () => {
    render(<VideoPlayer {...defaultProps} videoUrl="/api/video/123/stream" />);
    const video = document.querySelector("video");
    expect(video).toBeInTheDocument();
    expect(video?.src).toContain("/api/video/123/stream");
  });

  it("renders play button", () => {
    render(<VideoPlayer {...defaultProps} videoUrl="/stream" />);
    // The play/pause button is always present when a video URL is provided
    const btn = document.querySelector("button");
    expect(btn).toBeInTheDocument();
  });
});
