import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import StatsPanel from "../StatsPanel";
import * as apiClient from "../../api/client";

vi.mock("../../api/client", () => ({
  getStats: vi.fn(),
}));

describe("StatsPanel", () => {
  const mockStats = {
    videos: { total: 5, by_status: { PROCESSED: 3, UPLOADED: 2 }, total_duration_human: "12.5m" },
    segments: { total: 489 },
    live_feeds: { active: 1, total: 2 },
    config: {
      pipeline_mode: "fast",
      nvidia_cloud: false,
      vlm_model: "moondream",
      stt_model: "whisper-small",
      embedding_model: "bge-small",
    },
  };
  const mockGetStats = vi.mocked(apiClient.getStats);

  beforeEach(() => {
    mockGetStats.mockResolvedValue(mockStats);
  });

  it("renders stat cards after loading", async () => {
    render(<StatsPanel />);
    await waitFor(() => {
      expect(screen.getByText("5")).toBeInTheDocument();
      expect(screen.getByText("Videos")).toBeInTheDocument();
      expect(screen.getByText("489")).toBeInTheDocument();
      expect(screen.getByText("Segments")).toBeInTheDocument();
    });
  });

  it("shows live feed count", async () => {
    render(<StatsPanel />);
    await waitFor(() => {
      expect(screen.getByText("1")).toBeInTheDocument();
      expect(screen.getByText("Live Feeds")).toBeInTheDocument();
    });
  });

  it("shows pipeline mode", async () => {
    render(<StatsPanel />);
    await waitFor(() => {
      expect(screen.getByText("fast")).toBeInTheDocument();
      expect(screen.getByText("Mode")).toBeInTheDocument();
    });
  });

  it("renders nothing when API fails", () => {
    mockGetStats.mockRejectedValue(new Error("fail"));
    const { container } = render(<StatsPanel />);
    expect(container.innerHTML).toBe("");
  });
});
