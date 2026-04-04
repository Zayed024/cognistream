import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import StatsPanel from "../StatsPanel";

// Mock axios
vi.mock("axios", () => {
  const mockApi = {
    get: vi.fn(),
    create: vi.fn(() => mockApi),
  };
  return { default: mockApi };
});

import axios from "axios";
const mockAxios = axios as unknown as { get: ReturnType<typeof vi.fn> };

describe("StatsPanel", () => {
  const mockStats = {
    data: {
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
    },
  };

  beforeEach(() => {
    mockAxios.get.mockResolvedValue(mockStats);
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
    mockAxios.get.mockRejectedValue(new Error("fail"));
    const { container } = render(<StatsPanel />);
    expect(container.innerHTML).toBe("");
  });
});
