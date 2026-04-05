import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import KnowledgeGraph from "../KnowledgeGraph";
import * as apiClient from "../../api/client";

vi.mock("../../api/client", () => ({
  getVideoGraph: vi.fn(),
}));

beforeAll(() => {
  class MockResizeObserver {
    private readonly callback: ResizeObserverCallback;

    constructor(callback: ResizeObserverCallback) {
      this.callback = callback;
    }

    observe(target: Element) {
      this.callback(
        [
          {
            target,
            contentRect: {
              width: 640,
              height: 380,
              top: 0,
              left: 0,
              bottom: 380,
              right: 640,
              x: 0,
              y: 0,
              toJSON: () => ({}),
            },
          } as ResizeObserverEntry,
        ],
        this as unknown as ResizeObserver
      );
    }

    disconnect() {}
    unobserve() {}
  }

  Object.defineProperty(window, "ResizeObserver", {
    writable: true,
    value: MockResizeObserver,
  });

  Object.defineProperty(window, "requestAnimationFrame", {
    writable: true,
    value: vi.fn(() => 1),
  });

  Object.defineProperty(window, "cancelAnimationFrame", {
    writable: true,
    value: vi.fn(),
  });

  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    setTransform: vi.fn(),
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

describe("KnowledgeGraph", () => {
  const mockGetVideoGraph = vi.mocked(apiClient.getVideoGraph);

  beforeEach(() => {
    mockGetVideoGraph.mockResolvedValue({
      nodes: [
        {
          id: "car",
          label: "car",
          type: "vehicle",
          count: 3,
          first_seen: 1,
          last_seen: 10,
        },
        {
          id: "traffic_light",
          label: "traffic_light",
          type: "object",
          count: 1,
          first_seen: 1,
          last_seen: 6,
        },
      ],
      edges: [
        {
          source: "car",
          target: "traffic_light",
          action: "stopping",
          timestamp: 4,
        },
      ],
    });
  });

  it("renders relationship preview data from the API", async () => {
    render(<KnowledgeGraph videoId="video-1" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/2 entities and 1 relationships/i)).toBeInTheDocument();
      expect(screen.getByText(/relationship preview/i)).toBeInTheDocument();
      expect(screen.getByText("stopping")).toBeInTheDocument();
      expect(screen.getByText(/car -> traffic light/i)).toBeInTheDocument();
    });
  });

  it("selects a node when the canvas is clicked", async () => {
    render(<KnowledgeGraph videoId="video-1" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/relationship preview/i)).toBeInTheDocument();
    });

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
    Object.defineProperty(canvas, "getBoundingClientRect", {
      value: () => ({
        left: 0,
        width: 640,
        top: 0,
        height: 380,
        right: 640,
        bottom: 380,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }),
    });

    fireEvent.mouseDown(canvas!, { clientX: 426, clientY: 190 });

    await waitFor(() => {
      expect(screen.getByText(/^car$/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /clear/i })).toBeInTheDocument();
    });
  });
});
