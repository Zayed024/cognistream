import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import SearchBar from "../SearchBar";

const baseProps = {
  onSearch: vi.fn(),
  onClear: vi.fn(),
  isLoading: false,
  searchMode: "hybrid" as const,
  onSearchModeChange: vi.fn(),
};

describe("SearchBar", () => {
  it("renders input and buttons", () => {
    render(<SearchBar {...baseProps} />);
    expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
  });

  it("calls onSearch when form is submitted", () => {
    const onSearch = vi.fn();
    render(<SearchBar {...baseProps} onSearch={onSearch} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.change(input, { target: { value: "red car" } });
    fireEvent.submit(input.closest("form")!);
    expect(onSearch).toHaveBeenCalledWith("red car");
  });

  it("disables input when loading", () => {
    render(<SearchBar {...baseProps} isLoading={true} />);
    const input = screen.getByPlaceholderText(/search/i);
    expect(input).toBeDisabled();
  });

  it("does not call onSearch with empty query", () => {
    const onSearch = vi.fn();
    render(<SearchBar {...baseProps} onSearch={onSearch} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.submit(input.closest("form")!);
    expect(onSearch).not.toHaveBeenCalled();
  });

  it("renders search mode toggle buttons", () => {
    render(<SearchBar {...baseProps} />);
    expect(screen.getByText(/Visual/)).toBeInTheDocument();
    expect(screen.getByText(/Speech/)).toBeInTheDocument();
    expect(screen.getByText(/All/)).toBeInTheDocument();
  });

  it("calls onSearchModeChange when mode button clicked", () => {
    const onModeChange = vi.fn();
    render(<SearchBar {...baseProps} onSearchModeChange={onModeChange} />);
    fireEvent.click(screen.getByText(/Speech/));
    expect(onModeChange).toHaveBeenCalledWith("speech");
  });
});
