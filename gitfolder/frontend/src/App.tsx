import { useCallback, useMemo, useState } from "react";
import SearchBar from "./components/SearchBar";
import VideoPlayer from "./components/VideoPlayer";
import TimelineMarkers from "./components/TimelineMarkers";
import ResultsPanel from "./components/ResultsPanel";
import { useSearch } from "./hooks/useSearch";
import { useVideo } from "./hooks/useVideo";
import { getVideoStreamUrl } from "./api/client";

export default function App() {
  const { results, isLoading, error, query, search, clear } = useSearch();
  const {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    seekTo,
    onTimeUpdate,
    onLoadedMetadata,
    togglePlay,
  } = useVideo();

  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  // Derive the video URL from the first result's video_id
  const videoUrl = useMemo(() => {
    if (results.length === 0) return null;
    return getVideoStreamUrl(results[0].video_id);
  }, [results]);

  const activeResult = activeIndex !== null ? results[activeIndex] ?? null : null;

  const handleResultClick = useCallback(
    (index: number) => {
      setActiveIndex(index);
      const result = results[index];
      if (result) seekTo(result.start_time);
    },
    [results, seekTo]
  );

  const handleSearch = useCallback(
    (q: string) => {
      setActiveIndex(null);
      search(q);
    },
    [search]
  );

  const handleClear = useCallback(() => {
    setActiveIndex(null);
    clear();
  }, [clear]);

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>
          <span style={styles.titleAccent}>Cogni</span>Stream
        </h1>
        <p style={styles.subtitle}>Multimodal Video Retrieval</p>
      </header>

      {/* Search */}
      <section style={styles.searchSection}>
        <SearchBar onSearch={handleSearch} onClear={handleClear} isLoading={isLoading} />
      </section>

      {/* Main content: player + results side by side */}
      <main style={styles.main}>
        {/* Left column: video + timeline */}
        <div style={styles.playerColumn}>
          <VideoPlayer
            videoUrl={videoUrl}
            videoRef={videoRef}
            currentTime={currentTime}
            duration={duration}
            isPlaying={isPlaying}
            onTimeUpdate={onTimeUpdate}
            onLoadedMetadata={onLoadedMetadata}
            togglePlay={togglePlay}
            activeResult={activeResult}
          />
          <TimelineMarkers
            results={results}
            duration={duration}
            currentTime={currentTime}
            activeIndex={activeIndex}
            onMarkerClick={handleResultClick}
          />
        </div>

        {/* Right column: results */}
        <div style={styles.resultsColumn}>
          <ResultsPanel
            results={results}
            query={query}
            isLoading={isLoading}
            error={error}
            activeIndex={activeIndex}
            onResultClick={handleResultClick}
          />
        </div>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  app: {
    maxWidth: "1280px",
    margin: "0 auto",
    padding: "24px 32px",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    color: "#1e293b",
    minHeight: "100vh",
  },
  header: {
    marginBottom: "24px",
  },
  title: {
    fontSize: "28px",
    fontWeight: 800,
    margin: 0,
    letterSpacing: "-0.02em",
  },
  titleAccent: {
    color: "#3b82f6",
  },
  subtitle: {
    fontSize: "14px",
    color: "#64748b",
    margin: "4px 0 0 0",
  },
  searchSection: {
    marginBottom: "24px",
  },
  main: {
    display: "flex",
    gap: "28px",
    alignItems: "flex-start",
  },
  playerColumn: {
    flex: "1 1 58%",
    minWidth: 0,
  },
  resultsColumn: {
    flex: "1 1 42%",
    minWidth: 0,
  },
};
