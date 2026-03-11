import { useCallback, useMemo, useState } from "react";
import SearchBar from "./components/SearchBar";
import VideoPlayer from "./components/VideoPlayer";
import TimelineMarkers from "./components/TimelineMarkers";
import ResultsPanel from "./components/ResultsPanel";
import VideoList from "./components/VideoList";
import VideoUpload from "./components/VideoUpload";
import { useSearch } from "./hooks/useSearch";
import { useVideo } from "./hooks/useVideo";
import { getVideoStreamUrl, processVideo } from "./api/client";
import type { VideoMeta } from "./types";

type View = "list" | "search";

export default function App() {
  // Navigation state
  const [view, setView] = useState<View>("list");
  const [selectedVideo, setSelectedVideo] = useState<VideoMeta | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  // Search hook - pass video_id when in search view
  const { results, isLoading, error, query, search, clear } = useSearch(
    selectedVideo?.video_id
  );
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

  // Video URL: use selected video when in search view
  const videoUrl = useMemo(() => {
    if (selectedVideo) {
      return getVideoStreamUrl(selectedVideo.video_id);
    }
    return null;
  }, [selectedVideo]);

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

  const handleSelectVideo = useCallback((video: VideoMeta) => {
    setSelectedVideo(video);
    setActiveIndex(null);
    clear();
    setView("search");
  }, [clear]);

  const handleBackToList = useCallback(() => {
    setView("list");
    setSelectedVideo(null);
    setActiveIndex(null);
    clear();
  }, [clear]);

  // Track refresh trigger for VideoList
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadComplete = useCallback((videoId: string, shouldProcess: boolean) => {
    setShowUpload(false);
    // Trigger refresh so VideoList fetches new video
    setRefreshTrigger((prev) => prev + 1);
    // Start processing if requested - fire and forget
    if (shouldProcess) {
      processVideo(videoId).catch((err) => {
        console.error("Failed to start processing:", err);
      });
    }
  }, []);

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          {view === "search" && (
            <button onClick={handleBackToList} style={styles.backBtn}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back
            </button>
          )}
          <div>
            <h1 style={styles.title}>
              <span style={styles.titleAccent}>Cogni</span>Stream
            </h1>
            <p style={styles.subtitle}>
              {view === "search" && selectedVideo
                ? selectedVideo.filename
                : "Multimodal Video Retrieval"}
            </p>
          </div>
        </div>
      </header>

      {/* Main content */}
      {view === "list" ? (
        <VideoList
          onSelectVideo={handleSelectVideo}
          onUploadClick={() => setShowUpload(true)}
          refreshTrigger={refreshTrigger}
        />
      ) : (
        <>
          {/* Search */}
          <section style={styles.searchSection}>
            <SearchBar onSearch={handleSearch} onClear={handleClear} isLoading={isLoading} />
          </section>

          {/* Player + Results */}
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
        </>
      )}

      {/* Upload modal */}
      {showUpload && (
        <VideoUpload
          onComplete={handleUploadComplete}
          onCancel={() => setShowUpload(false)}
        />
      )}
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
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  headerLeft: {
    display: "flex",
    alignItems: "flex-start",
    gap: "16px",
  },
  backBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 14px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: 500,
    color: "#475569",
    cursor: "pointer",
    marginTop: "4px",
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
