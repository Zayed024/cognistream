import { useCallback, useRef, useState } from "react";

interface UseVideoReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  seekTo: (time: number, options?: { autoplay?: boolean }) => void;
  onTimeUpdate: () => void;
  onLoadedMetadata: () => void;
  togglePlay: () => void;
}

export function useVideo(): UseVideoReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const seekTo = useCallback((time: number, options?: { autoplay?: boolean }) => {
    const el = videoRef.current;
    if (!el) return;
    const maxTime = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : duration || time;
    const nextTime = Math.max(0, Math.min(time, maxTime));
    el.currentTime = nextTime;
    setCurrentTime(nextTime);

    if (options?.autoplay) {
      el.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [duration]);

  const onTimeUpdate = useCallback(() => {
    const el = videoRef.current;
    if (el) setCurrentTime(el.currentTime);
  }, []);

  const onLoadedMetadata = useCallback(() => {
    const el = videoRef.current;
    if (el) {
      setDuration(el.duration);
      setCurrentTime(el.currentTime);
    }
  }, []);

  const togglePlay = useCallback(() => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(() => {});
      setIsPlaying(true);
    } else {
      el.pause();
      setIsPlaying(false);
    }
  }, []);

  return {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    seekTo,
    onTimeUpdate,
    onLoadedMetadata,
    togglePlay,
  };
}
