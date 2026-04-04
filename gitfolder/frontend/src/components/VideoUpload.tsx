import { useState, useRef, useCallback } from "react";
import { uploadVideo } from "../api/client";

interface VideoUploadProps {
  onComplete: (videoId: string, shouldProcess: boolean) => void;
  onCancel: () => void;
}

const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
const MAX_SIZE_BYTES = 2048 * 1024 * 1024; // 2 GB

function validateFile(f: File): string | null {
  const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `Unsupported format. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`;
  }
  if (f.size > MAX_SIZE_BYTES) {
    return "File too large. Maximum size: 2 GB";
  }
  return null;
}

export default function VideoUpload({ onComplete, onCancel }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [autoProcess, setAutoProcess] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      const err = validateFile(droppedFile);
      if (err) {
        setError(err);
      } else {
        setFile(droppedFile);
        setError(null);
      }
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const err = validateFile(selectedFile);
      if (err) {
        setError(err);
      } else {
        setFile(selectedFile);
        setError(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const result = await uploadVideo(file, setProgress);
      setProgress(100);
      // Pass video_id and autoProcess to parent - parent handles processing
      onComplete(result.video_id, autoProcess);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setUploading(false);
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>Upload Video</h2>
          <button onClick={onCancel} style={styles.closeBtn} disabled={uploading}>
            ×
          </button>
        </div>

        {/* Drop zone */}
        <div
          style={{
            ...styles.dropzone,
            borderColor: dragActive ? "#3b82f6" : "#e2e8f0",
            background: dragActive ? "#eff6ff" : "#f8fafc",
          }}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => !uploading && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            onChange={handleFileSelect}
            style={{ display: "none" }}
            disabled={uploading}
          />

          {file ? (
            <div style={styles.fileInfo}>
              <svg style={styles.fileIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <path d="M10 9l5 3-5 3V9z" />
              </svg>
              <div>
                <p style={styles.fileName}>{file.name}</p>
                <p style={styles.fileSize}>{formatFileSize(file.size)}</p>
              </div>
              {!uploading && (
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  style={styles.removeBtn}
                >
                  Remove
                </button>
              )}
            </div>
          ) : (
            <>
              <svg style={styles.uploadIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p style={styles.dropText}>
                Drag and drop a video file, or <span style={styles.browseLink}>browse</span>
              </p>
              <p style={styles.dropSubtext}>MP4, MOV, AVI, MKV, WebM up to 2 GB</p>
            </>
          )}
        </div>

        {error && (
          <div style={styles.error}>
            {error}
          </div>
        )}

        {/* Progress bar */}
        {uploading && (
          <div style={styles.progressContainer}>
            <div style={styles.progressTrack}>
              <div style={{ ...styles.progressFill, width: `${progress}%` }} />
            </div>
            <span style={styles.progressText}>
              {progress < 100 ? `Uploading... ${progress}%` : "Starting processing..."}
            </span>
          </div>
        )}

        {/* Options */}
        <label style={styles.checkbox}>
          <input
            type="checkbox"
            checked={autoProcess}
            onChange={(e) => setAutoProcess(e.target.checked)}
            disabled={uploading}
          />
          <span>Start processing immediately after upload</span>
        </label>

        {/* Actions */}
        <div style={styles.actions}>
          <button onClick={onCancel} style={styles.cancelBtn} disabled={uploading}>
            Cancel
          </button>
          <button
            onClick={handleUpload}
            style={{
              ...styles.uploadBtn,
              opacity: !file || uploading ? 0.6 : 1,
            }}
            disabled={!file || uploading}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────── */

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/* ── Styles ──────────────────────────────────────────────────── */

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    padding: "20px",
  },
  modal: {
    background: "#fff",
    borderRadius: "16px",
    width: "100%",
    maxWidth: "500px",
    padding: "24px",
    boxShadow: "0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
  },
  title: {
    fontSize: "18px",
    fontWeight: 700,
    margin: 0,
    color: "#1e293b",
  },
  closeBtn: {
    background: "none",
    border: "none",
    fontSize: "24px",
    color: "#94a3b8",
    cursor: "pointer",
    padding: "0 4px",
    lineHeight: 1,
  },
  dropzone: {
    border: "2px dashed",
    borderRadius: "12px",
    padding: "32px 24px",
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  uploadIcon: {
    width: "40px",
    height: "40px",
    color: "#94a3b8",
    marginBottom: "12px",
  },
  dropText: {
    fontSize: "14px",
    color: "#475569",
    margin: "0 0 4px 0",
  },
  browseLink: {
    color: "#3b82f6",
    fontWeight: 500,
  },
  dropSubtext: {
    fontSize: "12px",
    color: "#94a3b8",
    margin: 0,
  },
  fileInfo: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    textAlign: "left",
  },
  fileIcon: {
    width: "40px",
    height: "40px",
    color: "#3b82f6",
    flexShrink: 0,
  },
  fileName: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#1e293b",
    margin: "0 0 2px 0",
    wordBreak: "break-all",
  },
  fileSize: {
    fontSize: "12px",
    color: "#64748b",
    margin: 0,
  },
  removeBtn: {
    marginLeft: "auto",
    background: "none",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    padding: "6px 12px",
    fontSize: "12px",
    color: "#64748b",
    cursor: "pointer",
  },
  error: {
    marginTop: "12px",
    padding: "10px 12px",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "8px",
    color: "#dc2626",
    fontSize: "13px",
  },
  progressContainer: {
    marginTop: "16px",
  },
  progressTrack: {
    height: "8px",
    background: "#e2e8f0",
    borderRadius: "4px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#3b82f6",
    borderRadius: "4px",
    transition: "width 0.3s ease",
  },
  progressText: {
    display: "block",
    marginTop: "6px",
    fontSize: "12px",
    color: "#64748b",
  },
  checkbox: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    marginTop: "16px",
    fontSize: "13px",
    color: "#475569",
    cursor: "pointer",
  },
  actions: {
    display: "flex",
    gap: "12px",
    marginTop: "20px",
    paddingTop: "16px",
    borderTop: "1px solid #f1f5f9",
  },
  cancelBtn: {
    flex: 1,
    padding: "12px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    fontSize: "14px",
    fontWeight: 500,
    color: "#475569",
    cursor: "pointer",
  },
  uploadBtn: {
    flex: 1,
    padding: "12px",
    background: "#3b82f6",
    border: "none",
    borderRadius: "8px",
    fontSize: "14px",
    fontWeight: 600,
    color: "#fff",
    cursor: "pointer",
  },
};
