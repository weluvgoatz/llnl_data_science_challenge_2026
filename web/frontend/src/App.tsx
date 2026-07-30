import {
  Box,
  Check,
  ChevronLeft,
  Download,
  FileArchive,
  FileJson,
  FileText,
  LoaderCircle,
  Maximize2,
  Play,
  RotateCcw,
  TriangleAlert,
  UploadCloud,
  X,
} from "lucide-react";
import { DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { StlViewer } from "./components/StlViewer";
import { DefectViewer } from "./components/DefectViewer";
import { DesignGraphViewer } from "./components/DesignGraphViewer";
import { ZoomableImage } from "./components/ZoomableImage";
import { VERDICT_COLORS, VERDICT_DESCRIPTIONS, VERDICT_LABELS, VERDICT_ORDER } from "./defects";
import type {
  ChatTurn,
  DefectClassification,
  DefectStage,
  InputFile,
  Job,
  MountedSurface,
  StrutVerdict,
  TimelineEntry,
} from "./types";

const DEFECT_STAGE_LABELS: Record<DefectStage, string> = {
  segmenting: "segmenting the scan",
  cleaning: "cleaning the segmentation",
  skeletonizing: "skeletonizing the lattice",
  building_graph: "building the as-built graph",
  classifying: "classifying every strut",
  bend_detail: "measuring strut bow",
  complete: "finishing up",
};

const SURFACE_LABELS: Record<MountedSurface["component"], string> = {
  ModelViewer: "3D model",
  DefectView: "Defect view",
  ReportView: "Report",
  DataViz: "Data",
};

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function fileIcon(kind: InputFile["kind"]) {
  if (kind === "stl") return <Box size={20} />;
  if (kind === "json") return <FileJson size={20} />;
  return <FileArchive size={20} />;
}

function MarkdownPreview({ url }: { url: string }) {
  const [content, setContent] = useState("Loading report…");
  useEffect(() => {
    fetch(url)
      .then((response) => response.text())
      .then(setContent)
      .catch(() => setContent("The report preview could not be loaded."));
  }, [url]);
  return <pre className="report-preview">{content}</pre>;
}

function UploadScreen({ onComplete, busy }: { onComplete: (files: File[]) => Promise<void>; busy: boolean }) {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const supported = new Set(["json", "tif", "tiff", "stl"]);

  const addFiles = (incoming: File[]) => {
    setError("");
    const invalid = incoming.find((file) => !supported.has(file.name.split(".").pop()?.toLowerCase() || ""));
    if (invalid) {
      setError(`${invalid.name} is not a supported JSON, TIFF, or STL file.`);
      return;
    }
    setFiles((current) => {
      const keys = new Set(current.map((file) => `${file.name}:${file.size}`));
      return [...current, ...incoming.filter((file) => !keys.has(`${file.name}:${file.size}`))];
    });
  };
  const drop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  };

  return (
    <section className="page upload-page">
      <div className="page-heading centered">
        <span className="eyebrow">New inspection</span>
        <h1>Bring your lattice into focus.</h1>
        <p>Upload one file or a matched set. Once they&rsquo;re in, just tell the agent what you want to see.</p>
      </div>
      <label
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <input
          type="file"
          multiple
          accept=".json,.tif,.tiff,.stl"
          onChange={(event) => addFiles(Array.from(event.target.files || []))}
        />
        <div className="upload-mark">
          <UploadCloud size={27} />
        </div>
        <h2>Drop lattice files here</h2>
        <p>
          or <span>browse your computer</span>
        </p>
        <small>JSON · TIFF stacks · STL meshes &nbsp;·&nbsp; 2 GB maximum per file</small>
      </label>
      {error && <div className="error-banner">{error}</div>}
      {files.length > 0 && (
        <div className="queue">
          <div className="queue-header">
            <h3>
              Ready to upload <span>{files.length}</span>
            </h3>
            <button className="text-button" onClick={() => setFiles([])}>
              Clear all
            </button>
          </div>
          {files.map((file, index) => {
            const extension = file.name.split(".").pop()?.toLowerCase();
            const kind = extension === "stl" ? "stl" : extension === "json" ? "json" : "tiff";
            return (
              <div className="queued-file" key={`${file.name}:${file.size}`}>
                <div className={`file-glyph ${kind}`}>{fileIcon(kind)}</div>
                <div className="file-copy">
                  <strong>{file.name}</strong>
                  <span>
                    {formatBytes(file.size)} · {kind.toUpperCase()}
                  </span>
                </div>
                <button
                  aria-label={`Remove ${file.name}`}
                  className="remove-button"
                  onClick={() => setFiles((current) => current.filter((_, item) => item !== index))}
                >
                  <X size={17} />
                </button>
              </div>
            );
          })}
          <button className="primary-button wide" disabled={busy} onClick={() => onComplete(files)}>
            {busy ? <LoaderCircle className="spin" size={19} /> : <UploadCloud size={19} />}
            {busy ? "Validating files…" : "Upload"}
          </button>
        </div>
      )}
    </section>
  );
}

function TiltBanner({ file }: { file: InputFile }) {
  switch (file.tiltStatus) {
    case "pending":
    case "checking":
      return (
        <div className="tilt-banner checking">
          <LoaderCircle className="spin" size={14} /> Checking slices for tilt…
        </div>
      );
    case "not_tilted":
      return (
        <div className="tilt-banner ok">
          <Check size={14} /> No meaningful tilt detected.
        </div>
      );
    case "corrected":
      return (
        <div className="tilt-banner corrected">
          <RotateCcw size={14} />
          Tilt detected (Z-Y {file.tiltZY?.toFixed(2)}°, Z-X {file.tiltZX?.toFixed(2)}°) — corrected copy shown below.
        </div>
      );
    case "failed":
      return (
        <div className="tilt-banner failed">
          <TriangleAlert size={14} /> Tilt check failed{file.tiltError ? `: ${file.tiltError}` : "."}
        </div>
      );
    default:
      return null;
  }
}

function TiffViewer({ file, initialIndex }: { file: InputFile; initialIndex?: number }) {
  const [index, setIndex] = useState(initialIndex ?? Math.floor((file.pageCount || 1) / 2));
  const [focused, setFocused] = useState<"original" | "corrected" | null>(null);
  const max = Math.max((file.pageCount || 1) - 1, 0);
  const originalUrl = `${file.sliceUrl}?index=${index}`;
  const correctedUrl = file.correctedSliceUrl ? `${file.correctedSliceUrl}?index=${index}` : null;
  const caption = `Slice ${index + 1} of ${max + 1}`;
  const comparing = file.tiltStatus === "corrected" && correctedUrl;
  const focusedUrl = focused === "corrected" ? correctedUrl : originalUrl;

  return (
    <div className="tiff-viewer">
      <TiltBanner file={file} />
      {comparing ? (
        <div className="slice-compare">
          <div className="slice-pane">
            <span className="slice-pane-label">Original</span>
            <button className="slice-frame" onClick={() => setFocused("original")} aria-label={`Enlarge original ${caption}`}>
              <img src={originalUrl} alt={`Original ${caption}`} />
            </button>
          </div>
          <div className="slice-pane">
            <span className="slice-pane-label">Tilt-fixed</span>
            <button className="slice-frame" onClick={() => setFocused("corrected")} aria-label={`Enlarge tilt-fixed ${caption}`}>
              <img src={correctedUrl} alt={`Tilt-fixed ${caption}`} />
            </button>
          </div>
        </div>
      ) : (
        <button className="slice-frame" onClick={() => setFocused("original")} aria-label={`Enlarge ${caption}`}>
          <img src={originalUrl} alt={caption} />
          <span className="viewer-hint">
            <Maximize2 size={13} /> Click to see the full frame
          </span>
        </button>
      )}
      <div className="slice-control">
        <label>
          Slice <strong>{index + 1}</strong> of {max + 1}
        </label>
        <input type="range" min="0" max={max} value={index} onChange={(e) => setIndex(Number(e.target.value))} />
      </div>
      {focused && focusedUrl && (
        <div className="lightbox" role="dialog" onClick={() => setFocused(null)}>
          <button aria-label="Close image">
            <X />
          </button>
          <img src={focusedUrl} alt={caption} />
          <strong>
            {focused === "corrected" ? "Tilt-fixed — " : comparing ? "Original — " : ""}
            {caption}
          </strong>
        </div>
      )}
    </div>
  );
}

function TiffCompareSection({ file }: { file: InputFile }) {
  const [index, setIndex] = useState(Math.floor((file.pageCount || 1) / 2));
  const max = Math.max((file.pageCount || 1) - 1, 0);
  const originalUrl = `${file.sliceUrl}?index=${index}`;
  const correctedUrl = file.correctedSliceUrl ? `${file.correctedSliceUrl}?index=${index}` : null;
  if (file.tiltStatus !== "corrected") return null;

  return (
    <section className="tiff-compare">
      <div className="page-heading">
        <span className="eyebrow">Tilt inspection</span>
        <h2>Original vs. tilt-corrected slices.</h2>
      </div>
      <div className="tiff-compare-grid">
        <div className="tiff-compare-pane">
          <span className="slice-pane-label">Original</span>
          <ZoomableImage src={originalUrl} alt={`Original slice ${index + 1} of ${file.name}`} />
        </div>
        <div className="tiff-compare-pane">
          <span className="slice-pane-label">Tilt-fixed</span>
          {correctedUrl && <ZoomableImage src={correctedUrl} alt={`Tilt-fixed slice ${index + 1} of ${file.name}`} />}
        </div>
      </div>
      <div className="slice-control">
        <label>
          Slice <strong>{index + 1}</strong> of {max + 1}
        </label>
        <input type="range" min="0" max={max} value={index} onChange={(event) => setIndex(Number(event.target.value))} />
      </div>
    </section>
  );
}

// The default/fallback surface: nothing mounted, just the uploaded files.
// Clicking a file previews it directly (a click is a request too, just
// routed locally instead of through the agent) -- chat handles everything
// derived or more complex; this handles the obvious case fast.
function FileListSurface({
  job,
  onPreview,
  onAnalyze,
  onAddFiles,
  addingFiles,
}: {
  job: Job;
  onPreview: (file: InputFile) => void;
  onAnalyze: () => void;
  onAddFiles: (files: File[]) => Promise<void>;
  addingFiles: boolean;
}) {
  const working = job.state === "analyzing";
  const canAddFiles = job.state === "intake_ready" && !addingFiles;
  return (
    <div className="file-list-surface">
      <div className="page-heading centered">
        <span className="eyebrow">Workspace</span>
        <h1>
          {job.files.length} file{job.files.length === 1 ? "" : "s"} ready.
        </h1>
        <p>
          Ask the agent to show a model, chart, or the defect map &mdash; or click a file below to preview it
          directly.
        </p>
      </div>
      {working && (
        <div className="hint-banner">
          <LoaderCircle className="spin" size={14} /> Analysis is running in the background &mdash; ask me how
          it&rsquo;s going, or check back here.
        </div>
      )}
      <div className="file-grid">
        {job.files.map((file) => (
          <button className="file-card" key={file.id} onClick={() => onPreview(file)}>
            <span className={`file-glyph ${file.kind}`}>{fileIcon(file.kind)}</span>
            <strong>{file.name}</strong>
            <span>{file.summary}</span>
          </button>
        ))}
        {canAddFiles && (
          <label className="file-card file-card-add">
            <input
              type="file"
              multiple
              accept=".json,.tif,.tiff,.stl"
              disabled={addingFiles}
              onChange={(event) => {
                const chosen = Array.from(event.target.files || []);
                event.target.value = "";
                if (chosen.length) onAddFiles(chosen);
              }}
            />
            {addingFiles ? <LoaderCircle className="spin" size={20} /> : <UploadCloud size={20} />}
            <strong>{addingFiles ? "Uploading…" : "Add a file"}</strong>
          </label>
        )}
      </div>
      {job.state === "intake_ready" &&
        job.files.some((file) => file.kind === "tiff") &&
        !job.files.some((file) => file.kind === "json") && (
          <div className="hint-banner">
            <TriangleAlert size={14} /> No design JSON uploaded &mdash; defect detection will be skipped this run
            (the rest of the analysis will still complete). Add one first if you want strut classification.
          </div>
        )}
      {(job.state === "intake_ready" || job.state === "failed") && (
        <button className="secondary-button" onClick={onAnalyze}>
          <Play size={16} /> {job.state === "failed" ? "Retry analysis" : "Run analysis"}
        </button>
      )}
    </div>
  );
}

function DefectSurface({
  job,
  classification,
  loadError,
  visible,
  onToggle,
  selectedStrutIds,
  onSelectStrut,
}: {
  job: Job;
  classification: DefectClassification | null;
  loadError: string;
  visible: Set<StrutVerdict>;
  onToggle: (verdict: StrutVerdict) => void;
  selectedStrutIds: Set<number>;
  onSelectStrut: (strutId: number) => void;
}) {
  if (job.defects?.status === "failed") {
    return <div className="error-banner">{job.defects.error || "The defect-detection pipeline failed."}</div>;
  }
  return (
    <div className="defects-view">
      {loadError && <div className="error-banner">{loadError}</div>}
      {classification ? (
        <>
          <p className="defects-summary">
            {(classification.meta.n - (classification.meta.counts.present || 0)).toLocaleString()} of{" "}
            {classification.meta.n.toLocaleString()} struts show a defect. Drag to rotate, scroll to zoom, click a
            strut to select it, or toggle a category below.
          </p>
          <DefectViewer
            struts={classification.struts}
            visibleVerdicts={visible}
            selectedStrutIds={selectedStrutIds}
            onSelectStrut={onSelectStrut}
          />
          <div className="defect-gallery">
            {VERDICT_ORDER.map((verdict) => {
              const count = classification.meta.counts[verdict] || 0;
              const percent = classification.meta.n ? ((100 * count) / classification.meta.n).toFixed(2) : "0.00";
              return (
                <button
                  key={verdict}
                  className={`defect-card ${visible.has(verdict) ? "active" : ""}`}
                  onClick={() => onToggle(verdict)}
                  aria-pressed={visible.has(verdict)}
                >
                  <span className="defect-swatch" style={{ background: VERDICT_COLORS[verdict] }} />
                  <span className="defect-card-body">
                    <strong>{VERDICT_LABELS[verdict]}</strong>
                    <span className="defect-count">
                      {count.toLocaleString()} <small>({percent}%)</small>
                    </span>
                    <p>{VERDICT_DESCRIPTIONS[verdict]}</p>
                  </span>
                </button>
              );
            })}
          </div>
        </>
      ) : (
        <p>Loading the classification produced by the strut error detection pipeline…</p>
      )}
    </div>
  );
}

function ImageArtifact({ artifact }: { artifact: Job["artifacts"][number] | undefined }) {
  if (!artifact) {
    return (
      <div className="empty-state">
        <h2>Image not found</h2>
        <p>The referenced artifact isn&rsquo;t in this job&rsquo;s list yet.</p>
      </div>
    );
  }
  return (
    <div className="artifact-solo">
      <img src={artifact.downloadUrl} alt={artifact.caption} />
      <div className="artifact-solo-caption">
        <strong>{artifact.caption}</strong>
        <a href={artifact.downloadUrl} download>
          <Download size={14} /> Download
        </a>
      </div>
    </div>
  );
}

function ModelSurface({ file, artifact }: { file?: InputFile; artifact?: Job["artifacts"][number] }) {
  const url = file?.contentUrl ?? artifact?.downloadUrl;
  if (!url) {
    return (
      <div className="empty-state">
        <h2>Model not found</h2>
        <p>The referenced file or artifact isn&rsquo;t in this job&rsquo;s list yet.</p>
      </div>
    );
  }
  return <StlViewer url={url} />;
}

function DataVizSurface({ file, artifact }: { file?: InputFile; artifact?: Job["artifacts"][number] }) {
  if (artifact) {
    if (artifact.mediaType === "image/png") return <ImageArtifact artifact={artifact} />;
    return (
      <div className="empty-state">
        <FileArchive size={42} />
        <h2>{artifact.name}</h2>
        <p>No in-browser viewer for this file type.</p>
        <a href={artifact.downloadUrl} download>
          <Download size={14} style={{ marginRight: 6 }} />
          Download {artifact.name}
        </a>
      </div>
    );
  }
  if (file?.kind === "tiff") {
    return (
      <>
        <TiffViewer file={file} />
        <TiffCompareSection file={file} />
      </>
    );
  }
  if (file?.kind === "json") {
    return <DesignGraphViewer url={file.contentUrl} />;
  }
  return (
    <div className="empty-state">
      <h2>Nothing to show</h2>
      <p>The referenced file or artifact isn&rsquo;t in this job&rsquo;s list yet.</p>
    </div>
  );
}

function ReportSurface({ job }: { job: Job }) {
  if (!job.report) {
    return (
      <div className="empty-state">
        <h2>No report yet</h2>
        <p>The report is generated once analysis completes.</p>
      </div>
    );
  }
  return (
    <div className="report-layout">
      <div className="report-document">
        <div className="document-bar">
          <FileText size={18} />
          <strong>{job.report.name}</strong>
          <span>Markdown</span>
        </div>
        <MarkdownPreview url={job.report.previewUrl} />
      </div>
      <aside className="report-summary">
        <dl>
          <div>
            <dt>Input files</dt>
            <dd>{job.files.length}</dd>
          </div>
          <div>
            <dt>Visual outputs</dt>
            <dd>{job.artifacts.length}</dd>
          </div>
          <div>
            <dt>Job ID</dt>
            <dd>{job.id.slice(0, 8)}</dd>
          </div>
        </dl>
        <a className="primary-button wide" href={job.report.downloadUrl} download>
          <Download size={18} /> Download report
        </a>
      </aside>
    </div>
  );
}

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [addingFiles, setAddingFiles] = useState(false);

  const [mounted, setMountedRaw] = useState<MountedSurface | null>(null);
  const [breadcrumb, setBreadcrumb] = useState<MountedSurface | null>(null);

  const [classification, setClassification] = useState<DefectClassification | null>(null);
  const [classificationError, setClassificationError] = useState("");
  const [visibleVerdicts, setVisibleVerdicts] = useState<Set<StrutVerdict>>(
    () => new Set(VERDICT_ORDER.filter((verdict) => verdict !== "present")),
  );
  const [selectedStrutIds, setSelectedStrutIds] = useState<Set<number>>(new Set());

  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [statusEvents, setStatusEvents] = useState<{ id: string; text: string; at: number }[]>([]);

  const mountSurface = useCallback((next: MountedSurface | null) => {
    setMountedRaw((current) => {
      if (current && next && current.component !== next.component) {
        // Replacing one surface with a different one -- remember what was there.
        setBreadcrumb(current);
      } else if (!next) {
        // Explicit close: nothing to "go back to" from the file list, and an
        // old breadcrumb here would misleadingly point at whatever was open
        // before this close, even after the user picks something unrelated.
        setBreadcrumb(null);
      }
      return next;
    });
    if (next?.component === "DefectView") {
      if (next.props.filter_verdicts) setVisibleVerdicts(new Set(next.props.filter_verdicts));
      if (next.props.select_strut_ids) setSelectedStrutIds(new Set(next.props.select_strut_ids));
    }
  }, []);

  useEffect(() => {
    localStorage.removeItem("lattice-job");
  }, []);

  useEffect(() => {
    if (!job || job.state !== "analyzing") return;
    const timer = window.setInterval(async () => {
      setJob(await api.getJob(job.id));
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job]);

  useEffect(() => {
    const tiltPending = job?.files.some(
      (file) => file.kind === "tiff" && (file.tiltStatus === "pending" || file.tiltStatus === "checking"),
    );
    if (!job || !tiltPending) return;
    const timer = window.setInterval(async () => {
      setJob(await api.getJob(job.id));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  // Real, honest "streaming" status: every polled state/stage change becomes
  // a narration line in the chat timeline. Not attributed to the model --
  // it's a direct reflection of observed backend state, visually distinct
  // (see .status-event), never a fabricated commentary.
  const prevProgressRef = useRef<{ state: string; stage: string | null | undefined } | null>(null);
  useEffect(() => {
    if (!job) return;
    const stage = job.defects?.stage;
    const prev = prevProgressRef.current;
    const lines: string[] = [];
    if (prev) {
      if (job.state !== prev.state) {
        if (job.state === "analyzing") lines.push("Analysis started…");
        else if (job.state === "complete") lines.push("Analysis complete.");
        else if (job.state === "failed") lines.push(`Analysis failed${job.error ? `: ${job.error}` : "."}`);
      }
      if (stage && stage !== prev.stage) lines.push(`${DEFECT_STAGE_LABELS[stage]}…`);
    }
    if (lines.length) {
      setStatusEvents((current) => [
        ...current,
        ...lines.map((text, index) => ({ id: `${Date.now()}-${index}`, text, at: Date.now() + index })),
      ]);
    }
    prevProgressRef.current = { state: job.state, stage };
  }, [job]);

  useEffect(() => {
    if (!job?.defects || job.defects.status !== "complete" || !job.defects.dataUrl) {
      setClassification(null);
      return;
    }
    setClassificationError("");
    setClassification(null);
    fetch(job.defects.dataUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`Request failed (${response.status})`);
        return response.json();
      })
      .then(setClassification)
      .catch(() => setClassificationError("The defect classification could not be loaded."));
  }, [job?.defects?.dataUrl]);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setGlobalError("");
    try {
      const created = await api.createJob();
      const uploaded = await api.upload(created.id, files);
      setJob(uploaded);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const analyze = async () => {
    if (!job) return;
    setGlobalError("");
    try {
      setJob(await api.analyze(job.id));
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Analysis could not start");
    }
  };

  const addFiles = async (files: File[]) => {
    if (!job || !files.length) return;
    setGlobalError("");
    setAddingFiles(true);
    try {
      setJob(await api.upload(job.id, files));
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setAddingFiles(false);
    }
  };

  const reset = () => {
    localStorage.removeItem("lattice-job");
    setJob(null);
    setChatTurns([]);
    setStatusEvents([]);
    setClassification(null);
    setSelectedStrutIds(new Set());
    mountSurface(null);
    setBreadcrumb(null);
  };

  const previewFile = (file: InputFile) => {
    if (file.kind === "stl") mountSurface({ component: "ModelViewer", props: { file_id: file.id } });
    else mountSurface({ component: "DataViz", props: { file_id: file.id } });
  };

  const sendChat = async (message: string) => {
    if (!job) return;
    setChatBusy(true);
    setGlobalError("");
    try {
      const turn = await api.sendChat(job.id, message);
      setChatTurns((current) => [...current, turn]);
      const refreshed = await api.getJob(job.id);
      setJob(refreshed);
      if ("mount" in turn) mountSurface(turn.mount ?? null);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "The chat request failed");
    } finally {
      setChatBusy(false);
    }
  };

  const toggleVerdict = (verdict: StrutVerdict) => {
    setVisibleVerdicts((current) => {
      const next = new Set(current);
      if (next.has(verdict)) next.delete(verdict);
      else next.add(verdict);
      return next;
    });
  };

  const timeline = useMemo<TimelineEntry[]>(() => {
    const entries: TimelineEntry[] = [
      ...chatTurns.map((turn) => ({ kind: "turn" as const, at: new Date(turn.timestamp).getTime(), turn })),
      ...statusEvents.map((event) => ({ kind: "status" as const, at: event.at, id: event.id, text: event.text })),
    ];
    return entries.sort((a, b) => a.at - b.at);
  }, [chatTurns, statusEvents]);

  const mountedFile = mounted?.props.file_id ? job?.files.find((f) => f.id === mounted.props.file_id) : undefined;
  const mountedArtifact = mounted?.props.artifact_id
    ? job?.artifacts.find((a) => a.id === mounted.props.artifact_id)
    : undefined;

  const chatDisabled = !job;
  const chatDisabledReason = "Upload a scan to get started";

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={reset} aria-label="Lattice Lens home">
          <span className="brand-mark">
            <span />
            <span />
            <span />
            <span />
          </span>
          <span>
            <strong>Lattice</strong> Lens
          </span>
        </button>
        <div className="status-chip">
          <span /> {job ? `Job ${job.id.slice(0, 8)} · ${job.state}` : "No inspection loaded"}
        </div>
      </header>

      {globalError && (
        <div className="global-error">
          <span>{globalError}</span>
          <button onClick={() => setGlobalError("")}>
            <X size={16} />
          </button>
        </div>
      )}

      {!job ? (
        <main>
          <UploadScreen onComplete={upload} busy={busy} />
        </main>
      ) : (
        <main className="workbench">
          <section className="canvas-pane">
            {mounted && (
              <div className="surface-bar">
                {breadcrumb && breadcrumb.component !== mounted.component && (
                  <button className="text-button" onClick={() => mountSurface(breadcrumb)}>
                    <ChevronLeft size={14} /> Back to {SURFACE_LABELS[breadcrumb.component]}
                  </button>
                )}
                <span className="surface-label">{SURFACE_LABELS[mounted.component]}</span>
                <button className="icon-button" onClick={() => mountSurface(null)} aria-label="Close">
                  <X size={16} />
                </button>
              </div>
            )}
            {job.error && <div className="error-banner">{job.error}</div>}
            <div className="canvas-body">
              {!mounted && (
                <FileListSurface job={job} onPreview={previewFile} onAnalyze={analyze} onAddFiles={addFiles} addingFiles={addingFiles} />
              )}
              {mounted?.component === "ModelViewer" && <ModelSurface file={mountedFile} artifact={mountedArtifact} />}
              {mounted?.component === "DataViz" && <DataVizSurface file={mountedFile} artifact={mountedArtifact} />}
              {mounted?.component === "DefectView" && (
                <DefectSurface
                  job={job}
                  classification={classification}
                  loadError={classificationError}
                  visible={visibleVerdicts}
                  onToggle={toggleVerdict}
                  selectedStrutIds={selectedStrutIds}
                  onSelectStrut={(id) => setSelectedStrutIds(new Set([id]))}
                />
              )}
              {mounted?.component === "ReportView" && <ReportSurface job={job} />}
            </div>
          </section>
          <ChatPanel timeline={timeline} onSend={sendChat} busy={chatBusy} disabled={chatDisabled} disabledReason={chatDisabledReason} />
        </main>
      )}
    </div>
  );
}
