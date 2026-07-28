import {
  Box,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  FileArchive,
  FileJson,
  FileText,
  LoaderCircle,
  Maximize2,
  Microscope,
  Play,
  RotateCcw,
  TriangleAlert,
  UploadCloud,
  X,
} from "lucide-react";
import { DragEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { StlViewer } from "./components/StlViewer";
import { DefectViewer } from "./components/DefectViewer";
import { ZoomableImage } from "./components/ZoomableImage";
import { VERDICT_COLORS, VERDICT_DESCRIPTIONS, VERDICT_LABELS, VERDICT_ORDER } from "./defects";
import type { DefectClassification, DefectStage, InputFile, Job, StrutVerdict } from "./types";

const DEFECT_STAGE_LABELS: Record<DefectStage, string> = {
  segmenting: "segmenting the scan",
  cleaning: "cleaning the segmentation",
  skeletonizing: "skeletonizing the lattice",
  building_graph: "building the as-built graph",
  classifying: "classifying every strut",
  bend_detail: "measuring strut bow",
  complete: "finishing up",
};

const steps = [
  { slug: "upload", label: "Upload", note: "Input data" },
  { slug: "inspect", label: "Inspect", note: "3D & slices" },
  { slug: "analysis", label: "Analysis", note: "Visual findings" },
  { slug: "report", label: "Report", note: "NDE summary" },
] as const;

type StepSlug = (typeof steps)[number]["slug"];

function allowedStep(job: Job | null) {
  if (!job || job.state === "new") return 0;
  if (job.state === "intake_ready" || job.state === "analyzing" || job.state === "failed") return 1;
  if (job.state === "complete") return job.report ? 3 : 2;
  return 0;
}

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

function UploadPage({
  onComplete,
  busy,
}: {
  onComplete: (files: File[]) => Promise<void>;
  busy: boolean;
}) {
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
        <p>Upload one file or a matched set. We’ll validate the data before opening the workspace.</p>
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
        <div className="upload-mark"><UploadCloud size={27} /></div>
        <h2>Drop lattice files here</h2>
        <p>or <span>browse your computer</span></p>
        <small>JSON · TIFF stacks · STL meshes &nbsp;·&nbsp; 2 GB maximum per file</small>
      </label>
      {error && <div className="error-banner">{error}</div>}
      {files.length > 0 && (
        <div className="queue">
          <div className="queue-header">
            <h3>Ready to upload <span>{files.length}</span></h3>
            <button className="text-button" onClick={() => setFiles([])}>Clear all</button>
          </div>
          {files.map((file, index) => {
            const extension = file.name.split(".").pop()?.toLowerCase();
            const kind = extension === "stl" ? "stl" : extension === "json" ? "json" : "tiff";
            return (
              <div className="queued-file" key={`${file.name}:${file.size}`}>
                <div className={`file-glyph ${kind}`}>{fileIcon(kind)}</div>
                <div className="file-copy">
                  <strong>{file.name}</strong>
                  <span>{formatBytes(file.size)} · {kind.toUpperCase()}</span>
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
            {busy ? "Validating files…" : "Upload and open workspace"}
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

function TiffViewer({ file }: { file: InputFile }) {
  const [index, setIndex] = useState(Math.floor((file.pageCount || 1) / 2));
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
            <button
              className="slice-frame"
              onClick={() => setFocused("original")}
              aria-label={`Enlarge original ${caption}`}
            >
              <img src={originalUrl} alt={`Original ${caption}`} />
            </button>
          </div>
          <div className="slice-pane">
            <span className="slice-pane-label">Tilt-fixed</span>
            <button
              className="slice-frame"
              onClick={() => setFocused("corrected")}
              aria-label={`Enlarge tilt-fixed ${caption}`}
            >
              <img src={correctedUrl} alt={`Tilt-fixed ${caption}`} />
            </button>
          </div>
        </div>
      ) : (
        <button className="slice-frame" onClick={() => setFocused("original")} aria-label={`Enlarge ${caption}`}>
          <img src={originalUrl} alt={caption} />
          <span className="viewer-hint"><Maximize2 size={13} /> Click to see the full frame</span>
        </button>
      )}
      <div className="slice-control">
        <label>Slice <strong>{index + 1}</strong> of {max + 1}</label>
        <input type="range" min="0" max={max} value={index} onChange={(e) => setIndex(Number(e.target.value))} />
      </div>
      {focused && focusedUrl && (
        <div className="lightbox" role="dialog" onClick={() => setFocused(null)}>
          <button aria-label="Close image"><X /></button>
          <img src={focusedUrl} alt={caption} />
          <strong>{focused === "corrected" ? "Tilt-fixed — " : comparing ? "Original — " : ""}{caption}</strong>
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

  return (
    <section className="tiff-compare">
      <div className="page-heading">
        <span className="eyebrow">Tilt inspection</span>
        <h2>Original vs. tilt-corrected slices.</h2>
        <p>Scroll or use the buttons to zoom, drag to pan once zoomed in, double-click to reset.</p>
      </div>
      <TiltBanner file={file} />
      <div className="tiff-compare-grid">
        <div className="tiff-compare-pane">
          <span className="slice-pane-label">Original</span>
          <ZoomableImage src={originalUrl} alt={`Original slice ${index + 1} of ${file.name}`} />
        </div>
        <div className="tiff-compare-pane">
          <span className="slice-pane-label">Tilt-fixed</span>
          {correctedUrl ? (
            <ZoomableImage src={correctedUrl} alt={`Tilt-fixed slice ${index + 1} of ${file.name}`} />
          ) : (
            <div className="tiff-compare-empty">
              {file.tiltStatus === "not_tilted"
                ? "No meaningful tilt detected — the corrected slice would match the original."
                : file.tiltStatus === "failed"
                ? "Tilt check failed, so no corrected slice is available."
                : "Checking for tilt…"}
            </div>
          )}
        </div>
      </div>
      <div className="slice-control">
        <label>Slice <strong>{index + 1}</strong> of {max + 1}</label>
        <input type="range" min="0" max={max} value={index} onChange={(event) => setIndex(Number(event.target.value))} />
      </div>
    </section>
  );
}

function InspectPage({
  job,
  onAnalyze,
  onAddFiles,
  addingFiles,
}: {
  job: Job;
  onAnalyze: () => Promise<void>;
  onAddFiles: (files: File[]) => Promise<void>;
  addingFiles: boolean;
}) {
  const visualFiles = job.files.filter((file) => file.kind !== "json");
  const [selectedId, setSelectedId] = useState(visualFiles[0]?.id || job.files[0]?.id);
  const selected = job.files.find((file) => file.id === selectedId) || job.files[0];
  const working = job.state === "analyzing";
  const canAddFiles = job.state === "intake_ready" && !working;
  const tiffFile = job.files.find((file) => file.kind === "tiff");
  return (
    <section className="page">
      <div className="page-heading split">
        <div>
          <span className="eyebrow">Input workspace</span>
          <h1>Inspect the source data.</h1>
          <p>Confirm orientation and coverage before starting the agentic analysis.</p>
        </div>
        <button className="primary-button" disabled={working} onClick={onAnalyze}>
          {working ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
          {working ? "Analysis running…" : job.state === "failed" ? "Retry analysis" : "Run analysis"}
        </button>
      </div>
      {job.error && <div className="error-banner">{job.error}</div>}
      <div className="workspace-grid">
        <aside className="file-rail">
          <h3>Inputs <span>{job.files.length}</span></h3>
          {job.files.map((file) => (
            <button
              className={`rail-file ${selected?.id === file.id ? "active" : ""}`}
              key={file.id}
              onClick={() => setSelectedId(file.id)}
            >
              <span className={`file-glyph ${file.kind}`}>{fileIcon(file.kind)}</span>
              <span><strong>{file.name}</strong><small>{file.summary}</small></span>
            </button>
          ))}
          {canAddFiles && (
            <label className={`rail-add ${addingFiles ? "disabled" : ""}`}>
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
              {addingFiles ? <LoaderCircle className="spin" size={16} /> : <UploadCloud size={16} />}
              <span>{addingFiles ? "Uploading…" : "Add TIFF or JSON"}</span>
            </label>
          )}
        </aside>
        <div className="viewer-panel">
          <div className="viewer-title">
            <div><strong>{selected.name}</strong><span>{selected.summary}</span></div>
            <span className="format-pill">{selected.kind.toUpperCase()}</span>
          </div>
          {selected.kind === "stl" && <StlViewer url={selected.contentUrl} />}
          {selected.kind === "tiff" && <TiffViewer file={selected} />}
          {selected.kind === "json" && (
            <div className="json-placeholder">
              <FileJson size={42} />
              <h2>JSON validated</h2>
              <p>Geometry rendering is intentionally deferred. This file will be available to the analysis harness.</p>
              <a href={selected.contentUrl} target="_blank" rel="noreferrer">Open source JSON</a>
            </div>
          )}
          {working && (
            <div className="analysis-overlay">
              <LoaderCircle className="spin" size={30} />
              <strong>Codex workflow in progress</strong>
              <span>
                {job.defects?.status === "running" && job.defects.stage
                  ? `Detecting strut defects — ${DEFECT_STAGE_LABELS[job.defects.stage]}…`
                  : "Preparing visual findings and the NDE report…"}
              </span>
            </div>
          )}
        </div>
      </div>
      {tiffFile && <TiffCompareSection file={tiffFile} />}
    </section>
  );
}

function DefectSection({ job }: { job: Job }) {
  const [data, setData] = useState<DefectClassification | null>(null);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState<Set<StrutVerdict>>(
    () => new Set(VERDICT_ORDER.filter((verdict) => verdict !== "present")),
  );
  const dataUrl = job.defects?.status === "complete" ? job.defects.dataUrl : null;

  useEffect(() => {
    if (!dataUrl) return;
    setError("");
    setData(null);
    fetch(dataUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`Request failed (${response.status})`);
        return response.json();
      })
      .then(setData)
      .catch(() => setError("The defect classification could not be loaded."));
  }, [dataUrl]);

  const toggle = (verdict: StrutVerdict) => {
    setVisible((current) => {
      const next = new Set(current);
      if (next.has(verdict)) next.delete(verdict);
      else next.add(verdict);
      return next;
    });
  };

  if (!job.defects) return null;

  return (
    <section className="defect-section">
      <div className="page-heading">
        <span className="eyebrow">Strut error detection</span>
        <h1>Every designed strut, checked against the scan.</h1>
        {data ? (
          <p>
            {(data.meta.n - (data.meta.counts.present || 0)).toLocaleString()} of{" "}
            {data.meta.n.toLocaleString()} struts show a defect. Drag to rotate, scroll to zoom — click a
            category below to show or hide it.
          </p>
        ) : (
          <p>Loading the classification produced by the strut error detection pipeline…</p>
        )}
      </div>
      {job.defects.status === "failed" && (
        <div className="error-banner">{job.defects.error || "The defect-detection pipeline failed."}</div>
      )}
      {error && <div className="error-banner">{error}</div>}
      {data && (
        <>
          <DefectViewer struts={data.struts} visibleVerdicts={visible} />
          <div className="defect-gallery">
            {VERDICT_ORDER.map((verdict) => {
              const count = data.meta.counts[verdict] || 0;
              const percent = data.meta.n ? ((100 * count) / data.meta.n).toFixed(2) : "0.00";
              return (
                <button
                  key={verdict}
                  className={`defect-card ${visible.has(verdict) ? "active" : ""}`}
                  onClick={() => toggle(verdict)}
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
      )}
    </section>
  );
}

function AnalysisPage({ job }: { job: Job }) {
  return (
    <section className="page">
      <DefectSection job={job} />
    </section>
  );
}

function ReportPage({ job, onReset }: { job: Job; onReset: () => void }) {
  return (
    <section className="page">
      <div className="page-heading split">
        <div>
          <span className="eyebrow success"><Check size={13} /> Workflow complete</span>
          <h1>Your report is ready.</h1>
          <p>A traceable summary of the uploaded data and generated analysis.</p>
        </div>
        {job.report && (
          <a className="primary-button" href={job.report.downloadUrl} download>
            <Download size={18} /> Download report
          </a>
        )}
      </div>
      <div className="report-layout">
        <div className="report-document">
          <div className="document-bar"><FileText size={18} /><strong>{job.report?.name}</strong><span>Markdown</span></div>
          {job.report && <MarkdownPreview url={job.report.previewUrl} />}
        </div>
        <aside className="report-summary">
          <Microscope size={27} />
          <h2>Inspection complete</h2>
          <dl>
            <div><dt>Input files</dt><dd>{job.files.length}</dd></div>
            <div><dt>Visual outputs</dt><dd>{job.artifacts.length}</dd></div>
            <div><dt>Job ID</dt><dd>{job.id.slice(0, 8)}</dd></div>
          </dl>
          <button className="secondary-button wide" onClick={onReset}>Start a new inspection</button>
        </aside>
      </div>
    </section>
  );
}

export default function App() {
  const readHashPath = () => window.location.hash.slice(1) || "/";
  const [pathname, setPathname] = useState(readHashPath);
  const navigate = useCallback((target: string, options?: { replace?: boolean }) => {
    const url = `${window.location.pathname}${window.location.search}#${target}`;
    if (options?.replace) window.history.replaceState(null, "", url);
    else window.history.pushState(null, "", url);
    setPathname(target);
  }, []);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const pathParts = pathname.split("/").filter(Boolean);
  const pathStep = pathParts[2] as StepSlug | undefined;
  const currentIndex = Math.max(0, steps.findIndex((step) => step.slug === pathStep));
  const maxStep = allowedStep(job);

  useEffect(() => {
    const onPopState = () => setPathname(readHashPath());
    window.addEventListener("popstate", onPopState);
    window.addEventListener("hashchange", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("hashchange", onPopState);
    };
  }, []);

  // DEV MODE: "resume last job on reload" is temporarily disabled so every page
  // load starts fresh. This block also purges any job id left over in
  // localStorage from before this was disabled. To restore persistence, delete
  // this effect and uncomment the block below it.
  useEffect(() => {
    localStorage.removeItem("lattice-job");
  }, []);

  // useEffect(() => {
  //   const id = pathParts[0] === "jobs" ? pathParts[1] : localStorage.getItem("lattice-job");
  //   if (!id) return;
  //   api.getJob(id)
  //     .then((loaded) => {
  //       setJob(loaded);
  //       localStorage.setItem("lattice-job", loaded.id);
  //       if (pathParts[0] !== "jobs") navigate(`/jobs/${loaded.id}/${steps[allowedStep(loaded)].slug}`, { replace: true });
  //     })
  //     .catch(() => {
  //       localStorage.removeItem("lattice-job");
  //       if (pathParts[0] === "jobs") navigate("/", { replace: true });
  //     });
  //   // Route identity is intentionally the refresh trigger.
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, [pathParts[1]]);

  useEffect(() => {
    if (!job || job.state !== "analyzing") return;
    const timer = window.setInterval(async () => {
      const updated = await api.getJob(job.id);
      setJob(updated);
      if (updated.state === "complete") navigate(`/jobs/${updated.id}/analysis`);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job, navigate]);

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

  useEffect(() => {
    if (job && currentIndex > allowedStep(job)) {
      navigate(`/jobs/${job.id}/${steps[allowedStep(job)].slug}`, { replace: true });
    }
  }, [currentIndex, job, navigate]);

  const currentStep = useMemo(() => steps[currentIndex] || steps[0], [currentIndex]);
  const upload = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setGlobalError("");
    try {
      const created = await api.createJob();
      const uploaded = await api.upload(created.id, files);
      setJob(uploaded);
      // localStorage.setItem("lattice-job", uploaded.id); // DEV MODE: see note above
      navigate(`/jobs/${uploaded.id}/inspect`);
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
  const [addingFiles, setAddingFiles] = useState(false);
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
    navigate("/");
  };
  const move = (index: number) => {
    if (!job || index > maxStep) return;
    navigate(`/jobs/${job.id}/${steps[index].slug}`);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={reset} aria-label="Lattice Lens home">
          <span className="brand-mark"><span /><span /><span /><span /></span>
          <span><strong>Lattice</strong> Lens</span>
        </button>
        <div className="status-chip"><span /> Agent workflow ready</div>
      </header>
      <nav className="stepper" aria-label="Workflow progress">
        {steps.map((step, index) => {
          const available = index <= maxStep;
          const complete = index < maxStep;
          return (
            <button
              key={step.slug}
              disabled={!available || !job}
              className={`${index === currentIndex ? "current" : ""} ${complete ? "complete" : ""}`}
              onClick={() => move(index)}
              title={!available ? `Available after ${steps[index - 1].label.toLowerCase()}` : ""}
            >
              <span className="step-number">{complete ? <Check size={15} /> : index + 1}</span>
              <span><strong>{step.label}</strong><small>{step.note}</small></span>
              {index < steps.length - 1 && <i />}
            </button>
          );
        })}
      </nav>
      <main>
        {globalError && <div className="global-error"><span>{globalError}</span><button onClick={() => setGlobalError("")}><X size={16} /></button></div>}
        {currentStep.slug === "upload" && <UploadPage onComplete={upload} busy={busy} />}
        {currentStep.slug === "inspect" && job && (
          <InspectPage job={job} onAnalyze={analyze} onAddFiles={addFiles} addingFiles={addingFiles} />
        )}
        {currentStep.slug === "analysis" && job && <AnalysisPage job={job} />}
        {currentStep.slug === "report" && job && <ReportPage job={job} onReset={reset} />}
      </main>
      {job && currentStep.slug !== "upload" && (
        <footer className="page-controls">
          <button className="secondary-button" disabled={currentIndex === 0} onClick={() => move(currentIndex - 1)}>
            <ChevronLeft size={18} /> Back
          </button>
          <span>Step {currentIndex + 1} of 4</span>
          <button className="secondary-button" disabled={currentIndex >= maxStep} onClick={() => move(currentIndex + 1)}>
            Next <ChevronRight size={18} />
          </button>
        </footer>
      )}
    </div>
  );
}
