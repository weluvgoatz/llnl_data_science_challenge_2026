import {
  Box,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  FileArchive,
  FileJson,
  FileText,
  Images,
  LoaderCircle,
  Microscope,
  Play,
  UploadCloud,
  X,
} from "lucide-react";
import { DragEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { StlViewer } from "./components/StlViewer";
import type { Artifact, InputFile, Job } from "./types";

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

function TiffViewer({ file }: { file: InputFile }) {
  const [index, setIndex] = useState(Math.floor((file.pageCount || 1) / 2));
  const max = Math.max((file.pageCount || 1) - 1, 0);
  return (
    <div className="tiff-viewer">
      <div className="slice-frame">
        <img src={`${file.sliceUrl}?index=${index}`} alt={`Slice ${index} of ${file.name}`} />
      </div>
      <div className="slice-control">
        <label>Slice <strong>{index + 1}</strong> of {max + 1}</label>
        <input type="range" min="0" max={max} value={index} onChange={(e) => setIndex(Number(e.target.value))} />
      </div>
    </div>
  );
}

function InspectPage({
  job,
  onAnalyze,
}: {
  job: Job;
  onAnalyze: () => Promise<void>;
}) {
  const visualFiles = job.files.filter((file) => file.kind !== "json");
  const [selectedId, setSelectedId] = useState(visualFiles[0]?.id || job.files[0]?.id);
  const selected = job.files.find((file) => file.id === selectedId) || job.files[0];
  const working = job.state === "analyzing";
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
              <span>Preparing visual findings and the NDE report…</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function AnalysisPage({ artifacts }: { artifacts: Artifact[] }) {
  const [focused, setFocused] = useState<Artifact | null>(null);
  return (
    <section className="page">
      <div className="page-heading split">
        <div>
          <span className="eyebrow success"><Check size={13} /> Analysis complete</span>
          <h1>Visual findings.</h1>
          <p>Review generated perspectives and download any image for closer inspection.</p>
        </div>
        <div className="finding-count"><Images size={20} /><strong>{artifacts.length}</strong> images</div>
      </div>
      {artifacts.length ? (
        <div className="artifact-grid">
          {artifacts.map((item) => (
            <article className="artifact-card" key={item.id}>
              <button className="artifact-image" onClick={() => setFocused(item)}>
                <img src={item.downloadUrl} alt={item.caption} />
              </button>
              <div>
                <span><strong>{item.caption}</strong><small>PNG analysis output</small></span>
                <a className="icon-button" href={item.downloadUrl} download><Download size={17} /> Download</a>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-state"><Images size={40} /><h2>No PNG outputs</h2><p>The run completed without visual artifacts.</p></div>
      )}
      {focused && (
        <div className="lightbox" role="dialog" onClick={() => setFocused(null)}>
          <button aria-label="Close image"><X /></button>
          <img src={focused.downloadUrl} alt={focused.caption} />
          <strong>{focused.caption}</strong>
        </div>
      )}
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

  useEffect(() => {
    const id = pathParts[0] === "jobs" ? pathParts[1] : localStorage.getItem("lattice-job");
    if (!id) return;
    api.getJob(id)
      .then((loaded) => {
        setJob(loaded);
        localStorage.setItem("lattice-job", loaded.id);
        if (pathParts[0] !== "jobs") navigate(`/jobs/${loaded.id}/${steps[allowedStep(loaded)].slug}`, { replace: true });
      })
      .catch(() => {
        localStorage.removeItem("lattice-job");
        if (pathParts[0] === "jobs") navigate("/", { replace: true });
      });
    // Route identity is intentionally the refresh trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathParts[1]]);

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
      localStorage.setItem("lattice-job", uploaded.id);
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
        {currentStep.slug === "inspect" && job && <InspectPage job={job} onAnalyze={analyze} />}
        {currentStep.slug === "analysis" && job && <AnalysisPage artifacts={job.artifacts} />}
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
