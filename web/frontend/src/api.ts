import type { ApiHealth, Job } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function absoluteUrl(path: string | undefined) {
  if (!path || /^https?:\/\//.test(path)) return path;
  return apiUrl(path);
}

function normalizeJob(job: Job): Job {
  return {
    ...job,
    files: job.files.map((file) => ({
      ...file,
      contentUrl: absoluteUrl(file.contentUrl)!,
      sliceUrl: absoluteUrl(file.sliceUrl),
      correctedSliceUrl: file.correctedSliceUrl ? absoluteUrl(file.correctedSliceUrl) : file.correctedSliceUrl,
    })),
    artifacts: job.artifacts.map((artifact) => ({
      ...artifact,
      downloadUrl: absoluteUrl(artifact.downloadUrl)!,
    })),
    report: job.report
      ? {
          ...job.report,
          downloadUrl: absoluteUrl(job.report.downloadUrl)!,
          previewUrl: absoluteUrl(job.report.previewUrl)!,
        }
      : null,
    defects: job.defects
      ? { ...job.defects, dataUrl: job.defects.dataUrl ? absoluteUrl(job.defects.dataUrl) : job.defects.dataUrl }
      : job.defects,
  };
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  if (!API_BASE && import.meta.env.PROD) {
    throw new Error(
      "This deployment has no backend API configured. Set the VITE_API_BASE_URL GitHub Actions variable.",
    );
  }
  const response = await fetch(apiUrl(url), options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the status-based message for non-JSON errors.
    }
    throw new Error(message);
  }
  const payload = await response.json();
  return (url.startsWith("/api/jobs") ? normalizeJob(payload as Job) : payload) as T;
}

export const api = {
  health: async (): Promise<ApiHealth> => {
    if (!API_BASE && import.meta.env.PROD) {
      throw new Error("No backend API is configured.");
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8_000);
    try {
      const response = await fetch(apiUrl("/api/health"), {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Health check failed (${response.status})`);
      return await response.json();
    } finally {
      window.clearTimeout(timeout);
    }
  },
  createJob: () => request<Job>("/api/jobs", { method: "POST" }),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  upload: async (id: string, files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request<Job>(`/api/jobs/${id}/files`, { method: "POST", body });
  },
  analyze: (id: string) =>
    request<Job>(`/api/jobs/${id}/analysis`, { method: "POST" }),
};
