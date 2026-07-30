import type { ChatTurn, Job } from "./types";

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

async function request<T>(url: string, options?: RequestInit, normalize = false): Promise<T> {
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
  return (normalize ? normalizeJob(payload as Job) : payload) as T;
}

export const api = {
  createJob: () => request<Job>("/api/jobs", { method: "POST" }, true),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`, undefined, true),
  upload: async (id: string, files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request<Job>(`/api/jobs/${id}/files`, { method: "POST", body }, true);
  },
  analyze: (id: string) =>
    request<Job>(`/api/jobs/${id}/analysis`, { method: "POST" }, true),
  sendChat: (id: string, message: string) =>
    request<ChatTurn>(`/api/jobs/${id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    }),
  getChatHistory: (id: string) => request<{ turns: ChatTurn[] }>(`/api/jobs/${id}/chat`),
};
