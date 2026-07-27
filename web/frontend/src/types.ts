export type WorkflowState =
  | "new"
  | "intake_ready"
  | "analyzing"
  | "complete"
  | "failed";

export interface InputFile {
  id: string;
  name: string;
  size: number;
  kind: "json" | "tiff" | "stl";
  summary: string;
  pageCount?: number;
  width?: number;
  height?: number;
  triangleCount?: number | null;
  contentUrl: string;
  sliceUrl?: string;
}

export interface Artifact {
  id: string;
  name: string;
  caption: string;
  mediaType: string;
  downloadUrl: string;
}

export interface Report {
  name: string;
  mediaType: string;
  downloadUrl: string;
  previewUrl: string;
}

export interface Job {
  id: string;
  state: WorkflowState;
  files: InputFile[];
  artifacts: Artifact[];
  report: Report | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}
