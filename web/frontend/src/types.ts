export type WorkflowState =
  | "new"
  | "intake_ready"
  | "analyzing"
  | "complete"
  | "failed";

export type TiltStatus = "pending" | "checking" | "not_tilted" | "corrected" | "failed";

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
  tiltStatus?: TiltStatus;
  tiltZY?: number | null;
  tiltZX?: number | null;
  tiltError?: string | null;
  correctedSliceUrl?: string | null;
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

export type DefectStage =
  | "segmenting"
  | "cleaning"
  | "skeletonizing"
  | "building_graph"
  | "classifying"
  | "bend_detail"
  | "complete";

export interface DefectsInfo {
  status: "running" | "complete" | "failed";
  stage?: DefectStage | null;
  error?: string | null;
  dataUrl?: string | null;
}

export type StrutVerdict = "present" | "missing" | "bent" | "thin" | "disconnected";

export interface Strut {
  p0: [number, number, number];
  p1: [number, number, number];
  verdict: StrutVerdict;
}

export interface DefectClassification {
  struts: Strut[];
  meta: {
    counts: Partial<Record<StrutVerdict, number>>;
    n: number;
    volume_shape_zyx: [number, number, number];
  };
}

export interface Job {
  id: string;
  state: WorkflowState;
  files: InputFile[];
  artifacts: Artifact[];
  report: Report | null;
  defects?: DefectsInfo | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}
