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
  | "detecting"
  | "classifying"
  | "complete";

export interface DefectVersion {
  id: number;
  label: string;
  path: string;
  params: Record<string, number>;
  counts: Partial<Record<StrutVerdict, number>>;
  n: number;
  createdAt: string;
}

export interface DefectsInfo {
  status: "running" | "complete" | "failed";
  stage?: DefectStage | null;
  error?: string | null;
  dataUrl?: string | null;
  versions?: DefectVersion[];
  activeVersionId?: number;
}

export type StrutVerdict = "present" | "missing" | "bent" | "thin" | "disconnected";

export interface StrutEvidence {
  reason: "as_built_edge_matched" | "material_sample_between_anchors" | "no_metal_at_anchor";
  junction_a_anchor?: string;
  junction_b_anchor?: string;
  junction_a_snap_dist_vox?: number | null;
  junction_b_snap_dist_vox?: number | null;
  bow_um?: number;
  bow_threshold_um?: number;
  mean_density?: number;
  density_cutoff?: number;
  density_median?: number;
  density_mad_scaled?: number;
  outlier_k?: number;
  measured_radius_um?: number | null;
  nominal_radius_um?: number;
  metal_fraction?: number;
  missing_fraction_threshold?: number;
  longest_gap_fraction?: number;
  gap_fraction_threshold?: number;
  as_built_node_a?: number;
  as_built_node_b?: number;
}

export interface Strut {
  id: number;
  p0: [number, number, number];
  p1: [number, number, number];
  verdict: StrutVerdict;
  design_thickness?: number | null;
  evidence?: StrutEvidence;
}

export interface DefectClassification {
  struts: Strut[];
  meta: {
    counts: Partial<Record<StrutVerdict, number>>;
    n: number;
    volume_shape_zyx: [number, number, number];
  };
}

export interface ToolCall {
  tool: string;
  input: Record<string, unknown>;
  output: unknown;
  is_error: boolean;
}

export interface SubagentTrace {
  request: string;
  final_text: string;
  tool_calls: ToolCall[];
  stop_reason: string;
}

export type SurfaceComponent = "ModelViewer" | "DefectView" | "ReportView" | "DataViz";

export interface SurfaceProps {
  file_id?: string;
  artifact_id?: string;
  version_id?: number;
  filter_verdicts?: StrutVerdict[];
  select_strut_ids?: number[];
  slice_index?: number;
  show_tilt_pane?: boolean;
}

export interface MountedSurface {
  component: SurfaceComponent;
  props: SurfaceProps;
}

export interface ChatTurn {
  timestamp: string;
  user_message: string;
  reply: string;
  orchestrator_tool_calls: ToolCall[];
  subagent_traces: Record<string, SubagentTrace[]>;
  // Key ABSENT (undefined) = the agent didn't touch the surface this turn,
  // leave whatever's mounted alone. Key present with value null = an
  // explicit unmount_surface call, clear back to the file list. Key present
  // with a MountedSurface = mount that. Collapsing "absent" and "null" to
  // the same thing would make a plain Q&A turn indistinguishable from "hide
  // what you're looking at" -- keep them distinct end to end.
  mount?: MountedSurface | null;
}

// A chat-pane timeline entry. "turn" is a real backend chat turn. "status"
// is a frontend-only narration line (e.g. "skeletonizing the lattice…")
// derived from real polled job state -- deliberately understated (a small
// pill), never attributed to the model. "announcement" is also frontend-
// synthesized from real polled state (not a new LLM call), but for exactly
// one event -- analysis reaching "complete" -- it's rendered with the same
// visual weight as a real assistant reply, since the whole point is for the
// user to reliably notice it; the text itself is still a fixed template
// over real state, never an invented number or claim.
export type TimelineEntry =
  | { kind: "turn"; at: number; turn: ChatTurn }
  | { kind: "status"; at: number; id: string; text: string }
  | { kind: "announcement"; at: number; id: string; text: string };

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
