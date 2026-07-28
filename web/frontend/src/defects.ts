import type { StrutVerdict } from "./types";

// Defects first (most important to see), healthy struts last.
export const VERDICT_ORDER: StrutVerdict[] = ["missing", "disconnected", "bent", "thin", "present"];

// Matches analysis/defect_detection/sample_output/README.md's colour key.
export const VERDICT_COLORS: Record<StrutVerdict, string> = {
  present: "#78B4FF",
  missing: "#FF4141",
  bent: "#F546F0",
  thin: "#FFE12D",
  disconnected: "#FF8C1E",
};

export const VERDICT_LABELS: Record<StrutVerdict, string> = {
  present: "Present",
  missing: "Missing",
  bent: "Bent",
  thin: "Thin",
  disconnected: "Disconnected",
};

// Matches the decision rules in INTEGRATION.md / describe_thresholds().
export const VERDICT_DESCRIPTIONS: Record<StrutVerdict, string> = {
  present: "Healthy strut — none of the defect conditions below apply.",
  missing: "Metal along the strut is under 15% of its length — essentially absent.",
  disconnected: "One continuous gap spans at least 25% of the strut's length.",
  thin: "Strut density is a robust statistical low outlier versus the rest of the specimen.",
  bent: "Peak bow off the strut's own axis exceeds one strut radius.",
};
