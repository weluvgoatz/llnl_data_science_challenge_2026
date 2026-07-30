"""Foundational report-building layer: locate strut records in a defect-
results JSON of unknown shape, summarize them, and render the summary as
Markdown or a PDF.

Reconstructed to match the exact interface
analysis/defect_detection/defect_results_reporting_agent.py imports and
calls (that file's own source is the contract -- this module supplies what
it depends on, nothing more, nothing renamed). Ground truth for the expected
output shape/wording is
analysis/defect_detection/octet_9x9x9_defect_report.pdf, a report the
original author generated from
analysis/defect_detection/sample_output/octet_9x9x9_defects.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

# Same candidate key lists the caller (defect_results_reporting_agent.py)
# uses for the *registered design* side -- kept in sync so both sides agree
# on what "a strut collection" looks like.
STRUT_COLLECTION_KEYS = ("struts", "edges")
VERDICT_KEYS = ("verdict", "classification", "defect_type", "predicted_class")

# Display label for each raw verdict string. "uncertain" is listed because
# the sample report enumerates it (as an absent/zero category) even though
# no producer in this codebase emits it -- see _uncertain_warning().
VERDICT_DISPLAY = {
    "present": "Intact",
    "missing": "Missing",
    "disconnected": "Broken (source: disconnected)",
    "thin": "Thinned (source: thin)",
    "bent": "Bent",
    "uncertain": "Uncertain",
}
# Order matches the sample report's Defect Distribution table.
DISTRIBUTION_ORDER = ("present", "missing", "disconnected", "bent", "thin", "uncertain")
DEFECT_VERDICTS = ("missing", "disconnected", "bent", "thin", "uncertain")


def verdict_order_key(verdict: str) -> int:
    """Sort key matching DISTRIBUTION_ORDER, for any listing of verdicts
    (severe-category names, per-region counts, ...) so they read in the
    same order as the Defect Distribution table rather than alphabetically."""
    try:
        return DISTRIBUTION_ORDER.index(verdict)
    except ValueError:
        return len(DISTRIBUTION_ORDER)


@dataclass
class ReportingConfig:
    """Tunable knobs, factored out so a caller can override without editing
    code. Defaults reproduce the sample report's documented behavior."""

    # Verdicts counted toward "severe defects" (sample: 410 missing + 153
    # disconnected = 563, exactly the sample's severe-defect count).
    severe_verdicts: frozenset[str] = field(default_factory=lambda: frozenset({"missing", "disconnected"}))
    # Condition-rating cutoffs on defective percentage. The one calibration
    # point available (6.58% defective -> "Good" in the sample) is satisfied
    # by any monotonic scheme with a "Good" band spanning 6.58%; this is
    # documented here as the assumption it is, not a recovered constant.
    condition_bands: tuple[tuple[float, str], ...] = (
        (5.0, "Excellent"),
        (15.0, "Good"),
        (30.0, "Fair"),
        (100.0, "Poor"),
    )
    # Concentration ratio above which a spatial/cube region is flagged --
    # this value IS stated verbatim in the sample report's text ("meets the
    # 1.50 relative threshold").
    concentration_ratio_threshold: float = 1.50


def _condition_rating(defective_pct: float, config: ReportingConfig) -> str:
    for cutoff, label in config.condition_bands:
        if defective_pct <= cutoff:
            return label
    return config.condition_bands[-1][1]


def _find_records(data: Any) -> tuple[list[Any] | None, str | None]:
    """Locate the strut-level record list inside a defect-results JSON of
    unknown shape. Checked keys mirror the registered-design side's
    STRUT_COLLECTION_KEYS so "a strut collection" means the same thing on
    both sides of the report."""
    if isinstance(data, list):
        return data, None
    if not isinstance(data, Mapping):
        return None, None
    for key in STRUT_COLLECTION_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return value, key
    # Some producers nest everything under "results" or "data".
    for key in ("results", "data", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return value, key
    return None, None


def _verdict_of(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    for key in VERDICT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value.casefold()
    return None


@dataclass
class ReportAnalysis:
    """The "whole-object" defect summary -- everything the Executive
    Summary / Source Information / Defect Distribution / Key Findings /
    Data Validation / Limitations / Recommended Next Steps sections of the
    sample report are built from."""

    specimen: str
    input_json_path: Path | None
    generated_at: datetime
    detected_structure_key: str | None
    total_struts: int
    counts: dict[str, int]                 # raw verdict -> count, zero-filled for known verdicts
    distribution_rows: list[tuple[str, str, int, float]]  # (verdict, display_label, count, pct)
    intact_count: int
    intact_pct: float
    defective_count: int
    defective_pct: float
    severe_count: int
    severe_pct: float
    most_common_defect: str | None         # display label
    condition_rating: str
    key_findings: list[str]
    warnings: list[str]
    meta_volume_shape_zyx: list[int] | None
    records: list[Any] = field(default_factory=list, repr=False)

    LIMITATIONS = (
        "The report depends on the supplied defect-detection output.",
        "The reporting agent does not independently verify the 3D CT scan.",
        "Defect counts do not describe the structural importance of individual struts.",
        "Defect percentages do not determine structural safety.",
        "Location-aware engineering analysis and FEM would be required for structural conclusions.",
    )
    RECOMMENDED_NEXT_STEPS = (
        "Inspect missing and broken/disconnected struts in the original CT volume.",
        "Use the endpoint coordinates to visualize defects and investigate connected clusters.",
        "Compare predictions with the registered STL or design JSON where available.",
        "Perform structural analysis before drawing engineering or safety conclusions.",
    )


class DefectResultsAnalysisError(ValueError):
    """Raised when a defect-results JSON has no recognizable strut records."""


def inspect_defect_results(
    path: Path,
    data: Any,
    *,
    config: ReportingConfig | None = None,
    generated_at: datetime | None = None,
) -> ReportAnalysis:
    """The core "whole-object" analysis: locate strut records, count
    verdicts, and derive the summary fields the report renders."""
    config = config or ReportingConfig()
    records, structure_key = _find_records(data)
    if not records:
        raise DefectResultsAnalysisError(
            f"No strut-level records found in {path} (looked for {STRUT_COLLECTION_KEYS!r})."
        )

    counts: dict[str, int] = {verdict: 0 for verdict in DISTRIBUTION_ORDER}
    unrecognized = 0
    for record in records:
        verdict = _verdict_of(record)
        if verdict is None:
            unrecognized += 1
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
    total = len(records)

    distribution_rows = []
    for verdict in DISTRIBUTION_ORDER:
        count = counts.get(verdict, 0)
        pct = 100 * count / total if total else 0.0
        distribution_rows.append((verdict, VERDICT_DISPLAY.get(verdict, verdict.title()), count, pct))

    intact_count = counts.get("present", 0)
    defective_count = total - intact_count
    defective_pct = 100 * defective_count / total if total else 0.0
    intact_pct = 100 - defective_pct if total else 0.0
    severe_count = sum(counts.get(v, 0) for v in config.severe_verdicts)
    severe_pct = 100 * severe_count / total if total else 0.0

    nonzero_defects = [(v, counts.get(v, 0)) for v in DEFECT_VERDICTS if counts.get(v, 0) > 0]
    most_common = max(nonzero_defects, key=lambda item: item[1])[0] if nonzero_defects else None
    least_common = min(nonzero_defects, key=lambda item: item[1])[0] if nonzero_defects else None

    findings = []
    if total and intact_pct >= 50:
        findings.append("Most struts are intact.")
    elif total:
        findings.append("Less than half of the struts are intact.")
    if most_common and least_common and most_common != least_common:
        findings.append(
            f"{VERDICT_DISPLAY[most_common].split(' (')[0]} is the most frequent nonzero defect; "
            f"{VERDICT_DISPLAY[least_common].split(' (')[0]} is the least frequent nonzero defect."
        )
    elif most_common:
        findings.append(f"{VERDICT_DISPLAY[most_common].split(' (')[0]} is the only nonzero defect category.")
    if severe_count:
        severe_names = " and ".join(
            VERDICT_DISPLAY[v].split(" (")[0] for v in sorted(config.severe_verdicts, key=verdict_order_key)
        )
        findings.append(f"{severe_names} members contribute {severe_count:,} severe defects.")

    warnings = []
    if counts.get("uncertain", 0) == 0:
        warnings.append("Expected category `uncertain` is absent or has zero records.")
    if unrecognized:
        warnings.append(f"{unrecognized} record(s) had no recognizable verdict field and were excluded from counts.")

    meta_volume_shape = None
    if isinstance(data, Mapping):
        meta = data.get("meta")
        if isinstance(meta, Mapping) and isinstance(meta.get("volume_shape_zyx"), Sequence):
            meta_volume_shape = [int(v) for v in meta["volume_shape_zyx"]]

    specimen = path.stem if path else "unknown"
    if isinstance(data, Mapping):
        for container in (data, data.get("meta")):
            if isinstance(container, Mapping):
                specimen = next(
                    (str(container[key]) for key in ("specimen_id", "specimen", "name") if isinstance(container.get(key), (str, int))),
                    specimen,
                )

    return ReportAnalysis(
        specimen=specimen,
        input_json_path=path,
        generated_at=generated_at or datetime.now(timezone.utc),
        detected_structure_key=structure_key,
        total_struts=total,
        counts=counts,
        distribution_rows=distribution_rows,
        intact_count=intact_count,
        intact_pct=intact_pct,
        defective_count=defective_count,
        defective_pct=defective_pct,
        severe_count=severe_count,
        severe_pct=severe_pct,
        most_common_defect=VERDICT_DISPLAY[most_common].split(" (")[0] if most_common else None,
        condition_rating=_condition_rating(defective_pct, config),
        key_findings=findings,
        warnings=warnings,
        meta_volume_shape_zyx=meta_volume_shape,
        records=records,
    )


@dataclass
class GeneratedReportPaths:
    defect_json_path: Path
    markdown_path: Path
    pdf_path: Path


class PartialReportGenerationError(RuntimeError):
    """Markdown was written successfully but PDF rendering failed -- the
    caller still has a usable (markdown) report, so this carries both
    paths plus the underlying cause rather than losing the partial result."""

    def __init__(self, markdown_path: Path, pdf_path: Path, cause: Exception) -> None:
        super().__init__(f"PDF generation failed after writing {markdown_path}: {cause}")
        self.markdown_path = markdown_path
        self.pdf_path = pdf_path
        self.cause = cause


def _distribution_table_rows(overall: ReportAnalysis) -> list[tuple[str, str, str]]:
    # "uncertain" is never a real category here (see VERDICT_DISPLAY's
    # comment) -- shown as a warning instead of a permanently-zero row.
    rows = [
        (label, f"{count:,}", f"{pct:.2f}%")
        for verdict, label, count, pct in overall.distribution_rows
        if verdict != "uncertain"
    ]
    rows.append(("Total defective", f"{overall.defective_count:,}", f"{overall.defective_pct:.2f}%"))
    rows.append(("Severe defects", f"{overall.severe_count:,}", f"{overall.severe_pct:.2f}%"))
    return rows


def render_markdown(overall: ReportAnalysis, *, generated_at: datetime, config: ReportingConfig) -> str:
    """Base (non-design, non-spatial, non-cube) sections of the report, in
    the same numbered order defect_results_reporting_agent.py splices its
    own sections into (it looks for the literal marker "## 3. Executive
    Summary")."""
    lines = [
        "# Defect Detection Results Report",
        "",
        f"Specimen: {overall.specimen}",
        f"Source JSON: {overall.input_json_path.name if overall.input_json_path else 'N/A'}",
        f"Generated: {generated_at.isoformat()}",
        f"Overall condition rating: {overall.condition_rating}",
        "",
        "## 1. Summary Metrics",
        "",
        "(see Executive Summary below)",
        "",
        "## 2. Scope",
        "",
        "This report covers every strut record found in the supplied defect-detection output.",
        "",
        "## 3. Executive Summary",
        "",
        (
            f"The supplied result contains {overall.total_struts:,} struts. "
            f"{overall.intact_count:,} ({overall.intact_pct:.2f}%) are intact and "
            f"{overall.defective_count:,} ({overall.defective_pct:.2f}%) are classified as missing, "
            "broken/disconnected, bent, or thinned."
        ),
        "",
        "This condition rating is based only on detected defect frequency. It is not a structural safety determination.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Struts | {overall.total_struts:,} |",
        f"| Intact Percentage | {overall.intact_pct:.2f}% |",
        f"| Total Defective Percentage | {overall.defective_pct:.2f}% |",
        f"| Severe Defect Percentage | {overall.severe_pct:.2f}% |",
        f"| Most Common Defect | {overall.most_common_defect or 'N/A'} |",
        f"| Condition Rating | {overall.condition_rating} |",
        "",
        "### Source Information",
        "",
        f"Input JSON path: {overall.input_json_path}",
        f"Detected data structure: strut-level records at `{overall.detected_structure_key or 'unknown'}`",
    ]
    if overall.meta_volume_shape_zyx:
        lines.append(f"Meta.Volume Shape Zyx: {overall.meta_volume_shape_zyx}")
    lines += [
        "",
        "## 4. Defect Distribution",
        "",
        "| Strut Condition | Count | Percentage |",
        "|---|---:|---:|",
    ]
    for label, count_s, pct_s in _distribution_table_rows(overall):
        lines.append(f"| {label} | {count_s} | {pct_s} |")
    lines += ["", "## 5. Key Findings", ""]
    lines += [f"- {finding}" for finding in overall.key_findings] or ["- No notable findings."]
    lines += ["", "## 6. Data Validation and Warnings", ""]
    lines += [f"- {warning}" for warning in overall.warnings] or ["- No validation warnings."]
    lines += ["", "## 7. Limitations", ""]
    lines += [f"- {item}" for item in ReportAnalysis.LIMITATIONS]
    lines += ["", "## 8. Recommended Next Steps", ""]
    lines += [f"- {item}" for item in ReportAnalysis.RECOMMENDED_NEXT_STEPS]
    lines.append("")
    return "\n".join(lines)


def render_pdf(
    overall: ReportAnalysis,
    pdf_path: Path,
    *,
    generated_at: datetime,
    config: ReportingConfig,
    leading_story: list[Any] | None = None,
    extra_story: list[Any] | None = None,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("Defect Detection Results Report", styles["Title"]),
        Paragraph(f"Specimen: {escape(overall.specimen)}", styles["BodyText"]),
        Paragraph(
            f"Source JSON: {escape(overall.input_json_path.name if overall.input_json_path else 'N/A')}",
            styles["BodyText"],
        ),
        Paragraph(f"Generated: {generated_at.isoformat()}", styles["BodyText"]),
        Paragraph(f"Overall condition rating: {escape(overall.condition_rating)}", styles["BodyText"]),
        Spacer(1, 10),
    ]
    story.extend(leading_story or [])

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        escape(
            f"The supplied result contains {overall.total_struts:,} struts. "
            f"{overall.intact_count:,} ({overall.intact_pct:.2f}%) are intact and "
            f"{overall.defective_count:,} ({overall.defective_pct:.2f}%) are classified as missing, "
            "broken/disconnected, bent, or thinned."
        ),
        styles["BodyText"],
    ))
    story.append(Paragraph(
        "This condition rating is based only on detected defect frequency. It is not a structural safety determination.",
        styles["BodyText"],
    ))
    summary_rows = [
        ["Metric", "Value"],
        ["Total Struts", f"{overall.total_struts:,}"],
        ["Intact Percentage", f"{overall.intact_pct:.2f}%"],
        ["Total Defective Percentage", f"{overall.defective_pct:.2f}%"],
        ["Severe Defect Percentage", f"{overall.severe_pct:.2f}%"],
        ["Most Common Defect", overall.most_common_defect or "N/A"],
        ["Condition Rating", overall.condition_rating],
    ]
    summary_table = Table(summary_rows, repeatRows=1, colWidths=[2.7 * inch, 2.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([summary_table, Spacer(1, 8)])

    story.append(Paragraph("Source Information", styles["Heading3"]))
    story.append(Paragraph(f"Input JSON path: {escape(str(overall.input_json_path))}", styles["BodyText"]))
    story.append(Paragraph(
        f"Detected data structure: strut-level records at `{escape(overall.detected_structure_key or 'unknown')}`",
        styles["BodyText"],
    ))
    if overall.meta_volume_shape_zyx:
        story.append(Paragraph(f"Meta.Volume Shape Zyx: {overall.meta_volume_shape_zyx}", styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Defect Distribution", styles["Heading2"]))
    dist_rows = [["Strut Condition", "Count", "Percentage"], *_distribution_table_rows(overall)]
    dist_table = Table(dist_rows, repeatRows=1, colWidths=[2.7 * inch, 1.3 * inch, 1.3 * inch])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([dist_table, Spacer(1, 8)])

    story.append(Paragraph("Key Findings", styles["Heading2"]))
    for finding in (overall.key_findings or ["No notable findings."]):
        story.append(Paragraph(f"- {escape(finding)}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Data Validation and Warnings", styles["Heading2"]))
    for warning in (overall.warnings or ["No validation warnings."]):
        story.append(Paragraph(f"- {escape(warning)}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Limitations", styles["Heading2"]))
    for item in ReportAnalysis.LIMITATIONS:
        story.append(Paragraph(f"- {escape(item)}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Recommended Next Steps", styles["Heading2"]))
    for item in ReportAnalysis.RECOMMENDED_NEXT_STEPS:
        story.append(Paragraph(f"- {escape(item)}", styles["BodyText"]))

    story.extend(extra_story or [])

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    doc.build(story)
