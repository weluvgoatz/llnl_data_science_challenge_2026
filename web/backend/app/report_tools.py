"""Report-generation tools for report_agent: turns this job's current v2
classification output into a 3-section NDE report (Markdown + PDF).

Purely additive on top of the existing pipeline -- nothing here changes
detect_v2.py, registration, or any existing agent_tools.py/plot_tools.py
function. It reuses the SAME classification.json every other tool already
reads (agent_tools.load_classification), so it's always working from
whatever the active v2 detection run actually produced, never a separate or
stale copy.

Two-step tool flow, mirroring the "tools compute, the LLM narrates" pattern
used everywhere else in this app:

  1. generate_report_data(job_id) -- deterministic. Computes every number
     the report needs (via the reconstructed defect-results reporting
     agent: analysis/defect_detection/defect_results_reporting_agent.py +
     its v2/v3 dependencies in analysis/defect_detection/agents/ and
     src/defect_report_agent.py) and renders the plots for Section 2.
     Writes an intermediate cache (report/_report_data.json, job-scoped)
     and returns the real computed numbers to the caller -- nothing
     invented, everything sourced, same as every other tool in this app.
  2. finalize_report(job_id, situational_analysis) -- report_agent (the
     LLM) writes the Section 3 interpretation grounded in the numbers step
     1 returned, and this tool assembles the final Markdown + PDF from the
     cache plus that text, overwriting job["report"] (same field/surface
     ReportView already renders -- see workflow.py's automatic per-analysis
     report, which this only replaces when explicitly asked for) and
     registering the PDF as a downloadable artifact.

Every call recomputes from disk -- step 1 has no memoization across calls,
so re-running it after a rerun_classification always reflects the new
active version, never a stale one.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shutil

from . import agent_tools as tools
from . import plot_tools
from .store import job_dir, load_job, update_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.defect_report_agent import (  # noqa: E402
    ReportingConfig,
    inspect_defect_results,
)
from analysis.defect_detection.defect_results_reporting_agent import (  # noqa: E402
    extract_registered_design_summary,
    calculate_evaluation_coverage,
    render_registered_design_markdown,
    build_registered_design_pdf_story,
)
from analysis.defect_detection.agents.defect_results_reporting_agent_v2 import (  # noqa: E402
    analyze_spatial_concentration,
    render_spatial_markdown,
    build_spatial_pdf_story,
    SpatialConcentrationConfig,
)
from analysis.defect_detection.agents.defect_results_reporting_agent_v3 import (  # noqa: E402
    analyze_cube_regions,
    render_cube_markdown,
    build_cube_pdf_story,
    CubeRegionAnalysisConfig,
)

_CACHE_NAME = "_report_data.json"


def _report_dir(job_id: str) -> Path:
    out = job_dir(job_id) / "report"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _input_files_markdown(job: dict[str, Any]) -> str:
    lines = ["### Uploaded Input Files", "", "| File | Kind | Details |", "|---|---|---|"]
    for f in job["files"]:
        lines.append(f"| {f['name']} | {f['kind'].upper()} | {f.get('summary', '')} |")
    return "\n".join(lines)


def _clean_specimen_id(design_summary: Any, job: dict[str, Any], design_path: Path | None) -> None:
    """extract_registered_design_summary() (reused verbatim from the pushed
    agent) falls back to the design JSON's own path stem when the file has
    no specimen_id/specimen/name field -- true of every design JSON this
    pipeline produces. On disk that stem carries the upload's file-id
    prefix (see main.py's upload_files: "{file_id}-{safe_name}"), so the
    fallback would show a UUID-prefixed mess as "Specimen ID" instead of
    the design file's actual name. Overridden here, in the wiring layer,
    rather than in the reused function itself."""
    design_item = next((f for f in job["files"] if f["kind"] == "json"), None)
    if not design_item or not design_path:
        return
    if design_summary.specimen_id in (None, design_path.stem):
        design_summary.specimen_id = Path(design_item["name"]).stem


def _scan_parameters_markdown(meta: dict[str, Any]) -> str:
    thresholds = meta.get("thresholds") or {}
    rows = [("Volume shape (Z, Y, X)", meta.get("volume_shape_zyx")),
            ("Voxel size (um)", thresholds.get("voxel_um")),
            ("Nominal strut diameter (um)", thresholds.get("nominal_diameter_um"))]
    lines = ["### Scan Physical Parameters", "", "| Parameter | Value |", "|---|---:|"]
    for label, value in rows:
        if value is not None:
            lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _renumber_headings(markdown: str, bump: int) -> str:
    """Shift every "## "/"### " heading in a block of Markdown down by
    `bump` levels so it nests correctly under this report's own numbered
    sections, without hand-rewriting the render_*_markdown functions
    themselves (they're shared with the standalone reconstructed agent,
    which has its own top-level heading structure -- see
    analysis/defect_detection/defect_results_reporting_agent.py)."""
    out_lines = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out_lines.append("#" * (level + bump) + line[level:])
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def generate_report_data(job_id: str) -> dict[str, Any]:
    """Compute every number + plot the report needs from this job's
    CURRENT active v2 classification (report_agent). Always recomputes
    from disk -- never reuses a previous call's numbers -- so it reflects
    whatever the active classification version is right now, including
    after a rerun_classification.

    Returns the real computed statistics (defect distribution, hotspot
    regions/faces, condition rating) so report_agent can ground its
    Section 3 interpretation in them, then call finalize_report.
    """
    job = load_job(job_id)
    classification = tools.load_classification(job)
    design = tools.load_design(job)

    base = job_dir(job_id)
    classified_path = base / (job.get("defects") or {}).get("path", "defects/classification.json")
    design_item = next((f for f in job["files"] if f["kind"] == "json"), None)
    design_path = base / design_item["path"] if design_item else None

    config = ReportingConfig()
    generated_at = datetime.now(timezone.utc)
    overall = inspect_defect_results(classified_path, classification, config=config, generated_at=generated_at)
    # classification.json has no specimen name of its own (v2's meta only
    # carries counts/n/volume_shape_zyx/thresholds) -- prefer the job's
    # actual uploaded scan filename over inspect_defect_results' fallback
    # (the classification file's own stem, "classification").
    tiff_item = next((f for f in job["files"] if f["kind"] == "tiff"), None)
    if tiff_item:
        overall.specimen = Path(tiff_item["name"]).stem
    design_summary = extract_registered_design_summary(design_path, design)
    _clean_specimen_id(design_summary, job, design_path)
    coverage = calculate_evaluation_coverage(design, classification, design_summary)
    spatial = analyze_spatial_concentration(classification, overall, config, SpatialConcentrationConfig())
    cube = analyze_cube_regions(classification, overall, config, CubeRegionAnalysisConfig())

    verdict_plot = plot_tools.plot_verdict_counts(job_id)
    try:
        hotspot_plot = plot_tools.plot_hotspots_map(job_id, top_n=15)
    except tools.ToolError:
        hotspot_plot = None  # no design unit_cells, or nothing to rank -- report still works without it
    face_stats = [
        {"face": f.face, "defect_pct": f.defect_pct, "severe_pct": f.severe_pct, "dominant": f.dominant}
        for f in cube.faces_ranked
    ]
    try:
        face_plot = plot_tools.plot_face_defect_rates(job_id, face_stats) if face_stats else None
    except tools.ToolError:
        face_plot = None

    cache = {
        "generated_at": generated_at.isoformat(),
        "classified_path": str(classified_path),
        "design_path": str(design_path) if design_path else None,
        "overall": {
            "specimen": overall.specimen,
            "total_struts": overall.total_struts,
            "intact_count": overall.intact_count,
            "intact_pct": overall.intact_pct,
            "defective_count": overall.defective_count,
            "defective_pct": overall.defective_pct,
            "severe_count": overall.severe_count,
            "severe_pct": overall.severe_pct,
            "most_common_defect": overall.most_common_defect,
            "condition_rating": overall.condition_rating,
            "key_findings": overall.key_findings,
            "warnings": overall.warnings,
            "detected_structure_key": overall.detected_structure_key,
            "meta_volume_shape_zyx": overall.meta_volume_shape_zyx,
            "distribution_rows": overall.distribution_rows,
        },
        "design_summary_markdown": render_registered_design_markdown(design_summary, coverage),
        "spatial_markdown": render_spatial_markdown(spatial),
        "cube_markdown": render_cube_markdown(cube, CubeRegionAnalysisConfig()),
        "top_spatial_regions": [
            {"label": r.label, "defect_pct": r.defect_pct, "ratio": r.ratio, "dominant": r.dominant}
            for r in spatial.regions_ranked
        ],
        "top_faces": face_stats,
        "highly_concentrated_volumes": [
            {"name": r.name, "defect_pct": r.defect_pct, "ratio": r.ratio, "dominant": r.dominant}
            for r in cube.highly_concentrated
        ],
        "plots": {
            "verdict_counts": verdict_plot["artifact"],
            "hotspots_map": hotspot_plot["artifact"] if hotspot_plot else None,
            "face_defect_rates": face_plot["artifact"] if face_plot else None,
        },
        "input_files": [{"name": f["name"], "kind": f["kind"], "summary": f.get("summary", "")} for f in job["files"]],
    }
    # Keep the "story" flowables (registered design + spatial + cube, all
    # reused verbatim from the reconstructed agent) out of the JSON cache --
    # they're rebuilt fresh in finalize_report from the same source data
    # (design_summary/coverage/spatial/cube are cheap to recompute; there's
    # no benefit to serializing reportlab objects).
    cache_path = _report_dir(job_id) / _CACHE_NAME
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return {
        "source": "defect_results_reporting_agent.py (v2 classification, current active version)",
        "specimen": overall.specimen,
        "total_struts": overall.total_struts,
        "defective_count": overall.defective_count,
        "defective_pct": round(overall.defective_pct, 2),
        "severe_count": overall.severe_count,
        "most_common_defect": overall.most_common_defect,
        "condition_rating": overall.condition_rating,
        "key_findings": overall.key_findings,
        "top_spatial_regions": cache["top_spatial_regions"],
        "top_faces": cache["top_faces"],
        "highly_concentrated_volumes": cache["highly_concentrated_volumes"],
        "plots_generated": [k for k, v in cache["plots"].items() if v],
        "note": (
            "Call finalize_report(job_id, situational_analysis=...) next: write the manufacturing-"
            "process interpretation grounded in these numbers (dominant defect, which spatial "
            "regions/faces are concentrated, what that pattern plausibly implies about the print "
            "process, and what to check in the production pipeline). Do not invent numbers beyond "
            "what's returned here."
        ),
    }


def finalize_report(job_id: str, situational_analysis: str) -> dict[str, Any]:
    """Assemble the final 3-section report from generate_report_data's
    cache plus report_agent's own Section 3 interpretation (report_agent).
    Writes report/report.md and report/report.pdf (job-scoped, never a
    shared path), overwrites job["report"] (same field/ReportView surface
    the automatic per-analysis report already uses), and registers the PDF
    as a downloadable artifact.
    """
    job = load_job(job_id)
    cache_path = _report_dir(job_id) / _CACHE_NAME
    if not cache_path.is_file():
        raise tools.ToolError("No report data computed yet -- call generate_report_data first.")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    overall = cache["overall"]

    classification = tools.load_classification(job)
    design = tools.load_design(job)
    classified_path = Path(cache["classified_path"])
    design_path = Path(cache["design_path"]) if cache["design_path"] else None

    config = ReportingConfig()
    generated_at = datetime.now(timezone.utc)
    # Recomputed (cheap, deterministic) rather than deserialized from JSON --
    # see generate_report_data's comment on why the cache only stores the
    # rendered text/summaries, not these objects themselves.
    fresh_overall = inspect_defect_results(classified_path, classification, config=config, generated_at=generated_at)
    design_summary = extract_registered_design_summary(design_path, design)
    _clean_specimen_id(design_summary, job, design_path)
    coverage = calculate_evaluation_coverage(design, classification, design_summary)
    spatial = analyze_spatial_concentration(classification, fresh_overall, config, SpatialConcentrationConfig())
    cube = analyze_cube_regions(classification, fresh_overall, config, CubeRegionAnalysisConfig())

    md_lines = [
        "# Lattice NDE Report",
        "",
        f"Specimen: {overall['specimen']}",
        f"Generated: {generated_at.isoformat()}",
        f"Overall condition rating: {overall['condition_rating']}",
        "",
        "## 1. Input Metadata & Statistics",
        "",
        _renumber_headings(cache["design_summary_markdown"], bump=1),
        "",
        _input_files_markdown(job),
        "",
        _scan_parameters_markdown({"volume_shape_zyx": overall["meta_volume_shape_zyx"]}),
        "",
        "## 2. Output Statistics & Plots",
        "",
        "| Strut Condition | Count | Percentage |",
        "|---|---:|---:|",
    ]
    for verdict, label, count, pct in overall["distribution_rows"]:
        if verdict != "uncertain":
            md_lines.append(f"| {label} | {count:,} | {pct:.2f}% |")
    warning_lines = [f"- {warning}" for warning in overall["warnings"]] or ["- No validation warnings."]
    md_lines += [
        f"| Total defective | {overall['defective_count']:,} | {overall['defective_pct']:.2f}% |",
        f"| Severe defects | {overall['severe_count']:,} | {overall['severe_pct']:.2f}% |",
        "",
        "### Key Findings",
        "",
        *(f"- {finding}" for finding in overall["key_findings"]),
        "",
        "### Data Validation and Warnings",
        "",
        *warning_lines,
        "",
        "### Plots",
        "",
    ]
    for label, artifact in cache["plots"].items():
        if artifact:
            md_lines.append(f"- {artifact['caption']}: `{artifact['path']}`")
    md_lines += [
        "",
        _renumber_headings(cache["spatial_markdown"], bump=1),
        _renumber_headings(cache["cube_markdown"], bump=1),
        "",
        "## 3. Situational Analysis",
        "",
        situational_analysis.strip(),
        "",
        "### Limitations",
        "",
        "- The report depends on the supplied defect-detection output.",
        "- The reporting agent does not independently verify the 3D CT scan.",
        "- Defect counts do not describe the structural importance of individual struts.",
        "- Defect percentages do not determine structural safety.",
        "- Location-aware engineering analysis and FEM would be required for structural conclusions.",
        "",
        "### Recommended Next Steps",
        "",
        "- Inspect missing and broken/disconnected struts in the original CT volume.",
        "- Use the endpoint coordinates to visualize defects and investigate connected clusters.",
        "- Compare predictions with the registered STL or design JSON where available.",
        "- Perform structural analysis before drawing engineering or safety conclusions.",
        "",
    ]
    markdown_text = "\n".join(md_lines)

    report_dir = _report_dir(job_id)
    markdown_path = report_dir / "report.md"
    pdf_path = report_dir / "report.pdf"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    _render_report_pdf(
        overall, design_summary, coverage, spatial, cube, situational_analysis,
        pdf_path, generated_at, cache,
    )

    root = job_dir(job_id)

    def apply(current: dict[str, Any]) -> None:
        current["report"] = {
            "name": "Lattice NDE report.md",
            "path": str(markdown_path.relative_to(root)),
            "mediaType": "text/markdown",
        }

    update_job(job_id, apply)
    pdf_artifact = _register_pdf_copy(job_id, pdf_path, "Lattice NDE report (PDF)")

    return {
        "source": "report_tools.finalize_report",
        "markdown_path": str(markdown_path.relative_to(root)),
        "pdf_artifact": pdf_artifact,
        "condition_rating": overall["condition_rating"],
        "defective_pct": overall["defective_pct"],
    }


def _register_pdf_copy(job_id: str, pdf_path: Path, caption: str) -> dict[str, Any]:
    """Register a COPY of pdf_path as a downloadable artifact -- unlike
    agent_tools._move_and_register (move semantics, used by tools whose
    source file lives in a scratch/shared location), this report's PDF
    should stay in place at report/report.pdf for anyone browsing the job
    directory directly, so the original is never relocated."""
    root = job_dir(job_id)
    dest_dir = root / "artifacts" / "report"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf_path.name
    shutil.copy2(pdf_path, dest)
    relative = str(dest.relative_to(root))
    artifact = {
        "id": f"report_{dest.name}",
        "name": dest.name,
        "caption": caption,
        "path": relative,
        "mediaType": "application/pdf",
    }

    def apply(current: dict[str, Any]) -> None:
        artifacts = [a for a in current.get("artifacts", []) if a["id"] != artifact["id"]]
        artifacts.append(artifact)
        current["artifacts"] = artifacts

    update_job(job_id, apply)
    return artifact


def _render_report_pdf(
    overall: dict[str, Any],
    design_summary: Any,
    coverage: Any,
    spatial: Any,
    cube: Any,
    situational_analysis: str,
    pdf_path: Path,
    generated_at: datetime,
    cache: dict[str, Any],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("Lattice NDE Report", styles["Title"]),
        Paragraph(f"Specimen: {escape(str(overall['specimen']))}", styles["BodyText"]),
        Paragraph(f"Generated: {generated_at.isoformat()}", styles["BodyText"]),
        Paragraph(f"Overall condition rating: {escape(str(overall['condition_rating']))}", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("1. Input Metadata & Statistics", styles["Heading1"]),
    ]
    story.extend(build_registered_design_pdf_story(design_summary, coverage))
    story.append(Paragraph("Uploaded Input Files", styles["Heading3"]))
    small = styles["BodyText"].clone("SmallBody2")
    small.fontSize = 7.5
    small.leading = 9
    # Paragraph (not a plain string) for the filename cell -- a long
    # filename in a plain string overflows into the next column instead of
    # wrapping (same fix as the volume-region table in
    # defect_results_reporting_agent_v3.build_cube_pdf_story).
    file_rows = [["File", "Kind", "Details"], *[
        [Paragraph(escape(f["name"]), small), f["kind"].upper(), f["summary"]] for f in cache["input_files"]
    ]]
    file_table = Table(file_rows, repeatRows=1, colWidths=[2.2 * inch, 0.9 * inch, 2.9 * inch])
    file_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([file_table, Spacer(1, 10)])

    story.append(Paragraph("2. Output Statistics & Plots", styles["Heading1"]))
    dist_rows = [["Strut Condition", "Count", "Percentage"]]
    for verdict, label, count, pct in overall["distribution_rows"]:
        if verdict != "uncertain":
            dist_rows.append([label, f"{count:,}", f"{pct:.2f}%"])
    dist_rows.append(["Total defective", f"{overall['defective_count']:,}", f"{overall['defective_pct']:.2f}%"])
    dist_rows.append(["Severe defects", f"{overall['severe_count']:,}", f"{overall['severe_pct']:.2f}%"])
    dist_table = Table(dist_rows, repeatRows=1, colWidths=[2.7 * inch, 1.3 * inch, 1.3 * inch])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([dist_table, Spacer(1, 8)])
    for finding in overall["key_findings"]:
        story.append(Paragraph(f"- {escape(finding)}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    job_root = pdf_path.parents[1]
    for label, artifact in cache["plots"].items():
        if not artifact:
            continue
        image_path = job_root / artifact["path"]
        if image_path.is_file():
            story.append(Paragraph(escape(artifact["caption"]), styles["Heading3"]))
            story.append(Image(str(image_path), width=5.5 * inch, height=5.5 * inch * 0.62, kind="proportional"))
            story.append(Spacer(1, 8))

    story.extend(build_spatial_pdf_story(spatial))
    story.extend(build_cube_pdf_story(cube, CubeRegionAnalysisConfig()))

    story.append(PageBreak())
    story.append(Paragraph("3. Situational Analysis", styles["Heading1"]))
    for paragraph in situational_analysis.strip().split("\n\n"):
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph.strip()), styles["BodyText"]))
            story.append(Spacer(1, 6))
    story.append(Paragraph("Limitations", styles["Heading3"]))
    for item in (
        "The report depends on the supplied defect-detection output.",
        "The reporting agent does not independently verify the 3D CT scan.",
        "Defect counts do not describe the structural importance of individual struts.",
        "Defect percentages do not determine structural safety.",
        "Location-aware engineering analysis and FEM would be required for structural conclusions.",
    ):
        story.append(Paragraph(f"- {escape(item)}", styles["BodyText"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Recommended Next Steps", styles["Heading3"]))
    for item in (
        "Inspect missing and broken/disconnected struts in the original CT volume.",
        "Use the endpoint coordinates to visualize defects and investigate connected clusters.",
        "Compare predictions with the registered STL or design JSON where available.",
        "Perform structural analysis before drawing engineering or safety conclusions.",
    ):
        story.append(Paragraph(f"- {escape(item)}", styles["BodyText"]))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    doc.build(story)
