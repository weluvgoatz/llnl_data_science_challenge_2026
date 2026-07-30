"""Version 2 analysis: spatial defect concentration on a coordinate grid.

Bins every strut by its midpoint into an N x N x N grid (default 3x3x3,
matching analysis/defect_detection/octet_9x9x9_defect_report.pdf's "Spatial
Defect Concentration Analysis" section), ranks bins by defect rate, and
flags bins whose defect rate is disproportionately high relative to the
object as a whole (the report's "concentration ratio").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
from xml.sax.saxutils import escape

from src.defect_report_agent import ReportAnalysis, ReportingConfig, _find_records, _verdict_of, VERDICT_DISPLAY, verdict_order_key

AXIS_LABELS = ("X", "Y", "Z")
BIN_LABELS = ("low", "middle", "high")


@dataclass
class SpatialConcentrationConfig:
    grid: tuple[int, int, int] = (3, 3, 3)
    top_n: int = 3


@dataclass
class GridRegion:
    label: str                          # e.g. "X-low / Y-high / Z-high"
    bounds: dict[str, tuple[float, float]]
    struts: int
    defects: int
    defect_pct: float
    severe_pct: float
    ratio: float
    dominant: str | None                # display label
    counts: dict[str, int]              # verdict -> count, defect verdicts only
    flagged: bool


@dataclass
class SpatialConcentrationAnalysis:
    grid: tuple[int, int, int]
    regions_ranked: list[GridRegion]    # top_n, descending defect_pct
    threshold: float


def _endpoint_midpoint(record: Any) -> tuple[float, float, float] | None:
    if not isinstance(record, dict):
        return None
    p0, p1 = record.get("p0"), record.get("p1")
    if not (isinstance(p0, Sequence) and isinstance(p1, Sequence) and len(p0) == 3 and len(p1) == 3):
        return None
    try:
        return tuple((float(p0[i]) + float(p1[i])) / 2 for i in range(3))
    except (TypeError, ValueError):
        return None


def _bin_index(value: float, low: float, high: float, n: int) -> int:
    if high <= low:
        return 0
    frac = (value - low) / (high - low)
    return min(n - 1, max(0, int(frac * n)))


def analyze_spatial_concentration(
    defect_data: Any,
    overall: ReportAnalysis,
    config: ReportingConfig,
    spatial_config: SpatialConcentrationConfig | None,
) -> SpatialConcentrationAnalysis:
    spatial_config = spatial_config or SpatialConcentrationConfig()
    records, _ = _find_records(defect_data)
    records = records or []
    points = []
    for record in records:
        mid = _endpoint_midpoint(record)
        verdict = _verdict_of(record)
        if mid is not None and verdict is not None:
            points.append((mid, verdict))

    if not points:
        return SpatialConcentrationAnalysis(spatial_config.grid, [], config.concentration_ratio_threshold)

    nx, ny, nz = spatial_config.grid
    xs = [p[0][0] for p in points]
    ys = [p[0][1] for p in points]
    zs = [p[0][2] for p in points]
    bounds_all = ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))

    bins: dict[tuple[int, int, int], list[tuple[tuple[float, float, float], str]]] = {}
    for mid, verdict in points:
        key = (
            _bin_index(mid[0], *bounds_all[0], nx),
            _bin_index(mid[1], *bounds_all[1], ny),
            _bin_index(mid[2], *bounds_all[2], nz),
        )
        bins.setdefault(key, []).append((mid, verdict))

    total_defects = sum(1 for _, v in points if v != "present")
    overall_rate = total_defects / len(points) if points else 0.0

    regions: list[GridRegion] = []
    for (bx, by, bz), members in bins.items():
        n = len(members)
        counts: dict[str, int] = {}
        for _, verdict in members:
            if verdict != "present":
                counts[verdict] = counts.get(verdict, 0) + 1
        defects = sum(counts.values())
        defect_pct = 100 * defects / n if n else 0.0
        severe = sum(counts.get(v, 0) for v in config.severe_verdicts)
        severe_pct = 100 * severe / n if n else 0.0
        ratio = (defects / n) / overall_rate if overall_rate and n else 0.0
        dominant_verdict = max(counts, key=counts.get) if counts else None
        mxs = [m[0][0] for m in members]
        mys = [m[0][1] for m in members]
        mzs = [m[0][2] for m in members]
        region_bounds = {"x": (min(mxs), max(mxs)), "y": (min(mys), max(mys)), "z": (min(mzs), max(mzs))}
        label = " / ".join(
            f"{axis}-{BIN_LABELS[idx]}" for axis, idx in zip(AXIS_LABELS, (bx, by, bz))
        )
        regions.append(GridRegion(
            label=label, bounds=region_bounds, struts=n, defects=defects, defect_pct=defect_pct,
            severe_pct=severe_pct, ratio=ratio,
            dominant=VERDICT_DISPLAY.get(dominant_verdict, dominant_verdict).split(" (")[0] if dominant_verdict else None,
            counts=counts, flagged=ratio >= config.concentration_ratio_threshold,
        ))

    regions.sort(key=lambda r: r.defect_pct, reverse=True)
    ranked = regions[: spatial_config.top_n]
    return SpatialConcentrationAnalysis(spatial_config.grid, ranked, config.concentration_ratio_threshold)


def render_spatial_markdown(spatial: SpatialConcentrationAnalysis) -> str:
    if not spatial.regions_ranked:
        return "\n## Spatial Defect Concentration Analysis\n\nNo strut coordinates were available for spatial binning.\n"
    nx, ny, nz = spatial.grid
    lines = [
        "",
        "## Spatial Defect Concentration Analysis",
        "",
        f"Grouping method: coordinate grid with {nx}x{ny}x{nz} bins",
        "",
        "| Rank | Region | Struts | Defects | Defect % | Ratio | Dominant |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, region in enumerate(spatial.regions_ranked, start=1):
        lines.append(
            f"| {rank} | {region.label} | {region.struts} | {region.defects} | "
            f"{region.defect_pct:.2f}% | {region.ratio:.2f} | {region.dominant or 'N/A'} |"
        )
    lines += ["", "### Detailed Regional Findings", ""]
    for region in spatial.regions_ranked:
        bounds = "; ".join(f"{axis}={lo:.2f}-{hi:.2f}" for axis, (lo, hi) in region.bounds.items())
        counts_str = ", ".join(f"{VERDICT_DISPLAY.get(v, v).split(' (')[0].lower()}={c}" for v, c in sorted(region.counts.items(), key=lambda kv: verdict_order_key(kv[0])))
        lines += [
            f"**{region.label}**",
            f"Location: Grid region {region.label}; midpoint ranges {bounds} (source-coordinate units)",
            f"Defects: {region.defects}/{region.struts} ({region.defect_pct:.2f}%); ratio {region.ratio:.2f}; "
            f"severe {region.severe_pct:.2f}%; dominant {region.dominant or 'N/A'}.",
            f"Counts: {counts_str}." if counts_str else "Counts: none.",
            (
                f"Flagged because: concentration ratio {region.ratio:.2f} meets the {spatial.threshold:.2f} relative threshold"
                if region.flagged else
                f"Not flagged: concentration ratio {region.ratio:.2f} is below the {spatial.threshold:.2f} relative threshold"
            ),
            "",
        ]
    return "\n".join(lines)


def build_spatial_pdf_story(spatial: SpatialConcentrationAnalysis) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    story: list[Any] = [Spacer(1, 10), Paragraph("Spatial Defect Concentration Analysis", styles["Heading2"])]
    if not spatial.regions_ranked:
        story.append(Paragraph("No strut coordinates were available for spatial binning.", styles["BodyText"]))
        return story
    nx, ny, nz = spatial.grid
    story.append(Paragraph(f"Grouping method: coordinate grid with {nx}x{ny}x{nz} bins", styles["BodyText"]))
    rows = [["Rank", "Region", "Struts", "Defects", "Defect %", "Ratio", "Dominant"]]
    for rank, region in enumerate(spatial.regions_ranked, start=1):
        rows.append([
            str(rank), region.label, str(region.struts), str(region.defects),
            f"{region.defect_pct:.2f}%", f"{region.ratio:.2f}", region.dominant or "N/A",
        ])
    table = Table(rows, repeatRows=1, colWidths=[0.4 * inch, 1.6 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch, 0.6 * inch, 0.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]))
    story.extend([table, Spacer(1, 8), Paragraph("Detailed Regional Findings", styles["Heading3"])])
    for region in spatial.regions_ranked:
        bounds = "; ".join(f"{axis}={lo:.2f}-{hi:.2f}" for axis, (lo, hi) in region.bounds.items())
        story.append(Paragraph(f"<b>{escape(region.label)}</b>", styles["BodyText"]))
        story.append(Paragraph(
            escape(
                f"Defects: {region.defects}/{region.struts} ({region.defect_pct:.2f}%); ratio {region.ratio:.2f}; "
                f"severe {region.severe_pct:.2f}%; dominant {region.dominant or 'N/A'}."
            ),
            styles["BodyText"],
        ))
        flag_text = (
            f"Flagged because: concentration ratio {region.ratio:.2f} meets the {spatial.threshold:.2f} relative threshold"
            if region.flagged else
            f"Not flagged: concentration ratio {region.ratio:.2f} is below the {spatial.threshold:.2f} relative threshold"
        )
        story.append(Paragraph(escape(flag_text), styles["BodyText"]))
        story.append(Spacer(1, 4))
    return story
