"""Version 3 analysis: exterior faces and coarse volumetric regions of the
printed object's bounding cube, plus the base reporting-agent class
(input resolution + the "whole-object" .inspect() entry point).

Matches analysis/defect_detection/octet_9x9x9_defect_report.pdf's "Exterior
Faces of the Large Cube" (6 faces) and "Nine Volumetric Regions" (a 3x3x1
grid) sections.

Two constants here are documented assumptions rather than recovered
originals (the sample PDF states the object's exterior faces and a 3x3x1
volume grid but not the exact face-membership margin): FACE_MARGIN_FRACTION
and the axis->face-name mapping (Left/Right=X, Front/Back=Y, Bottom/Top=Z).
Struts near a corner can belong to more than one face, same as in the
sample (face strut counts don't need to sum to the total).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from src.defect_report_agent import (
    ReportAnalysis,
    ReportingConfig,
    _find_records,
    _verdict_of,
    VERDICT_DISPLAY,
    inspect_defect_results,
    verdict_order_key,
)
from analysis.defect_detection.agents.defect_results_reporting_agent_v2 import _endpoint_midpoint

FACE_AXIS = {"Left": ("x", "low"), "Right": ("x", "high"), "Front": ("y", "low"),
             "Back": ("y", "high"), "Bottom": ("z", "low"), "Top": ("z", "high")}
FACE_MARGIN_FRACTION = 0.075
VOLUME_GRID = (3, 3, 1)   # X, Y subdivided into 3; Z spans the full range


@dataclass
class CubeRegionAnalysisConfig:
    face_margin_fraction: float = FACE_MARGIN_FRACTION
    volume_grid: tuple[int, int, int] = VOLUME_GRID


@dataclass
class FaceStat:
    face: str
    struts: int
    defects: int
    defect_pct: float
    severe_pct: float
    dominant: str | None


@dataclass
class VolumeRegionStat:
    name: str
    bounds: dict[str, tuple[float, float]]
    struts: int
    defects: int
    defect_pct: float
    ratio: float
    dominant: str | None
    counts: dict[str, int]
    flagged: bool


@dataclass
class CubeRegionAnalysis:
    faces_ranked: list[FaceStat]              # descending defect_pct
    volume_regions_ranked: list[VolumeRegionStat]  # descending defect_pct
    highly_concentrated: list[VolumeRegionStat]    # flagged == True
    volume_grid: tuple[int, int, int]


def analyze_cube_regions(
    defect_data: Any,
    overall: ReportAnalysis,
    config: ReportingConfig,
    region_config: CubeRegionAnalysisConfig | None,
) -> CubeRegionAnalysis:
    region_config = region_config or CubeRegionAnalysisConfig()
    records, _ = _find_records(defect_data)
    records = records or []
    points = []
    for record in records:
        mid = _endpoint_midpoint(record)
        verdict = _verdict_of(record)
        if mid is not None and verdict is not None:
            points.append((mid, verdict))
    if not points:
        return CubeRegionAnalysis([], [], [], region_config.volume_grid)

    axis_index = {"x": 0, "y": 1, "z": 2}
    bounds_all = {
        axis: (min(p[0][i] for p in points), max(p[0][i] for p in points))
        for axis, i in axis_index.items()
    }
    total_defects = sum(1 for _, v in points if v != "present")
    overall_rate = total_defects / len(points) if points else 0.0

    # -- exterior faces: independent membership, a corner strut can count on multiple faces --
    faces: list[FaceStat] = []
    for face_name, (axis, side) in FACE_AXIS.items():
        lo, hi = bounds_all[axis]
        extent = hi - lo
        margin = extent * region_config.face_margin_fraction
        i = axis_index[axis]
        if side == "low":
            members = [(mid, v) for mid, v in points if mid[i] <= lo + margin]
        else:
            members = [(mid, v) for mid, v in points if mid[i] >= hi - margin]
        n = len(members)
        counts: dict[str, int] = {}
        for _, v in members:
            if v != "present":
                counts[v] = counts.get(v, 0) + 1
        defects = sum(counts.values())
        defect_pct = 100 * defects / n if n else 0.0
        severe = sum(counts.get(v, 0) for v in config.severe_verdicts)
        severe_pct = 100 * severe / n if n else 0.0
        dominant_verdict = max(counts, key=counts.get) if counts else None
        faces.append(FaceStat(
            face=face_name, struts=n, defects=defects, defect_pct=defect_pct, severe_pct=severe_pct,
            dominant=VERDICT_DISPLAY.get(dominant_verdict, dominant_verdict).split(" (")[0] if dominant_verdict else None,
        ))
    faces.sort(key=lambda f: f.defect_pct, reverse=True)

    # -- nine volumetric regions: X,Y subdivided (region_config.volume_grid), Z spans full range --
    nx, ny, nz = region_config.volume_grid

    def bin_index(value: float, lo: float, hi: float, n: int) -> int:
        if n <= 1 or hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        return min(n - 1, max(0, int(frac * n)))

    bins: dict[tuple[int, int, int], list[tuple[tuple[float, float, float], str]]] = {}
    for mid, verdict in points:
        key = (
            bin_index(mid[0], *bounds_all["x"], nx),
            bin_index(mid[1], *bounds_all["y"], ny),
            bin_index(mid[2], *bounds_all["z"], nz),
        )
        bins.setdefault(key, []).append((mid, verdict))

    bin_labels = ("LOW", "MIDDLE", "HIGH")
    regions: list[VolumeRegionStat] = []
    for (bx, by, bz), members in bins.items():
        n = len(members)
        counts = {}
        for _, v in members:
            if v != "present":
                counts[v] = counts.get(v, 0) + 1
        defects = sum(counts.values())
        defect_pct = 100 * defects / n if n else 0.0
        ratio = (defects / n) / overall_rate if overall_rate and n else 0.0
        dominant_verdict = max(counts, key=counts.get) if counts else None
        mxs = [m[0][0] for m in members]; mys = [m[0][1] for m in members]; mzs = [m[0][2] for m in members]
        name = f"VOLUME_X_{bin_labels[bx]}_Y_{bin_labels[by]}" + (f"_Z_{bin_labels[bz]}" if nz > 1 else "")
        regions.append(VolumeRegionStat(
            name=name,
            bounds={"x": (min(mxs), max(mxs)), "y": (min(mys), max(mys)), "z": (min(mzs), max(mzs))},
            struts=n, defects=defects, defect_pct=defect_pct, ratio=ratio,
            dominant=VERDICT_DISPLAY.get(dominant_verdict, dominant_verdict).split(" (")[0] if dominant_verdict else None,
            counts=counts, flagged=ratio >= config.concentration_ratio_threshold,
        ))
    regions.sort(key=lambda r: r.defect_pct, reverse=True)
    highly_concentrated = [r for r in regions if r.flagged]

    return CubeRegionAnalysis(faces, regions, highly_concentrated, region_config.volume_grid)


def render_cube_markdown(cube: CubeRegionAnalysis, region_config: CubeRegionAnalysisConfig | None) -> str:
    if not cube.faces_ranked and not cube.volume_regions_ranked:
        return "\n## Exterior Faces of the Large Cube\n\nNo strut coordinates were available for region binning.\n"
    lines = [
        "",
        "## Exterior Faces of the Large Cube",
        "",
        "This analysis applies only to the six exterior faces of the complete object. Internal "
        "volumetric-region boundaries are not treated as physical faces.",
        "",
        "| Rank | Face | Struts | Defects | Defect % | Severe % | Dominant |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, face in enumerate(cube.faces_ranked, start=1):
        lines.append(
            f"| {rank} | {face.face} | {face.struts} | {face.defects} | {face.defect_pct:.2f}% | "
            f"{face.severe_pct:.2f}% | {face.dominant or 'N/A'} |"
        )
    nx, ny, nz = cube.volume_grid
    lines += [
        "",
        "## Nine Volumetric Regions",
        "",
        f"Volume grid: {nx} x {ny} x {nz}. Total volumetric regions: {len(cube.volume_regions_ranked)}. "
        "No internal region boundary is analyzed as a physical face.",
        "",
        "| Rank | Volume Region | Bounds | Struts | Defects | Defect % | Ratio | Dominant |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, region in enumerate(cube.volume_regions_ranked, start=1):
        bounds = ", ".join(f"{axis}={lo:.2f}-{hi:.2f}" for axis, (lo, hi) in region.bounds.items())
        lines.append(
            f"| {rank} | {region.name} | {bounds} | {region.struts} | {region.defects} | "
            f"{region.defect_pct:.2f}% | {region.ratio:.2f} | {region.dominant or 'N/A'} |"
        )
    lines += ["", "### Highly Concentrated Volumetric Regions", ""]
    if cube.highly_concentrated:
        for region in cube.highly_concentrated:
            counts_str = ", ".join(f"{VERDICT_DISPLAY.get(v, v).split(' (')[0].lower()}={c}" for v, c in sorted(region.counts.items(), key=lambda kv: verdict_order_key(kv[0])))
            lines.append(
                f"**{region.name}**: {region.defects}/{region.struts} defects ({region.defect_pct:.2f}%), "
                f"ratio {region.ratio:.2f}, dominant {region.dominant or 'N/A'}. Counts: {counts_str}. "
                "Flagged by relative concentration threshold."
            )
    else:
        lines.append("No volumetric region met the relative concentration threshold.")
    lines += [
        "",
        "## Comparison of Surface and Volume Findings",
        "",
        "Surface and volume results are reported as separate normalized analyses. No internal "
        "volumetric-region face was created or evaluated.",
        "",
    ]
    return "\n".join(lines)


def build_cube_pdf_story(cube: CubeRegionAnalysis, region_config: CubeRegionAnalysisConfig | None) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    story: list[Any] = [Spacer(1, 10), Paragraph("Exterior Faces of the Large Cube", styles["Heading2"])]
    if not cube.faces_ranked and not cube.volume_regions_ranked:
        story.append(Paragraph("No strut coordinates were available for region binning.", styles["BodyText"]))
        return story
    face_rows = [["Rank", "Face", "Struts", "Defects", "Defect %", "Severe %", "Dominant"]]
    for rank, face in enumerate(cube.faces_ranked, start=1):
        face_rows.append([str(rank), face.face, str(face.struts), str(face.defects),
                           f"{face.defect_pct:.2f}%", f"{face.severe_pct:.2f}%", face.dominant or "N/A"])
    face_table = Table(face_rows, repeatRows=1, colWidths=[0.4 * inch, 0.9 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch])
    face_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]))
    story.extend([face_table, Spacer(1, 8), Paragraph("Nine Volumetric Regions", styles["Heading2"])])
    small = styles["BodyText"].clone("SmallBody")
    small.fontSize = 7.5
    small.leading = 9
    vol_rows = [["Rank", "Volume Region", "Struts", "Defects", "Defect %", "Ratio", "Dominant"]]
    for rank, region in enumerate(cube.volume_regions_ranked, start=1):
        # Paragraph (not a plain string) so a long region name wraps inside
        # its column instead of overflowing into the next one.
        vol_rows.append([str(rank), Paragraph(escape(region.name), small), str(region.struts), str(region.defects),
                          f"{region.defect_pct:.2f}%", f"{region.ratio:.2f}", region.dominant or "N/A"])
    vol_table = Table(vol_rows, repeatRows=1, colWidths=[0.4 * inch, 1.6 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch, 0.6 * inch, 0.9 * inch])
    vol_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]))
    story.extend([vol_table, Spacer(1, 8), Paragraph("Highly Concentrated Volumetric Regions", styles["Heading3"])])
    if cube.highly_concentrated:
        for region in cube.highly_concentrated:
            counts_str = ", ".join(f"{VERDICT_DISPLAY.get(v, v).split(' (')[0].lower()}={c}" for v, c in sorted(region.counts.items(), key=lambda kv: verdict_order_key(kv[0])))
            story.append(Paragraph(
                escape(
                    f"{region.name}: {region.defects}/{region.struts} defects ({region.defect_pct:.2f}%), "
                    f"ratio {region.ratio:.2f}, dominant {region.dominant or 'N/A'}. Counts: {counts_str}. "
                    "Flagged by relative concentration threshold."
                ),
                styles["BodyText"],
            ))
    else:
        story.append(Paragraph("No volumetric region met the relative concentration threshold.", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Comparison of Surface and Volume Findings", styles["Heading3"]))
    story.append(Paragraph(
        "Surface and volume results are reported as separate normalized analyses. No internal "
        "volumetric-region face was created or evaluated.",
        styles["BodyText"],
    ))
    return story


class DefectResultsReportingAgentV3:
    """Base agent: resolves an input path, runs the whole-object analysis
    (.inspect), and knows where a report's default output path lives.
    Subclassed by defect_results_reporting_agent.DefectResultsReportingAgent,
    which adds registered-design discovery and the spatial/cube analyses on
    top of what's here."""

    name = "Defect Results Reporting Agent V3"

    def __init__(
        self,
        config: ReportingConfig | None = None,
        spatial_config: Any | None = None,
        region_config: CubeRegionAnalysisConfig | None = None,
    ) -> None:
        self.config = config or ReportingConfig()
        self.spatial_config = spatial_config
        self.region_config = region_config
        self.last_analysis: ReportAnalysis | None = None

    def _resolve_input(
        self,
        input_path: Path | None,
        search_root: Path | None,
        specimen_id: str | None,
    ) -> Path:
        """An explicit path always wins. Otherwise search for a defect-
        results JSON (a JSON file containing a recognizable strut-record
        collection) under search_root, preferring one whose name matches
        specimen_id."""
        if input_path is not None:
            resolved = Path(input_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Defect-results JSON does not exist: {resolved}")
            return resolved
        root = Path(search_root).resolve() if search_root else Path(__file__).resolve().parents[3]
        candidates = []
        for path in root.rglob("*.json"):
            if any(part.lower() in {".git", ".venv", "venv", "node_modules", "__pycache__"} for part in path.parts):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            records, _ = _find_records(data)
            if records:
                candidates.append(path)
        if not candidates:
            raise FileNotFoundError(f"No defect-results JSON found under {root}.")
        if specimen_id:
            query = specimen_id.casefold()
            matched = [path for path in candidates if query in path.stem.casefold()]
            if matched:
                candidates = matched
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def default_output_path(self, selected: Path) -> Path:
        return selected.with_name(f"{selected.stem}_v3")

    def inspect(self, path: Path) -> ReportAnalysis:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        overall = inspect_defect_results(path, data, config=self.config, generated_at=datetime.now(timezone.utc))
        self.last_analysis = overall
        return overall
