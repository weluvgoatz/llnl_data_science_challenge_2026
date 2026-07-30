"""Canonical unified defect-results reporting agent.

This is the primary supported interface. It composes the latest Version 3
whole-object/spatial analysis with deterministic registered-design discovery,
metadata extraction, and evaluation-coverage checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from analysis.defect_detection.agents.defect_results_reporting_agent_v2 import (
    SpatialConcentrationAnalysis,
    analyze_spatial_concentration,
    build_spatial_pdf_story,
    render_spatial_markdown,
)
from analysis.defect_detection.agents.defect_results_reporting_agent_v3 import (
    CubeRegionAnalysis,
    CubeRegionAnalysisConfig,
    DefectResultsReportingAgentV3,
    analyze_cube_regions,
    build_cube_pdf_story,
    render_cube_markdown,
)
from src.defect_report_agent import (
    GeneratedReportPaths,
    PartialReportGenerationError,
    ReportAnalysis,
    ReportingConfig,
    _find_records,
    render_markdown,
    render_pdf,
)


DESIGN_COLLECTION_KEYS = ("struts", "edges")
POINT_COLLECTION_KEYS = ("junctions", "vertices", "nodes")
CELL_COLLECTION_KEYS = ("unit_cells", "octet_cells", "octets", "cells")
ID_KEYS = ("id", "strut_id", "edge_id", "member_id")


class RegisteredDesignError(ValueError):
    """Raised when an explicit registered-design input is invalid."""


@dataclass(frozen=True)
class ExtractedDesignValue:
    value: Any
    source_field: str | None
    method: str
    confidence: str
    warnings: tuple[str, ...] = ()


@dataclass
class RegisteredDesignCandidate:
    path: Path
    score: int
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    specimen_values: tuple[str, ...] = ()
    is_lfs_pointer: bool = False


@dataclass
class RegisteredDesignSummary:
    specimen_id: str | None
    registered_json_path: Path | None
    lattice_type: str | None
    total_struts: int | None
    total_junctions: int | None
    total_vertices: int | None
    total_edges: int | None
    total_octet_cells: int | None
    unit_cells_x: int | None
    unit_cells_y: int | None
    unit_cells_z: int | None
    coordinate_bounds: dict[str, tuple[float, float]] | None
    coordinate_system: str | None
    coordinate_units: str | None
    thickness_min: float | None
    thickness_max: float | None
    thickness_mean: float | None
    thickness_median: float | None
    unique_thickness_count: int | None
    registration_metadata: dict[str, Any]
    extraction_methods: dict[str, str]
    sources: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvaluationCoverage:
    registered_struts: int | None
    evaluated_struts: int
    coverage_percentage: float | None
    missing_evaluation_count: int | None
    unexpected_evaluation_count: int | None
    duplicate_registered_ids: int | None
    duplicate_evaluation_ids: int | None
    unmatched_registered_ids: list[str] = field(default_factory=list)
    unexpected_evaluation_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DefectResultsReportData:
    registered_design: RegisteredDesignSummary
    coverage: EvaluationCoverage
    whole_object_analysis: ReportAnalysis
    spatial_concentration_analysis: SpatialConcentrationAnalysis
    cube_region_analysis: CubeRegionAnalysis


def _na(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        first = path.open("r", encoding="utf-8", errors="replace").readline().strip()
    except OSError:
        return False
    return first == "version https://git-lfs.github.com/spec/v1"


def _load_design_json(path: Path) -> Any:
    if not path.is_file():
        raise RegisteredDesignError(f"Registered-design JSON does not exist: {path}")
    if path.suffix.lower() != ".json":
        raise RegisteredDesignError("Registered-design path must have a .json extension.")
    if _is_lfs_pointer(path):
        raise RegisteredDesignError(
            f"Registered-design file is a Git LFS pointer and its JSON content is not materialized: {path}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RegisteredDesignError(f"Invalid registered-design JSON in {path}: {exc}") from exc


def _collection(data: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, list[Any] | None]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return key, value
    return None, None


def inspect_registered_design_candidate(path: Path) -> RegisteredDesignCandidate:
    """Validate and score a potential registered graph by content."""
    resolved = Path(path).resolve()
    if _is_lfs_pointer(resolved):
        return RegisteredDesignCandidate(
            resolved, 0, False,
            warnings=["Git LFS pointer; registered JSON content is not materialized."],
            is_lfs_pointer=True,
        )
    try:
        data = _load_design_json(resolved)
    except RegisteredDesignError as exc:
        return RegisteredDesignCandidate(resolved, 0, False, warnings=[str(exc)])
    if not isinstance(data, Mapping):
        return RegisteredDesignCandidate(resolved, 0, False, warnings=["Top-level JSON is not an object."])
    score = 0
    reasons = []
    strut_key, struts = _collection(data, DESIGN_COLLECTION_KEYS)
    point_key, points = _collection(data, POINT_COLLECTION_KEYS)
    cell_key, cells = _collection(data, CELL_COLLECTION_KEYS)
    if struts is not None:
        geometric = sum(
            isinstance(item, Mapping)
            and any(key in item for key in ("junction0", "junction1", "p0", "p1", "thickness"))
            for item in struts[:1000]
        )
        classified = sum(
            isinstance(item, Mapping)
            and any(key in item for key in ("verdict", "classification", "defect_type", "predicted_class"))
            for item in struts[:1000]
        )
        if geometric:
            score += 45
            reasons.append(f"geometric `{strut_key}` collection (+45)")
        if classified and not geometric:
            score -= 50
            reasons.append("classification-only struts (-50)")
    if points is not None:
        score += 30
        reasons.append(f"`{point_key}` graph-point collection (+30)")
    if cells is not None:
        score += 25
        reasons.append(f"`{cell_key}` cell collection (+25)")
    if any(key in data for key in ("registration", "registration_metadata", "transform", "coordinate_system")):
        score += 5
        reasons.append("registration/coordinate metadata (+5)")
    specimen_values = tuple(
        str(data[key]) for key in ("specimen_id", "specimen", "name", "source")
        if isinstance(data.get(key), (str, int))
    )
    return RegisteredDesignCandidate(
        resolved, score, score >= 45 and (points is not None or cells is not None),
        reasons=reasons, specimen_values=specimen_values,
    )


def _numeric_position(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(component) for component in point) else None


def _point_bounds(data: Mapping[str, Any]) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
    for collection_key in POINT_COLLECTION_KEYS:
        collection = data.get(collection_key)
        if not isinstance(collection, list):
            continue
        points = []
        for record in collection:
            if isinstance(record, Mapping):
                point = next(
                    (_numeric_position(record.get(key)) for key in ("position", "coordinates", "point")
                     if _numeric_position(record.get(key)) is not None),
                    None,
                )
                if point is not None:
                    points.append(point)
        if points:
            return {
                axis: (min(point[index] for point in points), max(point[index] for point in points))
                for index, axis in enumerate(("x", "y", "z"))
            }, f"{collection_key}.*.position"
    struts = data.get("struts")
    if isinstance(struts, list):
        points = []
        for record in struts:
            if not isinstance(record, Mapping):
                continue
            for key in ("p0", "p1", "start", "end"):
                point = _numeric_position(record.get(key))
                if point is not None:
                    points.append(point)
        if points:
            return {
                axis: (min(point[index] for point in points), max(point[index] for point in points))
                for index, axis in enumerate(("x", "y", "z"))
            }, "struts endpoint coordinates"
    return None, None


def _cell_dimensions(
    data: Mapping[str, Any],
) -> tuple[tuple[int, int, int] | None, str | None, str]:
    explicit_sets = (
        ("unit_cells_x", "unit_cells_y", "unit_cells_z"),
        ("cells_x", "cells_y", "cells_z"),
        ("nx", "ny", "nz"),
    )
    for keys in explicit_sets:
        values = [data.get(key) for key in keys]
        if all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values):
            return tuple(values), ", ".join(keys), "explicit"
    for key in ("lattice_shape", "grid_shape", "dimensions"):
        value = data.get(key)
        if (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            and len(value) == 3
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
        ):
            return tuple(value), key, "explicit"
    cells = next((data.get(key) for key in CELL_COLLECTION_KEYS if isinstance(data.get(key), list)), None)
    if cells:
        indices = [
            tuple(int(value) for value in item["indices"])
            for item in cells
            if isinstance(item, Mapping)
            and isinstance(item.get("indices"), Sequence)
            and len(item["indices"]) == 3
            and all(isinstance(value, (int, float)) and float(value).is_integer() for value in item["indices"])
        ]
        if len(indices) == len(cells):
            dimensions = tuple(max(index[axis] for index in indices) + 1 for axis in range(3))
            if math.prod(dimensions) == len(cells):
                return dimensions, "unit_cells.*.indices", "inferred_from_indices"
    return None, None, "unavailable"


def extract_registered_design_summary(
    path: Path | None,
    data: Mapping[str, Any] | None,
    *,
    unavailable_warning: str | None = None,
) -> RegisteredDesignSummary:
    """Extract design metadata once for both report renderers."""
    if data is None:
        return RegisteredDesignSummary(
            None, path, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            {}, {}, {}, [unavailable_warning or "Registered design JSON was unavailable."],
        )
    methods: dict[str, str] = {}
    sources: dict[str, str] = {}
    strut_key, struts = _collection(data, DESIGN_COLLECTION_KEYS)
    point_key, points = _collection(data, POINT_COLLECTION_KEYS)
    cell_key, cells = _collection(data, CELL_COLLECTION_KEYS)
    total_struts = len(struts) if struts is not None else None
    if total_struts is not None:
        methods["total_struts"] = "counted"
        sources["total_struts"] = strut_key or ""
    total_junctions = (
        len(data["junctions"]) if isinstance(data.get("junctions"), list)
        else len(data["nodes"]) if isinstance(data.get("nodes"), list)
        else None
    )
    total_vertices = len(data["vertices"]) if isinstance(data.get("vertices"), list) else None
    total_edges = len(data["edges"]) if isinstance(data.get("edges"), list) else None
    dimensions, dimension_source, dimension_method = _cell_dimensions(data)
    total_cells = len(cells) if cells is not None else (
        math.prod(dimensions) if dimensions else None
    )
    if total_cells is None and struts:
        for identifier_key in ("unit_cell_id", "cell_id", "octet_id", "unit_cell_indices"):
            identifiers = []
            for record in struts:
                if not isinstance(record, Mapping) or identifier_key not in record:
                    continue
                value = record[identifier_key]
                identifiers.append(tuple(value) if isinstance(value, list) else value)
            if identifiers and len(identifiers) == len(struts):
                total_cells = len(set(identifiers))
                dimension_source = f"{strut_key}.*.{identifier_key}"
                dimension_method = "inferred_from_ids"
                break
    if dimensions is None and path is not None:
        match = re.search(r"(?:octet[_-]?)?(\d+)x(\d+)x(\d+)", path.stem.casefold())
        if match:
            dimensions = tuple(int(value) for value in match.groups())
            dimension_source = "validated specimen filename pattern"
            dimension_method = "inferred_from_filename"
            if total_cells is None:
                total_cells = math.prod(dimensions)
    if total_cells is not None:
        methods["total_octet_cells"] = "counted" if cells is not None else dimension_method
        sources["total_octet_cells"] = cell_key or dimension_source or ""
    bounds, bounds_source = _point_bounds(data)
    if bounds:
        methods["coordinate_bounds"] = "calculated"
        sources["coordinate_bounds"] = bounds_source or ""
    thicknesses = []
    invalid_thickness = 0
    if struts:
        for record in struts:
            value = record.get("thickness") if isinstance(record, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                thicknesses.append(float(value))
            elif value is not None:
                invalid_thickness += 1
    warnings = []
    if invalid_thickness:
        warnings.append(f"{invalid_thickness} invalid thickness value(s) were ignored.")
    if total_cells is None:
        warnings.append(
            "The registered JSON does not contain an explicit or reliably inferable octet-cell count."
        )
    metadata = {
        key: value for key, value in data.items()
        if key in ("registration", "registration_metadata", "transform")
        and isinstance(value, (Mapping, list, str, int, float, bool))
    }
    specimen = next(
        (str(data[key]) for key in ("specimen_id", "specimen", "name") if isinstance(data.get(key), (str, int))),
        path.stem if path else None,
    )
    lattice_type = next(
        (str(data[key]) for key in ("lattice_type", "type") if isinstance(data.get(key), str)),
        "octet unit-cell graph" if cells is not None else None,
    )
    units = next(
        (str(data[key]) for key in ("coordinate_units", "units") if isinstance(data.get(key), str)),
        None,
    )
    coordinate_system = str(data["coordinate_system"]) if isinstance(data.get("coordinate_system"), str) else None
    return RegisteredDesignSummary(
        specimen_id=specimen,
        registered_json_path=path,
        lattice_type=lattice_type,
        total_struts=total_struts,
        total_junctions=total_junctions,
        total_vertices=total_vertices,
        total_edges=total_edges,
        total_octet_cells=total_cells,
        unit_cells_x=dimensions[0] if dimensions else None,
        unit_cells_y=dimensions[1] if dimensions else None,
        unit_cells_z=dimensions[2] if dimensions else None,
        coordinate_bounds=bounds,
        coordinate_system=coordinate_system,
        coordinate_units=units,
        thickness_min=min(thicknesses) if thicknesses else None,
        thickness_max=max(thicknesses) if thicknesses else None,
        thickness_mean=fmean(thicknesses) if thicknesses else None,
        thickness_median=median(thicknesses) if thicknesses else None,
        unique_thickness_count=len(set(thicknesses)) if thicknesses else None,
        registration_metadata=metadata,
        extraction_methods=methods,
        sources=sources,
        warnings=warnings,
    )


def _ids(records: Sequence[Any]) -> tuple[set[str] | None, int]:
    values = []
    missing = False
    for record in records:
        if not isinstance(record, Mapping):
            missing = True
            continue
        found = next((str(record[key]) for key in ID_KEYS if key in record), None)
        if found is None:
            missing = True
        else:
            values.append(found)
    duplicates = len(values) - len(set(values))
    return (None if missing else set(values)), duplicates


def calculate_evaluation_coverage(
    design_data: Mapping[str, Any] | None,
    defect_data: Any,
    design_summary: RegisteredDesignSummary,
) -> EvaluationCoverage:
    """Compare reliable strut IDs; never infer ID-level coverage without IDs."""
    defect_records, _ = _find_records(defect_data)
    evaluated = len(defect_records or [])
    warnings = []
    if design_data is None or design_summary.total_struts is None:
        return EvaluationCoverage(
            design_summary.total_struts, evaluated, None, None, None, None, None,
            warnings=["Coverage is unavailable because registered design records were unavailable."],
        )
    _, design_records = _collection(design_data, DESIGN_COLLECTION_KEYS)
    design_ids, design_dupes = _ids(design_records or [])
    evaluation_ids, evaluation_dupes = _ids(defect_records or [])
    if design_ids is None or evaluation_ids is None:
        warnings.append(
            "ID-level coverage is unavailable because one or both strut collections lack reliable identifiers."
        )
        return EvaluationCoverage(
            design_summary.total_struts, evaluated, None, None, None,
            design_dupes, evaluation_dupes, warnings=warnings,
        )
    missing = sorted(design_ids - evaluation_ids)
    unexpected = sorted(evaluation_ids - design_ids)
    matched = len(design_ids & evaluation_ids)
    coverage = matched / len(design_ids) * 100 if design_ids else None
    return EvaluationCoverage(
        design_summary.total_struts, evaluated, coverage, len(missing), len(unexpected),
        design_dupes, evaluation_dupes, missing[:20], unexpected[:20], warnings,
    )


def _design_feature_rows(summary: RegisteredDesignSummary) -> list[tuple[str, str, str]]:
    dimensions = (
        f"{summary.unit_cells_x} × {summary.unit_cells_y} × {summary.unit_cells_z}"
        if None not in (summary.unit_cells_x, summary.unit_cells_y, summary.unit_cells_z)
        else "N/A"
    )
    bounds = (
        "; ".join(f"{axis}={low:g} to {high:g}" for axis, (low, high) in summary.coordinate_bounds.items())
        if summary.coordinate_bounds else "N/A"
    )
    values = [
        ("Specimen ID", _na(summary.specimen_id), "metadata/path"),
        ("Registered JSON filename", summary.registered_json_path.name if summary.registered_json_path else "N/A", "discovered/explicit"),
        ("Registered JSON path", _na(summary.registered_json_path), "discovered/explicit"),
        ("Lattice type", _na(summary.lattice_type), "metadata/schema"),
        ("Total struts", _na(summary.total_struts), summary.sources.get("total_struts", "unavailable")),
        ("Total junctions", _na(summary.total_junctions), "junctions"),
        ("Total vertices", _na(summary.total_vertices), "vertices"),
        ("Total edges", _na(summary.total_edges), "edges"),
        ("Total octet lattice cells", _na(summary.total_octet_cells), summary.sources.get("total_octet_cells", "unavailable")),
        ("Unit-cell dimensions", dimensions, "structured indices/metadata"),
        ("Coordinate bounds", bounds, summary.sources.get("coordinate_bounds", "unavailable")),
        ("Coordinate system", _na(summary.coordinate_system), "metadata"),
        ("Coordinate units", _na(summary.coordinate_units), "metadata"),
        ("Minimum strut thickness", _na(summary.thickness_min), "calculated"),
        ("Maximum strut thickness", _na(summary.thickness_max), "calculated"),
        ("Average strut thickness", _na(summary.thickness_mean), "calculated"),
        ("Median strut thickness", _na(summary.thickness_median), "calculated"),
        ("Unique thickness values", _na(summary.unique_thickness_count), "calculated"),
    ]
    return [row for row in values if row[1] != "N/A"]


def _coverage_rows(coverage: EvaluationCoverage) -> list[tuple[str, str]]:
    """Return only coverage metrics supported by the current inputs."""
    values = [
        ("Registered struts", _na(coverage.registered_struts)),
        ("Evaluated struts", str(coverage.evaluated_struts)),
        (
            "Coverage",
            "N/A" if coverage.coverage_percentage is None
            else f"{coverage.coverage_percentage:.2f}%",
        ),
        ("Missing evaluation records", _na(coverage.missing_evaluation_count)),
        ("Unexpected evaluation records", _na(coverage.unexpected_evaluation_count)),
        ("Duplicate registered IDs", _na(coverage.duplicate_registered_ids)),
        ("Duplicate evaluation IDs", _na(coverage.duplicate_evaluation_ids)),
    ]
    return [row for row in values if row[1] != "N/A"]


def render_registered_design_markdown(
    summary: RegisteredDesignSummary,
    coverage: EvaluationCoverage,
) -> str:
    rows = "\n".join(
        f"| {feature} | {value} | {source} |"
        for feature, value, source in _design_feature_rows(summary)
    )
    coverage_rows = "\n".join(
        f"| {metric} | {value} |" for metric, value in _coverage_rows(coverage)
    )
    warnings = "\n".join(f"- {warning}" for warning in (*summary.warnings, *coverage.warnings))
    return f"""## Registered Design Summary

| Feature | Value | Source |
|---|---:|---|
{rows}

### Evaluation Coverage

| Metric | Value |
|---|---:|
{coverage_rows}

{warnings}

"""


def build_registered_design_pdf_story(
    summary: RegisteredDesignSummary,
    coverage: EvaluationCoverage,
) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    story: list[Any] = [Paragraph("Registered Design Summary", styles["Heading2"])]
    rows: list[list[Any]] = [["Feature", "Value", "Source"]]
    rows.extend([
        [
            Paragraph(escape(feature), styles["BodyText"]),
            Paragraph(escape(value), styles["BodyText"]),
            Paragraph(escape(source), styles["BodyText"]),
        ]
        for feature, value, source in _design_feature_rows(summary)
    ])
    design_table = Table(rows, repeatRows=1, colWidths=[1.55*inch, 3.2*inch, 1.3*inch])
    design_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([design_table, Spacer(1, 8), Paragraph("Evaluation Coverage", styles["Heading3"])])
    coverage_rows = [["Metric", "Value"], *[
        [metric, value] for metric, value in _coverage_rows(coverage)
    ]]
    coverage_table = Table(coverage_rows, repeatRows=1, colWidths=[2.7*inch, 2.0*inch])
    coverage_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E3F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([coverage_table, Spacer(1, 8)])
    for warning in (*summary.warnings, *coverage.warnings):
        story.append(Paragraph(f"- {escape(warning)}", styles["BodyText"]))
    return story


class DefectResultsReportingAgent(DefectResultsReportingAgentV3):
    """Canonical agent combining registered design and all V3 analyses."""

    name = "Defect Results Reporting Agent"

    def __init__(
        self,
        config: ReportingConfig | None = None,
        spatial_config: Any | None = None,
        region_config: CubeRegionAnalysisConfig | None = None,
    ) -> None:
        super().__init__(config, spatial_config, region_config)
        self.last_registered_design_summary: RegisteredDesignSummary | None = None
        self.last_coverage: EvaluationCoverage | None = None
        self.last_registered_json_path: Path | None = None
        self.last_report_data: DefectResultsReportData | None = None

    def _provenance_path(self, defect_data: Any, defect_path: Path) -> Path | None:
        if isinstance(defect_data, Mapping):
            for container in (defect_data, defect_data.get("meta")):
                if not isinstance(container, Mapping):
                    continue
                for key in ("registered_json_path", "design_json_path", "registered_design", "design_json"):
                    value = container.get(key)
                    if isinstance(value, str):
                        path = Path(value)
                        return (defect_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        return None

    def discover_registered_json(
        self,
        defect_results_path: Path,
        defect_data: Any,
        *,
        search_root: Path | None = None,
        specimen_id: str | None = None,
    ) -> Path | None:
        """Locate the associated registered graph using deterministic evidence."""
        provenance = self._provenance_path(defect_data, defect_results_path)
        if provenance and provenance.exists():
            return provenance
        root = Path(search_root).resolve() if search_root else Path(__file__).resolve().parents[3]
        candidates = []
        for path in root.rglob("*.json"):
            if any(part.lower() in {".git", ".venv", "venv", "node_modules", "__pycache__"} for part in path.parts):
                continue
            candidate = inspect_registered_design_candidate(path)
            if candidate.valid:
                candidates.append(candidate)
        if not candidates:
            return None
        query = (specimen_id or defect_results_path.stem).casefold()
        dimensions = re.findall(r"\d+x\d+x\d+", query)
        def rank(candidate: RegisteredDesignCandidate) -> tuple[int, int, float]:
            text = (candidate.path.stem + " " + " ".join(candidate.specimen_values)).casefold()
            match = int(query in text or any(value in text for value in dimensions))
            return match, candidate.score, candidate.path.stat().st_mtime
        ranked = sorted(candidates, key=rank, reverse=True)
        if ranked and rank(ranked[0])[0]:
            return ranked[0].path
        # The pipeline's configured path is stronger than an unrelated schema match.
        try:
            from analysis.defect_detection.config import DESIGN_JSON
            configured = Path(DESIGN_JSON).resolve()
            if configured.exists():
                return configured
        except Exception:
            pass
        return None

    def run(
        self,
        input_path: Path | None = None,
        *,
        registered_json_path: Path | None = None,
        search_root: Path | None = None,
        specimen_id: str | None = None,
        output_dir: Path | None = None,
        markdown_output_path: Path | None = None,
        pdf_output_path: Path | None = None,
    ) -> GeneratedReportPaths:
        selected = self._resolve_input(input_path, search_root, specimen_id)
        defect_data = json.loads(selected.read_text(encoding="utf-8-sig"))
        explicit_design = registered_json_path is not None
        design_path = (
            Path(registered_json_path).resolve()
            if explicit_design
            else self.discover_registered_json(
                selected, defect_data, search_root=search_root, specimen_id=specimen_id
            )
        )
        design_data: Mapping[str, Any] | None = None
        unavailable_warning = None
        if design_path is not None:
            try:
                loaded = _load_design_json(design_path)
                candidate = inspect_registered_design_candidate(design_path)
                if not candidate.valid:
                    raise RegisteredDesignError(
                        f"File does not have a compatible registered-design graph schema: {design_path}"
                    )
                design_data = loaded
            except RegisteredDesignError as exc:
                if explicit_design:
                    raise
                unavailable_warning = str(exc)
        else:
            unavailable_warning = "No associated registered-design JSON could be located."

        summary = extract_registered_design_summary(
            design_path, design_data, unavailable_warning=unavailable_warning
        )
        defect_specimen = None
        if isinstance(defect_data, Mapping):
            for container in (defect_data, defect_data.get("meta")):
                if isinstance(container, Mapping):
                    defect_specimen = next(
                        (str(container[key]) for key in ("specimen_id", "specimen", "name")
                         if isinstance(container.get(key), (str, int))),
                        defect_specimen,
                    )
        if defect_specimen and summary.specimen_id:
            if defect_specimen.casefold() not in summary.specimen_id.casefold() and summary.specimen_id.casefold() not in defect_specimen.casefold():
                summary.warnings.append(
                    f"Specimen mismatch warning: defect results identify {defect_specimen!r}, "
                    f"while the registered design identifies {summary.specimen_id!r}."
                )
        elif design_path:
            defect_dims = re.findall(r"\d+x\d+x\d+", selected.stem.casefold())
            design_dims = re.findall(r"\d+x\d+x\d+", design_path.stem.casefold())
            if defect_dims and design_dims and defect_dims[0] != design_dims[0]:
                summary.warnings.append(
                    f"Specimen mismatch warning: filename dimensions differ "
                    f"({defect_dims[0]} versus {design_dims[0]})."
                )
        overall = self.inspect(selected)
        coverage = calculate_evaluation_coverage(design_data, defect_data, summary)
        spatial = analyze_spatial_concentration(defect_data, overall, self.config, self.spatial_config)
        cube = analyze_cube_regions(defect_data, overall, self.config, self.region_config)
        self.last_registered_json_path = design_path
        self.last_registered_design_summary = summary
        self.last_coverage = coverage
        self.last_spatial_analysis = spatial
        self.last_cube_analysis = cube
        self.last_report_data = DefectResultsReportData(summary, coverage, overall, spatial, cube)

        default_md = self.default_output_path(selected).with_name(
            re.sub(r"_v3$", "", self.default_output_path(selected).stem) + ".md"
        )
        base_dir = Path(output_dir) if output_dir is not None else selected.parent
        markdown = Path(markdown_output_path) if markdown_output_path else base_dir / default_md.name
        pdf = Path(pdf_output_path) if pdf_output_path else markdown.with_suffix(".pdf")
        if markdown.suffix.lower() != ".md":
            raise ValueError("Markdown output path must have a .md extension.")
        if pdf.suffix.lower() != ".pdf":
            raise ValueError("PDF output path must have a .pdf extension.")
        generated_at = datetime.now(timezone.utc)
        base_markdown = render_markdown(overall, generated_at=generated_at, config=self.config)
        design_markdown = render_registered_design_markdown(summary, coverage)
        marker = "## 3. Executive Summary"
        unified_markdown = (
            base_markdown.replace(marker, design_markdown + marker, 1)
            + render_spatial_markdown(spatial)
            + render_cube_markdown(cube, self.region_config)
        )
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(unified_markdown, encoding="utf-8")
        try:
            render_pdf(
                overall, pdf, generated_at=generated_at, config=self.config,
                leading_story=build_registered_design_pdf_story(summary, coverage),
                extra_story=build_spatial_pdf_story(spatial) + build_cube_pdf_story(cube, self.region_config),
            )
        except Exception as exc:
            raise PartialReportGenerationError(markdown, pdf, exc) from exc
        return GeneratedReportPaths(selected, markdown, pdf)
