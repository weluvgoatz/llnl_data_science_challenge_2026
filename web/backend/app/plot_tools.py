"""Plot-rendering tools for plot_agent.

Each function reads real numbers already computed by the pipeline (via
agent_tools' read-only aggregations, or directly from a classification JSON)
and renders exactly those numbers to a PNG -- no plot here shows a value that
wasn't already produced by the deterministic pipeline. Colors match the
verdict palette used everywhere else in the app (frontend/src/defects.ts) so
a chart and the 3D viewer read as one system.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Mirror app/workflow.py's matplotlib setup so this module behaves the same
# whether it's imported before or after workflow.py in the same process.
_MATPLOTLIB_CONFIG = Path(__file__).resolve().parents[2] / ".matplotlib"
_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import agent_tools as tools
from .store import job_dir, load_job, update_job

# Matches frontend/src/defects.ts VERDICT_COLORS exactly, so a chart and the
# 3D viewer never disagree on what a color means.
VERDICT_COLORS = {
    "present": "#78B4FF",
    "missing": "#FF4141",
    "bent": "#F546F0",
    "thin": "#FFE12D",
    "disconnected": "#FF8C1E",
}
VERDICT_ORDER = ["missing", "disconnected", "bent", "thin", "present"]


def _plots_dir(job_id: str) -> Path:
    out = job_dir(job_id) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _register_artifact(job_id: str, path: Path, caption: str) -> dict[str, Any]:
    root = job_dir(job_id)
    relative = str(path.relative_to(root))
    artifact = {
        "id": path.stem,
        "name": path.name,
        "caption": caption,
        "path": relative,
        "mediaType": "image/png",
    }

    def apply(current: dict[str, Any]) -> None:
        artifacts = [a for a in current.get("artifacts", []) if a["id"] != artifact["id"]]
        artifacts.append(artifact)
        current["artifacts"] = artifacts

    update_job(job_id, apply)
    return artifact


def plot_verdict_counts(job_id: str, version_id: int | None = None) -> dict:
    """Bar chart of strut counts by verdict for one classification version."""
    job = load_job(job_id)
    data = tools.load_classification(job, version_id)
    counts = data["meta"]["counts"]
    n = data["meta"]["n"]
    labels = [v for v in VERDICT_ORDER if v in counts]
    values = [counts[v] for v in labels]
    colors = [VERDICT_COLORS[v] for v in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:,}\n({100 * value / n:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Struts")
    ax.set_title(f"Strut verdicts (n={n:,})")
    fig.tight_layout()

    path = _plots_dir(job_id) / f"verdict_counts_v{version_id or 'active'}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    artifact = _register_artifact(job_id, path, "Strut verdict counts")
    return {"source": "classification JSON meta.counts", "path": str(path), "artifact": artifact}


def plot_hotspots_map(job_id: str, top_n: int = 15, version_id: int | None = None) -> dict:
    """Horizontal bar chart of the unit cells with the highest defect rate."""
    hotspots = tools.defect_hotspots(job_id, top_n=top_n, version_id=version_id)
    cells = hotspots["top_cells"]
    if not cells:
        raise tools.ToolError("No unit cells with struts to plot.")
    labels = [f"cell {c['unit_cell_id']} {tuple(c['grid_indices'] or [])}" for c in cells][::-1]
    fractions = [100 * c["defect_fraction"] for c in cells][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cells))), dpi=130)
    ax.barh(labels, fractions, color="#FF8C1E")
    ax.set_xlabel("Defect rate in cell (%)")
    ax.set_title(f"Top {len(cells)} unit cells by defect rate")
    fig.tight_layout()

    path = _plots_dir(job_id) / f"hotspots_v{version_id or 'active'}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    artifact = _register_artifact(job_id, path, "Defect hotspots by unit cell")
    return {
        "source": "design JSON unit_cells grouping + classification verdicts",
        "path": str(path),
        "artifact": artifact,
        "top_cells": cells,
    }


def plot_thickness_distribution(job_id: str, verdict: str | None = None, version_id: int | None = None) -> dict:
    """Histogram of measured as-built radius, with the nominal radius marked."""
    job = load_job(job_id)
    data = tools.load_classification(job, version_id)
    pool = [s for s in data["struts"] if verdict is None or s["verdict"] == verdict]
    radii = [
        s["evidence"]["measured_radius_um"]
        for s in pool
        if s.get("evidence", {}).get("measured_radius_um") is not None
    ]
    if not radii:
        raise tools.ToolError(f"No measured radii available for verdict={verdict!r}.")
    nominal = next(
        (s["evidence"]["nominal_radius_um"] for s in pool if s.get("evidence", {}).get("nominal_radius_um") is not None),
        None,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    color = VERDICT_COLORS.get(verdict, "#78B4FF")
    ax.hist(radii, bins=40, color=color, edgecolor="white", linewidth=0.3)
    if nominal is not None:
        ax.axvline(nominal, color="black", linestyle="--", linewidth=1.2,
                    label=f"nominal radius ({nominal:.0f} µm)")
        ax.legend()
    ax.set_xlabel("Measured as-built radius (µm)")
    ax.set_ylabel("Strut count")
    title = f"Measured radius distribution" + (f" — verdict={verdict}" if verdict else " — all struts")
    ax.set_title(f"{title} (n={len(radii):,})")
    fig.tight_layout()

    path = _plots_dir(job_id) / f"thickness_{verdict or 'all'}_v{version_id or 'active'}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    artifact = _register_artifact(job_id, path, "Measured vs. nominal strut radius")
    return {
        "source": "classification JSON evidence.measured_radius_um",
        "note": tools.THICKNESS_NOTE,
        "path": str(path),
        "artifact": artifact,
        "n": len(radii),
        "nominal_radius_um": nominal,
    }


def plot_version_comparison(job_id: str) -> dict:
    """Grouped bar chart comparing verdict counts across every rerun version."""
    job = load_job(job_id)
    versions = (job.get("defects") or {}).get("versions") or []
    if len(versions) < 2:
        raise tools.ToolError("Need at least two classification versions (run rerun_classification first).")

    labels = VERDICT_ORDER
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    width = 0.8 / len(versions)
    x = np.arange(len(labels))
    for i, version in enumerate(versions):
        counts = version["counts"]
        values = [counts.get(v, 0) for v in labels]
        ax.bar(x + i * width, values, width=width, label=f"v{version['id']}: {version['label']}")
    ax.set_xticks(x + width * (len(versions) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Struts")
    ax.set_title("Verdict counts across reruns")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = _plots_dir(job_id) / "version_comparison.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    artifact = _register_artifact(job_id, path, "Verdict counts across reruns")
    return {
        "source": "job.defects.versions[*].counts",
        "path": str(path),
        "artifact": artifact,
        "versions": [{"id": v["id"], "label": v["label"], "params": v["params"], "counts": v["counts"]} for v in versions],
    }
