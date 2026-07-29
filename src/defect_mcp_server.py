"""MCP server exposing the reference-free lattice defect-detection pipeline as
callable tools for Codex (or any MCP client).

Each heavy stage is wrapped as a tool that runs the corresponding, already-tested
script in ``analysis/defect_detection`` with the specimen selected by the
``base``/``stk_dir`` arguments (forwarded as LATTICE_BASE / LATTICE_STK env vars,
see ``analysis/defect_detection/config.py``). The lightweight ``summarize`` and
``describe_thresholds`` tools return results directly.

Run:  python src/defect_mcp_server.py     (stdio MCP server)

Pipeline order (each tool lists its prerequisites):
  segment -> skeletonize -> build_asbuilt_graph -> classify_lattice_defects
                                                 -> detect_bent_struts
                                                 -> render_defect_visualizations
                                                 -> export_defect_3d_models
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "analysis" / "defect_detection"

mcp = FastMCP("Lattice Defect Detection")

UM = 58.1
CATS = ("present", "missing", "bent", "thin", "disconnected")


def _run(
    script: str,
    base: str | None,
    stk_dir: str | None,
    timeout: int = 3600,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run a pipeline script in analysis/defect_detection with the chosen specimen."""
    path = DD / script
    if not path.is_file():
        return f"Error: script not found: {path}"
    env = dict(os.environ)
    if base:
        env["LATTICE_BASE"] = base
    if stk_dir:
        env["LATTICE_STK"] = stk_dir
    if extra_env:
        env.update(extra_env)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run([sys.executable, str(path)], cwd=str(DD), env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Error: {script} exceeded {timeout}s. Run it directly for long jobs."
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.returncode else "")
    return out.strip() or f"{script} finished (exit {proc.returncode}) with no output."


@mcp.tool()
def summarize_lattice_defects(classified_json_path: str) -> str:
    """Summarise a classification JSON: per-category counts and percentages.

    Args:
        classified_json_path: path to a ``*_unified_defects_accurate.json`` file
            (the output of classify_lattice_defects), whose ``struts`` each carry
            a ``verdict`` of present/missing/bent/thin/disconnected.
    Returns:
        A formatted table of counts + percentages and the total defect rate.
    """
    p = Path(classified_json_path)
    if not p.is_file():
        return f"Error: file not found: {p}"
    try:
        struts = json.loads(p.read_text(encoding="utf-8"))["struts"]
    except (OSError, ValueError, KeyError) as exc:
        return f"Error reading classification JSON: {exc}"
    n = len(struts)
    if n == 0:
        return "Error: no struts in file."
    counts = {c: 0 for c in CATS}
    for s in struts:
        counts[s.get("verdict", "present")] = counts.get(s.get("verdict", "present"), 0) + 1
    defective = n - counts["present"]
    lines = [f"Total designed struts: {n}", ""]
    for c in CATS:
        lines.append(f"  {c:13s}: {counts[c]:6d}   {100*counts[c]/n:6.2f}%")
    lines += ["", f"  {'DEFECTIVE':13s}: {defective:6d}   {100*defective/n:6.2f}%"]
    return "\n".join(lines)


@mcp.tool()
def describe_thresholds() -> str:
    """Return the exact defect-decision rules and where each threshold comes from."""
    r = 424.0 / 2 / UM
    return (
        "Reference-free defect rules (topology-first: is there a strut at all, then how is it):\n"
        f"  MISSING       metal along the strut < 15%  (or a joint sits on no metal).\n"
        f"                15% is just above the unavoidable joint-metal floor.\n"
        f"  DISCONNECTED  one continuous gap >= 25% of the strut length (a real break,\n"
        f"                not scattered thin spots). Anchored to strut length.\n"
        f"  THIN          strut density < median - 3*MAD of all struts (robust low\n"
        f"                outlier). Cutoff is computed from the specimen, not fixed.\n"
        f"  BENT          peak bow off the strut's own PCA axis > 1 strut radius\n"
        f"                = {r:.2f} vox = {r*UM:.0f} um. Physical scale; cross-checked vs median+3sigma.\n"
        f"  PRESENT       none of the above."
    )


@mcp.tool()
def build_asbuilt_graph(base: str = "", stk_dir: str = "") -> str:
    """Build the as-built lattice graph (nodes + struts) from the cached skeleton.

    Prerequisites in the stack dir: <base>_segmented_clean.tif and its
    <base>_segmented_clean.skelcoords.npz (from segmentation + skeletonization).
    Runs skel_to_json.py then clean_and_compare.py. Writes the cleaned graph JSON.
    """
    out = _run("skel_to_json.py", base or None, stk_dir or None)
    out += "\n\n=== cleanup ===\n" + _run("clean_and_compare.py", base or None, stk_dir or None)
    return out


@mcp.tool()
def classify_lattice_defects(
    base: str = "",
    stk_dir: str = "",
    prebuilt_graph_path: str = "",
) -> str:
    """Classify every designed strut as present/missing/bent/thin/disconnected.

    Metal-anchored, topology-first classifier (registers the design to the
    as-built graph, anchors each node to real metal, then applies the rules in
    describe_thresholds). Prerequisites: segmented mask, skeleton cache, and the
    design graph JSON. Writes <base>_unified_defects_accurate.json and prints the
    per-category counts. Use summarize_lattice_defects() on the output.
    """
    extra_env = (
        {"LATTICE_PREBUILT_GRAPH": prebuilt_graph_path}
        if prebuilt_graph_path
        else None
    )
    return _run(
        "unified_defects_accurate.py",
        base or None,
        stk_dir or None,
        extra_env=extra_env,
    )


@mcp.tool()
def detect_bent_struts(base: str = "", stk_dir: str = "") -> str:
    """Measure per-strut bow (max lateral deviation off the strut's PCA axis) and
    flag bent struts (> 1 radius = 212 um). Writes <base>_asbuilt_bent.json."""
    return _run("bent_struts.py", base or None, stk_dir or None)


@mcp.tool()
def render_defect_visualizations(
    base: str = "",
    stk_dir: str = "",
    output_dir: str = "",
) -> str:
    """Render the per-defect pipeline-validation figures and the zoomed defect
    atlas from the raw CT (requires a completed classification). Writes PNGs into
    analysis/defect_detection/."""
    extra_env = {"LATTICE_OUTPUT_DIR": output_dir} if output_dir else None
    out = _run(
        "viz_pipeline_perdefect.py",
        base or None,
        stk_dir or None,
        extra_env=extra_env,
    )
    out += "\n\n=== atlas ===\n" + _run(
        "viz_defect_atlas.py",
        base or None,
        stk_dir or None,
        extra_env=extra_env,
    )
    return out


@mcp.tool()
def export_defect_3d_models(base: str = "", stk_dir: str = "") -> str:
    """Export geometry-accurate coloured 3D models (PLY/STL): bent struts swept
    along their real curved centreline, thin struts at measured radius,
    disconnected struts with the real gap. Requires a completed classification."""
    return _run("export_realistic_models.py", base or None, stk_dir or None)


if __name__ == "__main__":
    mcp.run()
