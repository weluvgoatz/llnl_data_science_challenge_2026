"""Single entry point for the strut defect detection subagent.

    python run_pipeline.py                     # detect + defect list  (the default)
    python run_pipeline.py --validate 20       # + validation panels per category
    python run_pipeline.py --viz               # + the 3-D colour-coded model
    python run_pipeline.py --samples           # + 50 individual samples per category
    python run_pipeline.py --all               # everything

Input is a raw CT `.tif` plus the registered design `.json`; segmentation and
skeletonisation happen here and cache themselves, so nothing else is needed.

The 3-D model is DELIBERATELY OPT-IN (`--viz`).  It is the slowest artefact and
is only wanted when somebody actually asks to see the lattice, so the default run
stays fast and its output stays small.  Once built it persists in v2/model3d/ and
can be served without rebuilding.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parent


def run(script, *args, label=None):
    t0 = time.time()
    name = label or script
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, str(V2 / script), *args], cwd=str(V2))
    if r.returncode != 0:
        print(f"!! {script} failed (exit {r.returncode})", flush=True)
        return False
    print(f"--- {name} done in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, metavar="N", default=0,
                    help="render N validation panels per defect category")
    ap.add_argument("--viz", action="store_true",
                    help="build the 3-D colour-coded model (slow; opt-in)")
    ap.add_argument("--samples", action="store_true",
                    help="export 50 individual sample PNGs per category")
    ap.add_argument("--all", action="store_true", help="everything")
    ap.add_argument("--skip-detect", action="store_true",
                    help="reuse the existing classification JSON")
    args = ap.parse_args()
    if args.all:
        args.viz = args.samples = True
        args.validate = args.validate or 20

    t0 = time.time()
    ok = True

    # 1. classify (segments + skeletonises on first run, then caches)
    if not args.skip_detect:
        ok &= run("detect_v2.py", label="detect: segment -> skeleton -> classify")

    # 2. the defect list downstream agents consume
    if ok:
        ok &= run("export_defect_list.py", label="export: defect list (JSON + CSV)")

    # 3. optional artefacts
    if ok and args.validate:
        for cat in ("missing", "disconnected", "thin", "bent"):
            run("validate_v2.py", cat, str(args.validate),
                label=f"validate: {cat} x{args.validate}")
    if ok and args.samples:
        run("export_samples.py", label="samples: 50 per category")
    if ok and args.viz:
        run("export_3d.py", label="visualise: 3-D colour-coded model")

    # summary
    cj = V2 / "strut_classification_v2.json"
    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s total) ===")
    if cj.exists():
        m = json.load(open(cj))["meta"]
        c, ic = m["counts"], m.get("counts_interior", {})
        n = m["n"]
        print(f"  {n} struts")
        for k in ("present", "missing", "disconnected", "thin", "bent"):
            print(f"    {k:13s} {c.get(k,0):6d}  ({100*c.get(k,0)/n:5.2f}%)"
                  f"   interior {ic.get(k,0)}")
        d = n - c.get("present", 0)
        print(f"    {'DEFECTIVE':13s} {d:6d}  ({100*d/n:5.2f}%)")
    for label, p in [("classification", cj),
                     ("defect list", V2 / "defect_list/defective_struts.json"),
                     ("defect CSV", V2 / "defect_list/defective_struts.csv"),
                     ("3-D model", V2 / "model3d/lattice_full.ply"),
                     ("3-D defects only", V2 / "model3d/lattice_defects_only.ply")]:
        if p.exists():
            print(f"  {label:18s} {p.relative_to(V2)}  "
                  f"({p.stat().st_size/1e6:.1f} MB)")
    if not (V2 / "model3d/lattice_full.ply").exists():
        print("  3-D model         not built (run with --viz when a visualisation is wanted)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
