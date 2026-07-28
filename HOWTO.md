# How to run the Strut Defect Detection subagent

Give it a lattice CT scan (`.tif`) + the registered design JSON → it finds every
**missing / broken / thin / bent** strut. Here's how to run it on your machine.

## 1. Get the code
```bash
git clone https://github.com/weluvgoatz/llnl_data_science_challenge_2026.git
cd llnl_data_science_challenge_2026
git checkout defect-detection-agentic-layer
```

## 2. Install the Python packages
```bash
pip install -r requirements.txt
```

## 3. Tell Codex about the tools (one time)
Open `~/.codex/config.toml` and add this block — use **your** absolute paths for
the Python executable and the repo:
```toml
[mcp_servers.lattice-defects]
command = "<PATH_TO_PYTHON_EXE>"
args = ["<PATH_TO_REPO>/src/defect_mcp_server.py"]
env = {}
```
Then **restart the Codex CLI**. Run `/mcp` to check the lattice-defect tools show up.

## 4. Run it
Start the Codex CLI **from the repo root** (so it finds the subagent + skills),
then just ask in plain English:
```
Use the strut_error_detection_agent to detect defects in <path/to/your_scan.tif>
```
It segments → skeletonizes → classifies, and writes the results.

> Prefer plain Python (no agent)? Just run:
> ```bash
> python analysis/defect_detection/unified_defects_accurate.py
> ```

## 5. Your results
A file `<scan>_unified_defects_accurate.json` — every strut labelled
`present / missing / bent / thin / disconnected`.
On our 9×9×9 octet lattice the result is **6.58% defective**
(present 93.42%, missing 2.22%, bent 1.98%, thin 1.55%, disconnected 0.83%).

There's a ready-made sample output to look at in
[`analysis/defect_detection/sample_output/`](analysis/defect_detection/sample_output/).

## Run it on a DIFFERENT scan (optional)
No code edits — just set these before running:
```bash
export LATTICE_BASE="my_scan"                 # file name, no extension
export LATTICE_STK="data/my_scan/tif_stacks"  # folder with the .tif
export LATTICE_DESIGN_JSON="data/my_scan/design_graph.json"
```

## Notes
- Restart the Codex CLI after editing `~/.codex/config.toml` or the skills — it
  doesn't reload them mid-session.
- Same scan in → same result out (the detector is deterministic).
- Built for octet-truss lattices like the challenge specimen.

More detail: [`AGENTS.md`](AGENTS.md) (architecture) · [`INTEGRATION.md`](INTEGRATION.md) (hand-off format).
