"""Clean the as-built skeleton graph, then compare to the ground-truth design
to count MISSING and DEFECTIVE struts.

Cleaning (removes skeleton-extraction artifacts):
  1. merge node fragments (collapse the tangle at each real junction)
  2. dissolve degree-2 nodes (reconnect a strut that a false mid-node split)
  3. prune short dangling spurs
  4. drop residual sub-length artifact edges between junctions

Comparison:
  - register the ground-truth design to the as-built graph (similarity ICP)
  - match every designed strut to the nearest as-built strut
      no match           -> MISSING (no material there)
      matched + abnormal  -> DEFECTIVE (thin / low-density / broken)
      matched + normal    -> PRESENT
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from config import ROOT, ASBUILT_GRAPH as ASBUILT, DESIGN_JSON as GT, ASBUILT_GRAPH_CLEAN as OUT

D_MERGE = 20.0     # collapse node fragments within this radius (struts are ~55 apart)
SPUR_LEN = 30.0    # dangling edges shorter than this are spurs
MIN_STRUT = 30.0   # edges shorter than this between two junctions are artifacts
MATCH_TOL = 18.0   # designed strut matches an as-built strut within this midpoint dist
K = 3.0            # density/thickness outlier strength for "defective"


def robust(x):
    m = np.median(x); s = 1.4826 * np.median(np.abs(x - m))
    return m, (s if s > 0 else x.std() or 1.0)


def main():
    d = json.load(open(ASBUILT))
    N = len(d["junctions"])
    P = np.array([d["junctions"][i]["position"] for i in range(N)], float)  # [x,y,z]
    raw_edges = []
    for s in d["struts"]:
        a, b = s["junction0"], s["junction1"]
        raw_edges.append((a, b, float(s.get("thickness", 0)), float(s.get("mean_density", 0))))
    print(f"as-built (raw): {N} nodes, {len(raw_edges)} struts")

    # ---------- 1. merge node fragments ----------
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(P).query_pairs(D_MERGE):
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)
    gids = {r: i for i, r in enumerate(groups)}
    NP = np.array([P[idx].mean(0) for idx in groups.values()])
    remap = {i: gids[find(i)] for i in range(N)}

    # edges as dict {eid: [u,v,thick,dens]}; dedupe by node-pair (keep max length)
    edges = {}
    adj = defaultdict(set)
    pair_eid = {}
    eid = 0
    for a, b, th, de in raw_edges:
        u, v = remap[a], remap[b]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        L = float(np.linalg.norm(NP[u] - NP[v]))
        if key in pair_eid:
            continue  # dedupe: one strut per node pair
        e = pair_eid[key] = eid
        edges[e] = [key[0], key[1], L, th, de]
        adj[key[0]].add(e); adj[key[1]].add(e)
        eid += 1
    print(f"after node merge + dedupe: {len(NP)} nodes, {len(edges)} struts")

    # ---------- 2. dissolve degree-2 nodes (reconnect split struts) ----------
    def other(e, n):
        u, v = edges[e][0], edges[e][1]
        return v if u == n else u
    stack = [n for n in adj if len(adj[n]) == 2]
    while stack:
        n = stack.pop()
        if len(adj[n]) != 2:
            continue
        e1, e2 = list(adj[n])
        a, b = other(e1, n), other(e2, n)
        if a == b:                      # would be a self-loop; drop the pair
            for e in (e1, e2):
                edges.pop(e, None)
            adj[a].discard(e1); adj[a].discard(e2); adj[n].clear()
            continue
        L = edges[e1][2] + edges[e2][2]
        w1, w2 = edges[e1][2], edges[e2][2]
        th = (edges[e1][3] * w1 + edges[e2][3] * w2) / max(L, 1)
        de = (edges[e1][4] * w1 + edges[e2][4] * w2) / max(L, 1)
        ne = eid; eid += 1
        edges[ne] = [min(a, b), max(a, b), L, th, de]
        for e in (e1, e2):
            edges.pop(e, None)
        adj[a].discard(e1); adj[a].add(ne)
        adj[b].discard(e2); adj[b].add(ne)
        adj[n].clear()
        if len(adj[a]) == 2:
            stack.append(a)
        if len(adj[b]) == 2:
            stack.append(b)
    print(f"after dissolving deg-2 nodes: {len(edges)} struts")

    # ---------- 3. prune short spurs (dangling short edges) ----------
    changed = True
    while changed:
        changed = False
        deg = defaultdict(int)
        for e in edges.values():
            deg[e[0]] += 1; deg[e[1]] += 1
        for eidx in list(edges):
            u, v, L, th, de = edges[eidx]
            if L < SPUR_LEN and (deg[u] == 1 or deg[v] == 1):
                edges.pop(eidx); changed = True

    # ---------- 4. drop residual short artifact edges between two junctions ----------
    deg = defaultdict(int)
    for e in edges.values():
        deg[e[0]] += 1; deg[e[1]] += 1
    for eidx in list(edges):
        u, v, L, th, de = edges[eidx]
        if L < MIN_STRUT and deg[u] >= 3 and deg[v] >= 3:
            edges.pop(eidx)

    # rebuild node list (drop isolated)
    used = sorted({e[0] for e in edges.values()} | {e[1] for e in edges.values()})
    idmap = {o: i for i, o in enumerate(used)}
    nodes = NP[used]
    E = [[idmap[e[0]], idmap[e[1]], e[2], e[3], e[4]] for e in edges.values()]
    deg = defaultdict(int)
    for a, b, *_ in E:
        deg[a] += 1; deg[b] += 1
    dv = np.array([deg[i] for i in range(len(nodes))])
    Ls = np.array([e[2] for e in E])
    print(f"\nCLEANED as-built: {len(nodes)} nodes, {len(E)} struts")
    print(f"  median degree {int(np.median(dv))}  |  struts 45-70 vox: {int(((Ls>=45)&(Ls<70)).sum())}")

    # save cleaned as-built graph
    out = {"junctions": [{"id": i, "position": [float(x) for x in nodes[i]]}
                         for i in range(len(nodes))],
           "struts": [{"id": k, "junction0": a, "junction1": b,
                       "length": L, "thickness": th, "mean_density": de}
                      for k, (a, b, L, th, de) in enumerate(E)]}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"  saved cleaned graph -> {OUT.name}")

    # ================= COMPARE TO GROUND TRUTH =================
    gt = json.load(open(GT))
    GP = np.array([j["position"] for j in gt["junctions"]], float)     # [x,y,z]
    gt_struts = [(s["junction0"], s["junction1"]) for s in gt["struts"]]

    # register ground truth -> as-built (similarity: scale + translation, ICP)
    tree_nodes = cKDTree(nodes)
    s, t = 1.0, np.zeros(3)
    for _ in range(20):
        q = s * GP + t
        dd, idx = tree_nodes.query(q)
        m = dd < 40
        X, Y = GP[m], nodes[idx[m]]
        mx, my = X.mean(0), Y.mean(0)
        s = float(((Y - my) * (X - mx)).sum() / ((X - mx) ** 2).sum())
        t = my - s * mx
    print(f"\nregistered ground truth -> as-built: scale {s:.4f}")

    # midpoints
    gt_mid = s * np.array([(GP[a] + GP[b]) / 2 for a, b in gt_struts]) + t
    ab_mid = np.array([(nodes[a] + nodes[b]) / 2 for a, b, *_ in E])

    # classify as-built struts: defective if density/thickness low outlier or broken
    dens = np.array([e[4] for e in E]); thick = np.array([e[3] for e in E]); length = np.array([e[2] for e in E])
    dm, ds = robust(dens); tm, ts = robust(thick)
    ab_defective = (dens < dm - K * ds) | (thick < tm - K * ts) | (length < 42)

    # match each designed strut to nearest as-built strut
    ab_tree = cKDTree(ab_mid)
    dist, idx = ab_tree.query(gt_mid)
    matched = dist < MATCH_TOL

    n_missing = int((~matched).sum())
    matched_ids = idx[matched]
    n_defective = int(ab_defective[matched_ids].sum())
    n_present = int(matched.sum() - n_defective)
    total = len(gt_struts)

    print("\n" + "=" * 60)
    print("DESIGNED STRUTS CHECKED AGAINST THE AS-BUILT SCAN")
    print("=" * 60)
    print(f"  total designed struts : {total}")
    print(f"  PRESENT (healthy)     : {n_present:5d}  ({100*n_present/total:.2f}%)")
    print(f"  DEFECTIVE (thin/weak/broken): {n_defective:5d}  ({100*n_defective/total:.2f}%)")
    print(f"  MISSING (no material) : {n_missing:5d}  ({100*n_missing/total:.2f}%)")
    print(f"  -> total defects: {n_defective + n_missing} "
          f"({100*(n_defective+n_missing)/total:.2f}%)")
    print(f"\n  as-built real struts: {len(E)}  vs  designed: {total}  "
          f"(deficit {total-len(E)})")
    print(f"  (paper for this specimen: ~0.57% missing, ~5.13% disconnected)")


if __name__ == "__main__":
    main()
