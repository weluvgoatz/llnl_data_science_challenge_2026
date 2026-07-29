"""Build and cache the exact as-built graph consumed by the unified classifier."""

import json

import numpy as np
import tifffile

from config import ASBUILT_GRAPH_CLEAN, CLASSIFIER_GRAPH, MASK
from unified_defects_accurate import build_asbuilt_graph


def main() -> None:
    mask = tifffile.imread(MASK) > 0
    nodes, edge_of, bow = build_asbuilt_graph(mask)
    pairs = np.asarray(sorted(edge_of), dtype=np.int64).reshape(-1, 2)
    bows = np.asarray([bow[tuple(pair)] for pair in pairs], dtype=np.float32)
    np.savez_compressed(CLASSIFIER_GRAPH, nodes=nodes, edges=pairs, bow=bows)

    graph = {
        "junctions": [
            {"id": index, "position": [float(value) for value in point]}
            for index, point in enumerate(nodes)
        ],
        "struts": [
            {
                "id": index,
                "junction0": int(pair[0]),
                "junction1": int(pair[1]),
                "length": float(np.linalg.norm(nodes[pair[0]] - nodes[pair[1]])),
                "bow": float(bows[index]),
            }
            for index, pair in enumerate(pairs)
        ],
        "meta": {"coordinate_order": "zyx", "source": "classifier topology cache"},
    }
    ASBUILT_GRAPH_CLEAN.write_text(json.dumps(graph), encoding="utf-8")
    print(
        f"Saved {CLASSIFIER_GRAPH.name} and {ASBUILT_GRAPH_CLEAN.name}: "
        f"{len(nodes):,} nodes, {len(pairs):,} struts"
    )


if __name__ == "__main__":
    main()
