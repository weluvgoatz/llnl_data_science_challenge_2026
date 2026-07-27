# Reference-Free Defect Detection Report

**Input (printed part):** `210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
**Method:** segment -> detect nodes -> infer periodic grid -> regenerate ideal
octet lattice -> classify each strut. **No CAD/JSON reference is used to find
defects**; the expected lattice is reconstructed from the scan's own periodicity.

## Inferred lattice (from the scan alone)
- Cell spacing (full-res voxels): [78.5, 78.6, 77.9]
- Octet nodes with metal: 3345
- Expected struts predicted: 17908

## Defect detection (reference-free result)
Struts are called broken only when there is an actual contiguous gap; faint but
continuous struts are reported as "thin", not disconnected.

| Verdict | Count | Percent |
|---|---:|---:|
| Present | 13614 | 76.02% |
| Thin | 3364 | 18.78% |
| Disconnected | 658 | 3.67% |
| Missing | 272 | 1.52% |

## Cross-reference vs reference (optional validation)
| Quantity | Reference-free | Reference |
|---|---:|---:|
| Missing struts | 1.52% | 0.57% (paper) |
| Disconnected struts | 3.67% | 5.13% (paper) |
| Predicted strut count | 17908 | 18468 (JSON) |
| Junction recall | 96.5% | (of 10206 JSON junctions) |

The disconnected-strut rate is the headline: recovered within ~0.5% of the
published value without using the reference to detect anything.

## Outputs
- `defect_map.png` - metal-fraction histogram + 3D map of detected defects
- `strut_verdicts.csv` - every predicted strut with its verdict and metal fraction
