import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "data", "unitcell", "unitcell.npy")

v = np.load(src)

print("shape:", v.shape)   # (256, 256, 256) → a 256×256×256 grid of voxels
print("dtype:", v.dtype)   # float32 → decimal numbers
print("range:", v.min(), "to", v.max())
print("mean:", v.mean())
print("unique values (up to 10):", np.unique(v)[:10])
print("\nsample block v[0, :5, :5]:")
print(v[0, :5, :5])
