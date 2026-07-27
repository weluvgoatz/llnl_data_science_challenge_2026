import numpy as np

path = 'data/unitcell/unitcell.npy'
volume = np.load(path, mmap_mode='r')

print('Shape:', volume.shape)
print('Datatype:', volume.dtype)
print('Minimum:', float(volume.min()))
print('Maximum:', float(volume.max()))
print('Size in MiB:', volume.nbytes / 1024**2)