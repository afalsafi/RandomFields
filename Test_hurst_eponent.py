import Random_Field_fourier_synthesis as RFFS

from SurfaceAnalysis import CharacterisePeriodicSurface

from UniformLineScanAndTopography import Topography, UniformLineScan
from common import compute_wavevectors, ifftn

import numpy as np
import scipy.stats as stats

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = [301, 301, 301]
nx, ny, nz = n
s = [101, 101, 101]
sx, sy, sz = s
g = [sx/nx, sy/ny, sz/nz]
gx, gy, gz = g
hurst = 0.8
rms_height = 1.0

x = np.arange(0, sx, gx)
y = np.arange(0, sy, gy)
z = np.arange(0, sz, gz)
X, Y = np.meshgrid(x, y)

c = RFFS.fourier_synthesis((nx, ny, nz), (sx, sy, sz),
                           hurst, rms_height = rms_height)
# c = 2 * c
hursts = np.zeros(nx)
rmss = np.zeros(nx)
for i in range(nx):
    physical_sizes = np.array([sy, sz])
    topo = Topography(c[i,:,:], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    rmss[i] = np.sqrt(np.mean(c[i,:,:] ** 2))
    hursts[i] = analysis.estimate_hurst()

mean_hurst_x = np.mean(hursts)

print("\nX dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_x - hurst)
                   / hurst)))

mean_rms_x = np.mean(rmss)
print("X dir: given rmst: {}, hurst rel error: {:.3}%\n"
      .format(rms_height,
              100*((mean_rms_x - rms_height)
                   / hurst)))

hursts = np.zeros(ny)
rmss = np.zeros(ny)
for i in range(ny):
    physical_sizes = np.array([sx, sz])
    topo = Topography(c[:,i,:], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    rmss[i] = np.sqrt(np.mean(c[:,i,:] ** 2))
    hursts[i] = analysis.estimate_hurst()
mean_hurst_y = np.mean(hursts)
print("Y dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_y - hurst)
                   / hurst)))
mean_rms_y = np.mean(rmss)
print("Y dir: given rmst: {}, hurst rel error: {:.3}%\n"
      .format(rms_height,
              100*((mean_rms_y - rms_height)
                   / hurst)))

hursts = np.zeros(nz)
rmss = np.zeros(nz)
for i in range(nz):
    physical_sizes = np.array([sx, sy])
    topo = Topography(c[:,:,i], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    rmss[i] = np.sqrt(np.mean(c[:,:,i] ** 2))
    hursts[i] = analysis.estimate_hurst()

mean_hurst_z = np.mean(hursts)
print("Z dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_z - hurst)
                   / hurst)))
mean_rms_z = np.mean(rmss)
print("Z dir: given rmst: {}, hurst rel error: {:.3}%\n"
      .format(rms_height,
              100*((mean_rms_z - rms_height)
                   / hurst)))
