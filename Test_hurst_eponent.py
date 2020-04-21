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

nx = 101
ny = 101
nz = 101
n = [nx, ny, nz]
sx = 101
sy = 101
sz = 101
gx = sx/nx
gy = sy/ny
gz = sy/nz
hurst = 0.8

x = np.arange(0, sx, gx)
y = np.arange(0, sy, gy)
z = np.arange(0, sz, gz)
X, Y = np.meshgrid(x, y)

c = RFFS.fourier_synthesis((nx, ny, sz), (sx, sy, sz),
                           hurst, rms_height = 0.8)

hursts = np.zeros(nx)
for i in range(nx):
    physical_sizes = np.array([sy, sz])
    topo = Topography(c[i,:,:], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    hursts[i] = analysis.estimate_hurst()
    # print("given hurst: {}, hurst error: {:.2}".format(hurst,
    #                                                    ((hursts[i] - hurst)
    #                                                     / hurst)))
mean_hurst_x = np.mean(hursts)
print("\nX dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_x - hurst)
                   / hurst)))

hursts = np.zeros(ny)
for i in range(ny):
    physical_sizes = np.array([sx, sz])
    topo = Topography(c[:,i,:], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    hursts[i] = analysis.estimate_hurst()
mean_hurst_y = np.mean(hursts)
print("Y dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_y - hurst)
                   / hurst)))

hursts = np.zeros(nz)
for i in range(nz):
    physical_sizes = np.array([sx, sy])
    topo = Topography(c[:,:,i], physical_sizes, True)
    analysis = CharacterisePeriodicSurface(topo)
    hursts[i] = analysis.estimate_hurst()

mean_hurst_z = np.mean(hursts)
print("Z dir: given hurst: {}, hurst rel error: {:.3}%"
      .format(hurst,
              100*((mean_hurst_z - hurst)
                   / hurst)))
