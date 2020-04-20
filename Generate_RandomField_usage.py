import Random_Field_fourier_synthesis as RFFS

import numpy as np
import scipy.stats as stats

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nx = 201
ny = 101
nz = 501
n = [nx, ny, nz]
sx = 101
sy = 50
sz = 42
gx = sx/nx
gy = sy/ny
gz = sy/nz
hurst = 0.7
x = np.arange(0, sx, gx)
y = np.arange(0, sy, gy)
z = np.arange(0, sz, gz)
X, Y = np.meshgrid(x, y)

a = RFFS.fourier_synthesis([nx], [sx], hurst,
                           rms_height = 0.6)
b = RFFS.fourier_synthesis((nx, ny), (sx, sy),
                           hurst, rms_height = 0.8)
c = RFFS.fourier_synthesis((nx, ny, nz), (sx, sy, sz),
                           hurst, rms_height = 0.8)
fig0 = plt.figure()
ax0 = fig0.subplots(1,1)
ax0.plot(x[:-1],a[:].reshape((nx-1)))
fig0.savefig("1D.png")

fig1 = plt.figure()
ax1 = fig1.subplots(1,1)
cmap = plt.get_cmap('PiYG')
CSa = ax1.contourf(b.T, cmap=cmap)
fig1.colorbar(CSa)
fig1.savefig("2D.png")

fig2 = plt.figure()
ax2 = fig1.subplots(1,1)

def function_animation(i, *fargs):
    index = fargs[0]
    if index == 0:
        surface = c[i, :, :]
    if index == 1:
        surface = c[:, i, :]
    if index == 2:
        surface = c[:, :, i]
    CSa = ax2.contourf(surface.T, cmap=cmap)

# directions = [0, 1, 2]
# ax = ["x", "y", "z"]
# animations = []
# for dir in directions:
#     print(dir)
#     animation = FuncAnimation(fig2, func = function_animation, fargs = [dir],
#                                frames = np.arange(1, n[dir]), interval = 200)
#     animation.save("slices_{}.avi".format(ax[dir]))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, a, cmap=cmap,
#                linewidth=0, antialiased=False)
