from Random_Field_fourier_synthesis import fourier_synthesis
import numpy as np
n = 128
H = 0.74
rms_slope = 1.2
qs = 2 * np.pi / (0.4e-9)  # 1/m
s = 2e-6
topography = fourier_synthesis((n, n), (s, s),
                               H,
                               rms_slope=rms_slope,
                               long_cutoff=s / 4,
                               short_cutoff=4 * s / n)

realised_rms_heights = topography.rms_height()
print(realised_rms_heights)
