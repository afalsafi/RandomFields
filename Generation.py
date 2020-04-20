#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Helper functions for the generation of random fractal surfaces
"""

import numpy as np
import scipy.stats as stats


def _irfft2(karr, rarr):
    """
    Inverse 2d real-to-complex FFT

    Parameters
    ----------
    karr : array_like
        Fourier-space representation
    rarr : array_like
        Real-space representation
    progress_callback : function(i, n)
        Function that is called to report progress.
    """
    ncolumns, nrows, n = karr.shape
    for i in range(ncolumns):
        karr[i, :, 0] = np.fft.ifft(karr[i, :, 0])
    for j in range(nrows):
        if rarr.shape[0] % 2 == 0:
            rarr[:, j] = np.fft.irfft(karr[:, j, 0])
        else:
            rarr[:, j] = np.fft.irfft(karr[:, j, 0], n=rarr.shape[0])



def _irfft3(karr, rarr):
    """
    Inverse 2d real-to-complex FFT

    Parameters
    ----------
    karr : array_like
        Fourier-space representation
    rarr : array_like
        Real-space representation
    progress_callback : function(i, n)
        Function that is called to report progress.
    """
    nx, ny, nz = karr.shape
    print(karr[0, ...].shape)
    for i in range(nx):
        karr[i, ...] = np.fft.ifft2(karr[i, ...])
    for j in range(ny):
        for k in range(nz):
            if rarr.shape[0] % 2 == 0:
                rarr[:, j, k] = np.fft.irfft(karr[:, j, k])
            else:
                rarr[:, j, k] = np.fft.irfft(karr[:, j, k], n=rarr.shape[0])


def self_affine_prefactor(dim, nb_grid_pts, physical_sizes, Hurst, rms_height=None,
                          rms_slope=None, short_cutoff=None, long_cutoff=None):
    """
    Compute prefactor :math:`C_0` for the power-spectrum density of an ideal
    self-affine topography given by

    .. math ::

        C(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.

    Note:
    In the 2D case:

    .. math ::

        h^2_{rms} = \frac{1}{2 \pi} \int_{0}^{\infty} q C^{iso}(q) dq

    whereas in the 1D case:

    .. math ::

        h^2_{rms} = \frac{1}{\pi} \int_{0}^{\infty} C^{1D}(q) dq

    See Equations (1) and (4) in [1].


    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map or the line scan.
    physical_sizes : array_like
        Physical physical_sizes of the topography map or the line scan.
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope of the topography map or the line scan.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.

    Returns
    -------
    prefactor : float
        Prefactor :math:`\sqrt{C_0}`

    References
    -----------
    [1]: Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    """
    nb_grid_pts = np.asarray(nb_grid_pts)
    physical_sizes = np.asarray(physical_sizes)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(nb_grid_pts / physical_sizes)

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = 2 * np.pi * np.max(1 / physical_sizes)

    area = np.prod(physical_sizes)

    if rms_height is not None:
        # Assuming no rolloff region
        fac = 2 * rms_height / np.sqrt(q_min ** (-2 * Hurst) -
                                       q_max ** (-2 * Hurst)) * np.sqrt(Hurst * np.pi)
    elif rms_slope is not None:
        fac = 2 * rms_slope / np.sqrt(q_max ** (2 - 2 * Hurst) -
                                      q_min ** (2 - 2 * Hurst)) * np.sqrt((1 - Hurst) * np.pi)
    else:
        raise ValueError('Neither rms height nor rms slope is defined!')

    if dim == 1:
        fac /= np.sqrt(2)

    return fac * np.prod(nb_grid_pts) / np.sqrt(area)


def fourier_synthesis(nb_grid_pts, physical_sizes, hurst,
                      rms_height=None, rms_slope=None, c0=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0,
                      amplitude_distribution=lambda n: np.random.normal(size=n),
                      phases_maker = lambda m: np.exp(2 * np.pi * np.random.rand(m) * 1j),
                      rfn=None, kfn=None):
    """
    Create a self-affine, randomly rough surface using a Fourier filtering
    algorithm. The algorithm is described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map.
    physical_sizes : array_like
        Physical physical_sizes of the topography map.
    hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    c0: float
        self affine prefactor :math:`C_0`:
        :math:`C(q) = C_0 q^{-2-2H}`
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.
    rolloff : float
        Value for the power-spectral density (PSD) below the long-wavelength
        cutoff. This multiplies the value at the cutoff, i.e. unit will give a
        PSD that is flat below the cutoff, zero will give a PSD that is vanishes
        below cutoff. (Default: 1.0)
    amplitude_distribution : function
        Function that generates the distribution of amplitudes.
        (Default: np.random.normal)
    rfn : str
        Name of file that stores the real-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    kfn : str
        Name of file that stores the Fourie-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    progress_callback : function(i, n)
        Function that is called to report progress.

    Returns
    -------
    topography : UniformTopography or UniformLineScan
        The topography.
    """
    dim = len(nb_grid_pts)
    max_dim = 3
    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts) / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    if c0 is None:
        fac = self_affine_prefactor(dim ,nb_grid_pts, physical_sizes, hurst, rms_height=rms_height,
                                    rms_slope=rms_slope, short_cutoff=short_cutoff,
                                    long_cutoff=long_cutoff)
    else:
        # prefactor for the fourier heights
        fac = np.sqrt(c0) * np.prod(nb_grid_pts) / np.sqrt(np.prod(physical_sizes))
        #                   ^                       ^ C(q) = c0 q^(-2-2H) = 1 / A |fh(q)|^2
        #                   |                         and h(x,y) = sum(1/A fh(q) e^(iqx)))
        #                   compensate for the np.fft normalisation

    n = np.ones(max_dim, dtype = int)
    s = np.ones(max_dim)
    n[0:dim:1] = nb_grid_pts
    s[0:dim:1] = physical_sizes
    # kshape: the shape of the fourier series coeffs considering the symmetry of real Fourier transform
    kshape = n
    kn = n[0] // 2 + 1 # SYMMETRY
    kshape[0] = kn

    rarr = np.empty(nb_grid_pts, dtype=np.float64)
    karr = np.empty(kshape, dtype=np.complex128)

    qx = 2 * np.pi * np.arange(kn) / s[0]
    for z in range(n[2]):
        if z > n[2] // 2:
            qz = 2 * np.pi * (n[2] - z) / s[2]
        else:
            qz = 2 * np.pi * z / s[2]

        for y in range(n[1]):
            if y > n[1] // 2:
                qy = 2 * np.pi * (n[1] - y) / s[1]
            else:
                qy = 2 * np.pi * y / s[1]
            q_sq = qz ** 2 + qy ** 2 + qx ** 2
            if z == 0 and y == 0:
                q_sq[0] = 1.
            # making phases and amplitudes of the wave funcrion with random generating functions
            # this functions could be passed to the function in the first place and you can see their deafult
            # functions in the signature of the function
            phase = phases_maker(kn)
            ran = fac * phase * amplitude_distribution(kn)
            karr[:, y, z] = ran * q_sq ** (-((dim * 0.5) + hurst) / 2)
            karr[q_sq > q_max ** 2, y, z] = 0.
            if q_min is not None:
                mask = q_sq < q_min ** 2
                karr[mask, y, z] = (rolloff * ran[mask] *
                                 q_min ** (-((dim * 0.5) + hurst)))
        for ix in [0, -1] if n[0] % 2 == 0 else [0]:
            # Enforce symmetry
            if n[1] % 2 == 0:
                karr[ix, 0, :] = np.real(karr[ix, 0, :])
                karr[ix, n[1] // 2, :] = np.real(karr[ix, n[1] // 2, :])
                karr[ix, 1:n[1] // 2, :] = karr[ix, -1:n[1] // 2:-1, :].conj()
            else:
                karr[ix, 0, :] = np.real(karr[ix, 0, :])
                karr[ix, 1:n[1] // 2 + 1, :] = karr[ix, -1:n[1] // 2:-1, :].conj()
        if dim == 3:
            for ix in [0, -1] if n[0] % 2 == 0 else [0]:
                if n[2] % 2 == 0:
                    karr[ix, :, 0] = np.real(karr[ix, :, 0])
                    karr[ix, :, n[2] // 2] = np.real(karr[ix, :, n[2] // 2])
                    karr[ix, :, 1:n[2] // 2] = karr[ix, :, -1:n[2] // 2:-1].conj()
                else:
                    karr[ix, :, 0] = np.real(karr[ix, :, 0])
                    karr[ix, :, 1:n[2] // 2 + 1] = karr[ix, :, -1:n[2] // 2:-1].conj()

    if dim == 3:
        _irfft3(karr, rarr)
    elif dim == 2:
        _irfft2(karr, rarr)
    else:
        karr[0] = np.real(karr[0])
        rarr = np.fft.irfft(karr.T)
    return rarr
