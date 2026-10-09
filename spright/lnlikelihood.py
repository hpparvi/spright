#  Spright
#  Copyright (C) 2022-2026 Hannu Parviainen.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from numba import njit, prange
from numpy import log, ones, isfinite, inf, zeros, atleast_2d

from spright.analytical_model import model


@njit(cache=True)
def lnlikelihood(theta, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    """Calculate the log likelihood for a single parameter vector ignoring the uncertainties.

    Parameters
    ----------
    theta : ndarray
        Parameter vector in the sampling-space parameterisation.
    densities : ndarray
        Planet bulk densities [g/cm^3] with a shape ``(n_planets,)``.
    radii : ndarray
        Planet radii [R_earth] with a shape ``(n_planets,)``.
    rr0, rdr, rx0, rdx, drocky
        Rocky-planet density table and its grid definition (see
        `spright.analytical_model.model`).
    wr0, wdr, wx0, wdx, dwater
        Water-world density table and its grid definition (see
        `spright.analytical_model.model`).

    Returns
    -------
    float
        Log likelihood, or ``inf`` if the log likelihood is not finite.
    """
    lnl = log(model(densities, radii, theta, ones(4),
                    rr0, rdr, rx0, rdx, drocky,
                    wr0, wdr, wx0, wdx, dwater).sum(0)).sum()
    return lnl if isfinite(lnl) else inf


@njit(cache=True)
def lnlikelihood_v(pvp, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    """Calculate the log likelihoods for a set of parameter vectors ignoring the uncertainties.

    Parameters
    ----------
    pvp : ndarray
        Parameter vectors in the sampling-space parameterisation with a shape
        ``(n_vectors, n_parameters)``.
    densities : ndarray
        Planet bulk densities [g/cm^3] with a shape ``(n_planets,)``.
    radii : ndarray
        Planet radii [R_earth] with a shape ``(n_planets,)``.
    rr0, rdr, rx0, rdx, drocky
        Rocky-planet density table and its grid definition (see
        `spright.analytical_model.model`).
    wr0, wdr, wx0, wdx, dwater
        Water-world density table and its grid definition (see
        `spright.analytical_model.model`).

    Returns
    -------
    ndarray
        Log likelihoods with a shape ``(n_vectors,)``, with the non-finite values replaced
        by ``inf``.
    """
    npv = pvp.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in range(npv):
        lnl[i] = log(model(densities, radii, pvp[i], cs,
                           rr0, rdr, rx0, rdx, drocky,
                           wr0, wdr, wx0, wdx, dwater).sum(0)).sum()
        lnl[i] = lnl[i] if isfinite(lnl[i]) else inf
    return lnl


@njit(parallel=True)
def lnlikelihood_sample(pv, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    """Calculate the log likelihood for a single parameter vector using measurement samples.

    Parameters
    ----------
    pv : ndarray
        Parameter vector in the sampling-space parameterisation.
    densities : ndarray
        Planet bulk density samples [g/cm^3] with a shape ``(n_samples, n_planets)``.
    radii : ndarray
        Planet radius samples [R_earth] with a shape ``(n_samples, n_planets)``.
    rr0, rdr, rx0, rdx, drocky
        Rocky-planet density table and its grid definition (see
        `spright.analytical_model.model`).
    wr0, wdr, wx0, wdx, dwater
        Water-world density table and its grid definition (see
        `spright.analytical_model.model`).

    Returns
    -------
    float
        Log likelihood, or ``-inf`` if the rocky transition start (``r1``) is larger than
        the puffy transition end (``r4``).

    Notes
    -----
    The likelihood of each planet is the model probability density averaged over the
    planet's radius and density samples, which marginalises over the measurement
    uncertainties. The log likelihood is the sum of the logarithms of these averages.
    The planets are evaluated in parallel.
    """
    nob = densities.shape[1]
    cs = ones(3)
    lnt = zeros(nob)
    if pv[0] > pv[1]:
        return -inf
    else:
        lnt[:] = 0
        for j in prange(nob):
            lnt[j] = log(model(densities[:, j], radii[:, j], pv, cs,
                               rr0, rdr, rx0, rdx, drocky,
                               wr0, wdr, wx0, wdx, dwater).sum(0).mean())
        return lnt.sum()


@njit(parallel=True)
def lnlikelihood_vp(pvp, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    """Calculate the log likelihoods for a set of parameter vectors using measurement samples.

    Parameters
    ----------
    pvp : ndarray
        Parameter vectors in the sampling-space parameterisation with a shape
        ``(n_vectors, n_parameters)``, or a single parameter vector.
    densities : ndarray
        Planet bulk density samples [g/cm^3] with a shape ``(n_samples, n_planets)``.
    radii : ndarray
        Planet radius samples [R_earth] with a shape ``(n_samples, n_planets)``.
    rr0, rdr, rx0, rdx, drocky
        Rocky-planet density table and its grid definition (see
        `spright.analytical_model.model`).
    wr0, wdr, wx0, wdx, dwater
        Water-world density table and its grid definition (see
        `spright.analytical_model.model`).

    Returns
    -------
    ndarray
        Log likelihoods with a shape ``(n_vectors,)``. The log likelihood is ``-inf`` for
        the parameter vectors where the rocky transition start (``r1``) is larger than
        the puffy transition end (``r4``).

    Notes
    -----
    The likelihood of each planet is the model probability density averaged over the
    planet's radius and density samples, which marginalises over the measurement
    uncertainties. The log likelihood is the sum of the logarithms of these averages.
    The parameter vectors are evaluated in parallel.
    """
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    nob = densities.shape[1]
    lnl = zeros(npv)
    cs = ones(3)
    for i in prange(npv):
        lnt = zeros(nob)
        if pvp[i, 0] > pvp[i, 1]:
            lnl[i] = -inf
        else:
            lnt[:] = 0
            for j in range(nob):
                lnt[j] = log(model(densities[:, j], radii[:, j], pvp[i], cs,
                                   rr0, rdr, rx0, rdx, drocky,
                                   wr0, wdr, wx0, wdx, dwater).sum(0).mean())
            lnl[i] = lnt.sum()
    return lnl
