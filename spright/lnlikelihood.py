from numba import njit, prange
from numpy import log, ones, isfinite, inf, zeros, atleast_2d

from spright.analytical_model import model


@njit(cache=True)
def lnlikelihood(theta, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    lnl = log(model(densities, radii, theta, ones(4),
                    rr0, rdr, rx0, rdx, drocky,
                    wr0, wdr, wx0, wdx, dwater).sum(0)).sum()
    return lnl if isfinite(lnl) else inf


@njit(cache=True)
def lnlikelihood_v(pvp, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
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
