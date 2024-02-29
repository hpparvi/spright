from math import gamma

from numba import njit, prange
from numpy import sqrt, pi, clip, where, isfinite, zeros_like, atleast_1d, zeros, ndarray

from spright.lerp import bilerp_vr


@njit(cache=True)
def map_pv(pv):
    pv_mapped = pv[:11].copy()
    r1 = pv_mapped[0] = pv[0]
    r4 = pv_mapped[3] = pv[1]
    d = r4 - r1
    w = pv[2]   # WW population width
    p = pv[3]   # WW population shape
    a = 0.5 - abs(w - 0.5)
    r2 = pv_mapped[1] = r1 + d * (1.0 - w + p * a)
    r3 = pv_mapped[2] = r1 + d * (w + p * a)
    pv_mapped[8:11] = 10 ** pv[8:11]
    return pv_mapped


@njit(cache=True)
def spdf(x, m, s, l):
    """Non-standardized Student's t-distribution PDF"""
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


def weights_full(x, y, x1, x2, x3, y1, y2, y3):
    w1 = ((y2-y3)*(x - x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w2 = ((y3-y1)*(x - x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w3 = 1. - w1 - w2
    return w1, w2, w3


@njit(cache=True)
def map_r_to_xy(r, a1, a2, b1, b2):
    """Maps the planet radius to the mixture triangle (x,y) coordinates."""
    db = max(b2-b1, 1e-4)
    x = clip((r-b1)/db, 0.0, 1.0)
    da = max(a2-a1, 1e-4)
    y = clip(clip((r-a1)/da, 0.0, 1.0) - x, 0.0, 1.0)
    return x, y


@njit(cache=True)
def mixture_weights(x, y):
    """Calculates the mixture weights using interpolation inside a triangle."""
    w1 = 1. - x - y
    w2 = y
    w3 = 1. - w1 - w2
    return w1, w2, w3


@njit(cache=True)
def model(density, radius, pv, component, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater) -> ndarray:
    density = atleast_1d(density)
    radius = atleast_1d(radius)
    pvm = map_pv(pv)
    model = zeros((3, density.size))

    rwstart, rwend, wpstart, wpend = pvm[0:4]
    crocky, cwater, mpuffy, dpuffy = pvm[4:8]
    srocky, swater, spuffy = pvm[8:]

    mrocky = bilerp_vr(radius, crocky, rr0, rdr, rx0, rdx, drocky)
    mwater = bilerp_vr(radius, cwater, wr0, wdr, wx0, wdx, dwater)
    mpuffy = mpuffy * radius**dpuffy / 2.0**dpuffy

    tx, ty = map_r_to_xy(radius, rwstart, rwend, wpstart, wpend)
    w1, w2, w3 = mixture_weights(tx, ty)

    procky = component[0] * w1 * spdf(density, mrocky, srocky, 5.0)
    pwater = component[1] * w2 * spdf(density, mwater, swater, 5.0)
    ppuffy = component[2] * w3 * spdf(density, mpuffy, spuffy, 5.0)

    model[0, :] = where(isfinite(procky), procky, 0.0)
    model[1, :] = where(isfinite(pwater), pwater, 0.0)
    model[2, :] = ppuffy
    return model


@njit(parallel=True)
def average_model(samples, density, radius, components, rr0, rdr, rx0, rdx, drocky,  wr0, wdr, wx0, wdx, dwater):
    npv = samples.shape[0]
    t = zeros((3, density.size))
    for i in prange(npv):
        t += model(density, radius, samples[i], components,
                   rr0, rdr, rx0, rdx, drocky,
                   wr0, wdr, wx0, wdx, dwater)
    return t/npv
