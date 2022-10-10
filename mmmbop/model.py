from math import gamma
from numba import njit, prange
from numpy import clip, sqrt, zeros_like, zeros, exp, log, ones, inf, pi, isfinite


@njit
def lerp(x, a, b):
    return clip((x - a)/(b - a), 0.0, 1.0)


@njit
def spdf(x, m, s, l):
    """Student's distribution PDF"""
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


@njit
def model(rho, radius, theta, component):
    a1, a2, a3, a4 = theta[0:4]
    m1, m2, m3 = theta[4:7]
    s1, s2, s3 = theta[7:10]
    l1, l2, l3 = theta[10:13]
    dr = theta[13]

    x1 = lerp(radius, a1, a2)
    x2 = lerp(radius, a3, a4)
    m3 = m3 + (radius - a4)*dr

    c1 = component[0]*(1.0 - x1)*spdf(rho, m1, s1, l1)
    c2 = component[1]*x1*(1.0 - x2)*spdf(rho, m2, s2, l2)
    c3 = component[2]*x1*x2*spdf(rho, m3, s3, l3)
    return c1 + c2 + c3


@njit(parallel=True)
def average_model(samples, density, radius):
    npv = samples.shape[0]
    components = ones(3)
    t = zeros_like(density)
    for i in prange(npv):
        t += model(density, radius, samples[i], components)
    return t/npv


@njit
def lnlikelihood(theta, densities, radii):
    lnl = log(model(densities, radii, theta, ones(3))).sum()
    return lnl if isfinite(lnl) else inf


@njit
def lnlikelihood_v(pvp, densities, radii):
    npv = pvp.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in range(npv):
        lnl[i] = log(model(densities, radii, pvp[i], cs)).sum()
        lnl[i] = lnl[i] if isfinite(lnl[i]) else inf
    return lnl


@njit(parallel=True)
def lnlikelihood_vp(pvp, densities, radii):
    npv = pvp.shape[0]
    ns = densities.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in prange(npv):
        lnt = zeros(ns)
        if pvp[i, 0] > pvp[i, 1] or pvp[i, 2] > pvp[i, 3]:
            lnl[i] = -inf
        else:
            lnt[:] = 0
            for j in range(ns):
                lnt[j] = log(model(densities[j], radii[j], pvp[i], cs)).sum()
            maxl = max(lnt)
            lnl[i] = maxl + log(exp(lnt - maxl).mean())
    return lnl