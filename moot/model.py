import astropy.units as u

from math import gamma
from numba import njit, prange
from numpy import clip, sqrt, zeros_like, zeros, exp, log, ones, inf, pi, isfinite, linspace, meshgrid, ndarray
from numpy.random import normal, uniform
from scipy.interpolate import RegularGridInterpolator


@njit
def lerp(x, a, b):
    return clip((x - a)/(b - a), 0.0, 1.0)


@njit
def spdf(x, m, s, l):
    """Student's distribution PDF"""
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


@njit
def rocky_volume_density(v):
    return 12.2 + 0.989*exp(0.110*v) - 7.66*exp(-0.0722*v) - 0.704*v**(-0.414)


@njit
def rocky_radius_density(r):
    v = r**3
    return 12.2 + 0.989*exp(0.110*v) - 7.66*exp(-0.0722*v) - 0.704*v**(-0.414)


@njit
def model(rho, radius, theta, component):
    a1, a2, a3, a4 = theta[0:4]
    m1, m2, m3 = theta[4:7]
    s1, s2, s3 = theta[7:10]
    l1, l2, l3 = theta[10:13]
    dr = theta[13]

    x1 = lerp(radius, a1, a2)
    x2 = lerp(radius, a3, a4)
    m3 = m3 + (radius - a3)*dr

    r = rocky_radius_density(radius)

    c1 = component[0]*(1.0 - x1)*spdf(rho, m1*r, s1, l1)
    c2 = component[1]*x1*(1.0 - x2)*spdf(rho, m2*r, s2, l2)
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
        if pvp[i, 0] > pvp[i, 1] or pvp[i, 2] > pvp[i, 3] or pvp[i, 1] > pvp[i, 2]:
            lnl[i] = -inf
        else:
            lnt[:] = 0
            for j in range(ns):
                lnt[j] = log(model(densities[j], radii[j], pvp[i], cs)).sum()
            maxl = max(lnt)
            lnl[i] = maxl + log(exp(lnt - maxl).mean())
    return lnl


def invert_cdf(values, cdf, res):
    x = linspace(0, 1.0, res)
    y = zeros(res)
    y[0] = values[0]
    y[-1] = values[-1]
    i, j = 0, 0
    for j in range(res-2):
        while cdf[i] < x[j+1]:
            i += 1
        a = (x[j+1] - cdf[i-1]) / (cdf[i] - cdf[i-1])
        y[j+1] = (1-a)*values[i-1] + a*values[i]
    return x, y


def create_radius_density_map(pvs: ndarray,
                              rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                              rres: int = 200, dres: int = 100) -> (ndarray, ndarray, ndarray):
    radii = linspace(*rlims, num=rres)
    densities = linspace(*dlims, num=dres)
    dgrid, rgrid = meshgrid(densities, radii)
    m = average_model(pvs, dgrid.ravel(), rgrid.ravel()).reshape(rgrid.shape)
    return radii, densities, m


def create_radius_density_icdf(pvs: ndarray, pres: int = 100,
                               rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                               rres: int = 200, dres: int = 100) -> (ndarray, ndarray, ndarray):
    radii, densities, rdmap = create_radius_density_map(pvs, rlims, dlims, rres, dres)
    cdf = rdmap.cumsum(axis=1)
    cdf /= cdf[:, -1:]
    icdf = zeros((rres, pres))
    for i in range(rres):
        probs, icdf[i] = invert_cdf(densities, cdf[i], pres)
    return radii, densities, probs, rdmap, icdf


def sample_density(radius: tuple[float, float],
                   radii: ndarray, probs: ndarray, icdf: ndarray,
                   nsamples: int = 20_000) -> ndarray:
    rgi = RegularGridInterpolator((radii, probs), icdf, bounds_error=False)
    r = normal(radius[0], radius[1], nsamples)
    samples = rgi((r, uniform(size=r.size)))
    return samples[isfinite(samples)]


def sample_mass(radius: tuple[float, float],
                radii: ndarray, probs: ndarray, icdf: ndarray,
                nsamples: int = 20_000) -> ndarray:
    rgi = RegularGridInterpolator((radii, probs), icdf, bounds_error=False)
    r = normal(radius[0], radius[1], nsamples)
    v = 4/3 * pi * (r*u.R_earth).to(u.cm)**3
    m_g = v * rgi((r, uniform(size=r.size))) * (u.g / u.cm**3)
    samples = m_g.to(u.M_earth).value
    return samples[isfinite(samples)]

