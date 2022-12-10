import astropy.units as u

from math import gamma

from matplotlib.pyplot import subplots
from numba import njit, prange
from numpy import clip, sqrt, zeros_like, zeros, exp, log, ones, inf, pi, isfinite, linspace, meshgrid, ndarray, where, \
    nan, floor, nanmedian, atleast_2d
from numpy.random import normal, uniform
from scipy.interpolate import RegularGridInterpolator, interp1d

from .core import read_mr

@njit
def lerp(x, a, b):
    return clip((x - a)/(b - a), 0.0, 1.0)


@njit
def bilerp_s(r, c, r0, dr, c0, dc, data):
    nr = (r - r0) / dr
    ir = int(floor(nr))
    ar1 = nr - ir
    ar2 = 1.0 - ar1

    nc = (c - c0) / dc
    ic = int(floor(nc))
    ac1 = nc - ic
    ac2 = 1.0 - ac1

    if ic < 0 or ir < 0 or ic >= data.shape[0] - 1 or ir >= data.shape[1] - 1:
        return nan

    l00 = data[ic, ir]
    l01 = data[ic, ir + 1]
    l10 = data[ic + 1, ir]
    l11 = data[ic + 1, ir + 1]

    return (l00 * ac2 * ar2
            + l10 * ac1 * ar2
            + l01 * ac2 * ar1
            + l11 * ac1 * ar1)


@njit
def bilerp_vr(r, c, r0, dr, c0, dc, data):
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c, r0, dr, c0, dc, data)
    return d


@njit
def bilerp_vrvc(r, c, r0, dr, c0, dc, data):
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c[i], r0, dr, c0, dc, data)
    return d


@njit
def spdf(x, m, s, l):
    """Student's distribution PDF"""
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


@njit
def model(rho, radius, theta, component, r0, dr, drocky, dwater):
    rwstart, rwend, wpstart, wpend = theta[0:4]
    crocky, cwater, mpuffy = theta[4:7]
    srocky, swater, spuffy = theta[7:10]
    lrocky, lwater, lpuffy = theta[10:13]
    drdrpuffy = theta[13]

    x1 = lerp(radius, rwstart, rwend)
    x2 = lerp(radius, wpstart, wpend)

    mrocky = bilerp_vr(radius, crocky, r0, dr, 0.0, 0.05, drocky)
    mwater = bilerp_vr(radius, cwater, r0, dr, 0.05, 0.05, dwater)
    mpuffy = mpuffy + (radius - 2.2)*drdrpuffy

    procky = component[0] * (1.0 - x1) *              spdf(rho, mrocky, srocky, lrocky)
    pwater = component[1] *        x1  * (1.0 - x2) * spdf(rho, mwater, swater, lwater)
    ppuffy = component[2] *        x1  *        x2  * spdf(rho, mpuffy, spuffy, lpuffy)
    return where(isfinite(procky), procky, 1e-7) + where(isfinite(pwater), pwater, 1e-7) + ppuffy


@njit(parallel=True)
def average_model(samples, density, radius, components, r0, dr, drocky, dwater):
    npv = samples.shape[0]
    t = zeros_like(density)
    for i in prange(npv):
        t += model(density, radius, samples[i], components, r0, dr, drocky, dwater)
    return t/npv


@njit
def lnlikelihood(theta, densities, radii, r0, dr, drocky, dwater):
    lnl = log(model(densities, radii, theta, ones(4), r0, dr, drocky, dwater)).sum()
    return lnl if isfinite(lnl) else inf


@njit
def lnlikelihood_v(pvp, densities, radii, r0, dr, drocky, dwater):
    npv = pvp.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in range(npv):
        lnl[i] = log(model(densities, radii, pvp[i], cs, r0, dr, drocky, dwater)).sum()
        lnl[i] = lnl[i] if isfinite(lnl[i]) else inf
    return lnl


@njit(parallel=True)
def lnlikelihood_vp(pvp, densities, radii, r0, dr, drocky, dwater):
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
                lnt[j] = log(model(densities[j], radii[j], pvp[i], cs, r0, dr, drocky, dwater)).sum()
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


def create_radius_density_map(pvs: ndarray, r0, dr, drocky, dwater,
                              rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                              rres: int = 200, dres: int = 100, components = None) -> (ndarray, ndarray, ndarray):
    radii = linspace(*rlims, num=rres)
    densities = linspace(*dlims, num=dres)
    dgrid, rgrid = meshgrid(densities, radii)
    if components is None:
        components = ones(3)
    m = average_model(pvs, dgrid.ravel(), rgrid.ravel(), components, r0, dr, drocky, dwater).reshape(rgrid.shape)
    return radii, densities, m


def create_radius_density_icdf(pvs: ndarray, r0, dr, drocky, dwater, pres: int = 100,
                               rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                               rres: int = 200, dres: int = 100) -> (ndarray, ndarray, ndarray, ndarray, ndarray):
    radii, densities, rdmap = create_radius_density_map(pvs, r0, dr, drocky, dwater, rlims, dlims, rres, dres)
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


def model_means(pvp: ndarray, rdm, npt: int = 500, rmin: float = 0.5, rmax: float = 5.5, average: bool = True):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    radius = linspace(rmin, rmax, npt)
    models = {'rocky': zeros((2, npv, npt)), 'water': zeros((2, npv, npt)), 'puffy': zeros((2, npv, npt))}
    for i, pv in enumerate(pvp):
        models['rocky'][0, i] = where(radius < pv[1], rdm.evaluate_rocky(pv[4], radius), nan)
        models['rocky'][1, i] = where(radius < pv[0], rdm.evaluate_rocky(pv[4], radius), nan)
        models['water'][0, i] = where((radius >= pv[0]) & (radius <= pv[3]), rdm.evaluate_water(pv[5], radius), nan)
        models['water'][1, i] = where((radius >= pv[1]) & (radius <= pv[2]), rdm.evaluate_water(pv[5], radius), nan)
        models['puffy'][0, i] = where(radius > pv[2], pv[6] + (radius - 2.2) * pv[13], nan)
        models['puffy'][1, i] = where(radius > pv[3], pv[6] + (radius - 2.2) * pv[13], nan)

    if average:
        for k in models.keys():
            tmp = zeros((2, npt))
            for i in range(2):
                m = isfinite(models[k][i]).mean(0) < 0.5
                models[k][i, :, m] = nan
                tmp[i] = nanmedian(models[k][i], 0)
            models[k] = tmp

    return radius, models


def plot_model_means(pv: ndarray, rdm, plot_widths: bool = True,
                     npt: int = 500, dmin: float = 0.5, dmax: float = 5.5, ax=None):
    if ax is None:
        fig, ax = subplots()

    radius, models = model_means(pv, rdm, npt=npt, dmin=dmin, dmax=dmax)

    for j, (kind, model) in enumerate(models.items()):
        for i in range(2):
            ax.plot(radius, model[i], c=f"C{j}", alpha=(0.3, 0.8)[i % 2], ls=('--', '-')[i % 2])

    if plot_widths:
        sr, sw, sp = 10 ** pv[7:10]
        for i in (-1, 1):
            ax.plot(radius, models['rocky'][0] + i * sr, alpha=0.2, ls='--', c='C0')
            ax.plot(radius, models['water'][0] + i * sw, alpha=0.2, ls='--', c='C1')
            ax.plot(radius, models['puffy'][0] + i * sp, alpha=0.2, ls='--', c='C2')

    return ax
