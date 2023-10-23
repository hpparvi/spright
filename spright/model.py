import astropy.units as u

from math import gamma

from matplotlib.pyplot import subplots
from numba import njit, prange
from numpy import clip, sqrt, zeros_like, zeros, log, ones, inf, pi, isfinite, linspace, meshgrid, ndarray, where, \
    nan, nanmedian, atleast_2d, newaxis, diff
from numpy.random import normal, uniform
from scipy.interpolate import RegularGridInterpolator

from .lerp import bilerp_vr
from .rdmodel import RadiusDensityModel


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
def model(rho, radius, pv, component, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    pvm = map_pv(pv)
    rwstart, rwend, wpstart, wpend = pvm[0:4]
    crocky, cwater, mpuffy, dpuffy = pvm[4:8]
    srocky, swater, spuffy = pvm[8:]

    mrocky = bilerp_vr(radius, crocky, rr0, rdr, rx0, rdx, drocky)
    mwater = bilerp_vr(radius, cwater, wr0, wdr, wx0, wdx, dwater)
    mpuffy = mpuffy * radius**dpuffy / 2.0**dpuffy

    tx, ty = map_r_to_xy(radius, rwstart, rwend, wpstart, wpend)
    w1, w2, w3 = mixture_weights(tx, ty)

    procky = component[0] * w1 * spdf(rho, mrocky, srocky, 5.0)
    pwater = component[1] * w2 * spdf(rho, mwater, swater, 5.0)
    ppuffy = component[2] * w3 * spdf(rho, mpuffy, spuffy, 5.0)
    return where(isfinite(procky), procky, 1e-7) + where(isfinite(pwater), pwater, 1e-7) + ppuffy


@njit(parallel=True)
def average_model(samples, density, radius, components, rr0, rdr, rx0, rdx, drocky,  wr0, wdr, wx0, wdx, dwater):
    npv = samples.shape[0]
    t = zeros_like(density)
    for i in prange(npv):
        t += model(density, radius, samples[i], components,
                   rr0, rdr, rx0, rdx, drocky,
                   wr0, wdr, wx0, wdx, dwater)
    return t/npv


@njit(cache=True)
def lnlikelihood(theta, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    lnl = log(model(densities, radii, theta, ones(4),
                    rr0, rdr, rx0, rdx, drocky,
                    wr0, wdr, wx0, wdx, dwater)).sum()
    return lnl if isfinite(lnl) else inf


@njit(cache=True)
def lnlikelihood_v(pvp, densities, radii, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater):
    npv = pvp.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in range(npv):
        lnl[i] = log(model(densities, radii, pvp[i], cs,
                           rr0, rdr, rx0, rdx, drocky,
                           wr0, wdr, wx0, wdx, dwater)).sum()
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
            lnt[j] = log(model(densities[:,j], radii[:,j], pv, cs,
                               rr0, rdr, rx0, rdx, drocky,
                               wr0, wdr, wx0, wdx, dwater).mean())
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
                                   wr0, wdr, wx0, wdx, dwater).mean())
            lnl[i] = lnt.sum()
    return lnl


@njit(cache=True)
def invert_cdf(values, cdf, res):
    x = linspace(0, 1.0, res)
    y = zeros(res)
    y[0] = values[0]
    y[-1] = values[-1]
    i, j = 0, 0
    for j in range(res-2):
        while cdf[i] < x[j+1]:
            i += 1
        if i > 0:
            a = (x[j+1] - cdf[i-1]) / (cdf[i] - cdf[i-1])
            y[j+1] = (1-a)*values[i-1] + a*values[i]
        else:
            y[j+1] = values[0]
    return x, y


def create_radius_density_map(pvs: ndarray, rd: RadiusDensityModel,
                              rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                              rres: int = 200, dres: int = 100, components = None) -> (ndarray, ndarray, ndarray):
    radii = linspace(*rlims, num=rres)
    densities = linspace(*dlims, num=dres)
    dgrid, rgrid = meshgrid(densities, radii)
    if components is None:
        components = ones(3)
    m = average_model(pvs, dgrid.ravel(), rgrid.ravel(), components,
                      rd._rr0, rd._rdr, rd._rx0, rd._rdx, rd.drocky,
                      rd._wr0, rd._wdr, rd._wx0, rd._wdx, rd.dwater).reshape(rgrid.shape)
    return radii, densities, m


def create_radius_mass_map(pvs: ndarray, rd: RadiusDensityModel,
                           rlims: tuple[float, float] = (0.5, 6.0), mlims: tuple[float, float] = (0, 24),
                           rres: int = 200, mres: int = 100, components=None) -> (ndarray, ndarray, ndarray):
    radii = linspace(*rlims, num=rres)
    masses = linspace(*mlims, num=mres)
    mgrid, rgrid = meshgrid(masses, radii)
    volumes = (4 / 3 * pi * (radii * u.R_earth).to(u.cm) ** 3).value
    dgrid = (mgrid * u.M_earth).to(u.g).value / volumes[:, newaxis]
    c = ((1 * u.M_earth).to(u.g).value / (4 / 3 * pi * (radii * u.R_earth).to(u.cm) ** 3).value)
    if components is None:
        components = ones(3)
    m = average_model(pvs, dgrid.ravel(), rgrid.ravel(), components,
                      rd._rr0, rd._rdr, rd._rx0, rd._rdx, rd.drocky,
                      rd._wr0, rd._wdr, rd._wx0, rd._wdx, rd.dwater).reshape(rgrid.shape)
    return radii, masses, c[:, newaxis]*m


def create_radius_density_icdf(pvs: ndarray,rd: RadiusDensityModel, pres: int = 100,
                               rlims: tuple[float, float] = (0.5, 6.0), dlims: tuple[float, float] = (0, 12),
                               rres: int = 200, dres: int = 100) -> (ndarray, ndarray, ndarray, ndarray, ndarray):
    radii, densities, rdmap = create_radius_density_map(pvs, rd, rlims, dlims, rres, dres)
    cdf = rdmap.cumsum(axis=1)
    cdf /= cdf[:, -1:]
    icdf = zeros((rres, pres))
    for i in range(rres):
        probs, icdf[i] = invert_cdf(densities, cdf[i], pres)

    # fix the upper and lower boundaries
    icdf[:, 0] = clip(icdf[:, 1] - diff(icdf[:, 1:3]).ravel(), 0.0, inf)
    icdf[:, -1] = icdf[:, -2] + diff(icdf[:, -3:-1]).ravel()
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


def model_means(samples, rdm, quantity: str = 'density', rlims=(0.5, 4.0), nr: int = 200):
    radius = linspace(*rlims, num=nr)
    npv = samples.shape[0]
    npt = radius.size
    models = {'rocky': zeros((2, npv, npt)), 'water': zeros((2, npv, npt)), 'puffy': zeros((2, npv, npt))}

    if quantity == 'density':
        c = 1.0
    elif quantity == 'mass':
        g_to_me = (1 * u.g).to(u.M_earth).value
        v = 4/3 * pi * (radius*u.R_earth).to(u.cm)**3  # Planet's volume in cm^3
        c = v * g_to_me
    else:
        raise ValueError

    for i, pv in enumerate(samples):
        pv = map_pv(pv)
        models['rocky'][0, i] = c * rdm.evaluate_rocky(pv[4], radius)
        models['water'][0, i] = c * rdm.evaluate_water(pv[5], radius)
        models['puffy'][0, i] = c * pv[6] * radius ** pv[7] / 2.0 ** pv[7]

        tx, ty = map_r_to_xy(radius, *pv[:4])
        models['rocky'][1, i], models['water'][1, i], models['puffy'][1, i] = mixture_weights(tx, ty)

    for k in models.keys():
        tmp = zeros((2, npt))
        for i in range(2):
            tmp[i] = nanmedian(models[k][i], 0)
        models[k] = tmp

    return radius, models


def plot_model_means(samples, rdm, quantity='density', ax=None, rlims=(0.5, 4.0), nr: int = 200, lw: float = 1):
    if ax is None:
        fig, ax = subplots()

    radius, models = model_means(samples, rdm, quantity, rlims, nr)

    for i, m in enumerate(('rocky', 'water', 'puffy')):
        meanf, weight = models[m]
        ax.plot(radius, where((weight > 0.99), meanf, nan), '-', c='k', lw=lw)
        ax.plot(radius, where((weight > 0.01) & (weight < 0.99), meanf, nan), '--', c='k', lw=lw)
    return ax
