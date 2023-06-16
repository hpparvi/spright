import astropy.units as u

from math import gamma

from matplotlib.pyplot import subplots
from numba import njit, prange
from numpy import clip, sqrt, zeros_like, zeros, exp, log, ones, inf, pi, isfinite, linspace, meshgrid, ndarray, where, \
    nan, floor, nanmedian, atleast_2d, mean, newaxis
from numpy.random import normal, uniform, permutation
from scipy.interpolate import RegularGridInterpolator, interp1d

@njit
def map_pv(pv):
    pv_mapped = pv[:11].copy()
    r1 = pv_mapped[0] = pv[0]
    r4 = pv_mapped[3] = pv[1]
    whw = 0.5*pv[2]*(r4-r1)
    pv_mapped[1] = pv[3] - whw
    pv_mapped[2] = pv[3] + whw
    pv_mapped[8:11] = 10 ** pv[8:11]
    return pv_mapped


@njit
def spdf(x, m, s, l):
    """Non-standardized Student's t-distribution PDF"""
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


def weights_full(x, y, x1, x2, x3, y1, y2, y3):
    w1 = ((y2-y3)*(x - x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w2 = ((y3-y1)*(x - x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w3 = 1. - w1 - w2
    return w1, w2, w3

@njit
def map_r_to_xy(r, a1, a2, b1, b2):
    """Maps the planet radius to the mixture triangle (x,y) coordinates."""
    x = clip((r-b1)/(b2-b1), 0.0, 1.0)
    y = clip(clip((r-a1)/(a2-a1), 0.0, 1.0) - x, 0.0, 1.0)
    return x, y

@njit
def mixture_weights(x, y):
    """Calculates the mixture weights using interpolation inside a triangle."""
    w1 = 1. - x - y
    w2 = y
    w3 = 1. - w1 - w2
    return w1, w2, w3


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
def model(rho, radius, pv, component, r0, dr, drocky, dwater):
    pvm = map_pv(pv)
    rwstart, rwend, wpstart, wpend = pvm[0:4]
    crocky, cwater, mpuffy, dpuffy = pvm[4:8]
    srocky, swater, spuffy = pvm[8:]

    mrocky = bilerp_vr(radius, crocky, r0, dr, 0.0, 0.05, drocky)
    mwater = bilerp_vr(radius, cwater, r0, dr, 0.05, 0.05, dwater)
    mpuffy = mpuffy * radius**dpuffy / 2.0**dpuffy

    tx, ty = map_r_to_xy(radius, rwstart, rwend, wpstart, wpend)
    w1, w2, w3 = mixture_weights(tx, ty)

    procky = component[0] * w1 * spdf(rho, mrocky, srocky, 5.0)
    pwater = component[1] * w2 * spdf(rho, mwater, swater, 5.0)
    ppuffy = component[2] * w3 * spdf(rho, mpuffy, spuffy, 5.0)
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
def lnlikelihood_sample(pv, densities, radii, r0, dr, drocky, dwater):
    ns = densities.shape[0]
    cs = ones(3)
    lnt = zeros(ns)
    if pv[0] > pv[1] or pv[2] > pv[3] or pv[0] > pv[2] or pv[1] > pv[3]:
        return -inf
    else:
        lnt[:] = 0
        for j in prange(ns):
            lnt[j] = log(model(densities[j], radii[j], pv, cs, r0, dr, drocky, dwater)).sum()
        maxl = max(lnt)
        return maxl + log(exp(lnt - maxl).mean())


@njit(parallel=True)
def lnlikelihood_vp(pvp, densities, radii, r0, dr, drocky, dwater):
    pvp = atleast_2d(pvp)
    npv = pvp.shape[0]
    ns = densities.shape[0]
    lnl = zeros(npv)
    cs = ones(3)
    for i in prange(npv):
        lnt = zeros(ns)
        r1, r4 = pvp[i, 0], pvp[i, 1]
        r2 = pvp[i, 3] - 0.5*pvp[i, 2]*(r4-r1)
        r3 = pvp[i, 3] + 0.5*pvp[i, 2]*(r4-r1)
        if r1 > r2 or r3 > r4 or r1 > r3 or r2 > r4:
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


def create_radius_mass_map(pvs: ndarray, r0, dr, drocky, dwater,
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
    m = average_model(pvs, dgrid.ravel(), rgrid.ravel(), components, r0, dr, drocky, dwater).reshape(rgrid.shape)
    return radii, masses, c[:, newaxis]*m


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
        models['puffy'][0, i] = where(radius > pv[2], pv[6]*radius**pv[7] / 2.0**pv[7], nan)
        models['puffy'][1, i] = where(radius > pv[3], pv[6]*radius**pv[7] / 2.0**pv[7], nan)

    if average:
        for k in models.keys():
            tmp = zeros((2, npt))
            for i in range(2):
                m = isfinite(models[k][i]).mean(0) < 0.5
                models[k][i, :, m] = nan
                tmp[i] = nanmedian(models[k][i], 0)
            models[k] = tmp

    return radius, models


def plot_model_means(samples, rdm, ax=None, ns: int = 500, res: int = 400, dmin: float = 0.5, dmax: float = 4.0, lwscale=1.0):
    if ax is None:
        fig, ax = subplots()

    r = linspace(dmin, dmax, res)
    weights = zeros((ns, 3, r.size))
    models = zeros((ns, 3, r.size))
    for i, j in enumerate(permutation(samples.shape[0])[:ns]):
        d = samples.iloc[j]
        r2 = d.wc - 0.5*d.ww*(d.r4-d.r1)
        r3 = d.wc - 0.5*d.ww*(d.r4-d.r1)
        weights[i] = mixture_weights(*map_r_to_xy(r, d.r1, r2, r3, d.r4))
        models[i, 0] = rdm.evaluate_rocky(d.cr, r)
        models[i, 1] = rdm.evaluate_water(d.cw, r)
        models[i, 2] = d.ip * r**d.sp / 2**d.sp
    models = mean(models, 0)
    weights = weights.mean(0)

    lims = 0.70, 0.40, 0.1
    fmts = 'k-', 'k--', 'k:'
    lws = 2, 1, 1

    for i in range(3):
        for l, f, lw in zip(lims, fmts, lws):
            ax.plot(r, where(weights[i] > l, models[i], nan), f, lw=lw*lwscale)
    return ax
