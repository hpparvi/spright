from pathlib import Path
from typing import Union

import arviz as az
import astropy.io.fits as pf
import numpy as np
from astropy.table import Table
from matplotlib.pyplot import subplots, setp
from numpy import arange, inf, percentile, argmin
from pytransit.utils.de import DiffEvol as DE
from scipy.optimize import minimize

from mmmbop.model import sample_density, sample_mass, rocky_radius_density
from mmmbop.model import spdf

np.seterr(invalid='ignore')

class RadiusMassRelation:
    def __init__(self, fname: Union[str, Path] = 'rdmap.fits'):
        with pf.open(fname) as f:
            self.rd_posterior = f[0].data.copy()
            self.icdf = f[1].data.copy()
            h0 = f[0].header
            self.densities = h0['CRVAL1'] + arange(h0['NAXIS1']) * h0['CDELT1']
            h1 = f[1].header
            self.radii = h1['CRVAL2'] + arange(h1['NAXIS2']) * h1['CDELT2']
            self.probs = h1['CRVAL1'] + arange(h1['NAXIS1']) * h1['CDELT1']
            self.posterior_samples = Table.read(fname).to_pandas()

    def _identify_modes(self, r, x, y, d, x_is_mass: bool = False):
        ps = self.posterior_samples

        rw_start = ps.rrw1.quantile(0.15)
        rw_end = ps.rrw2.quantile(0.85)
        wp_start = ps.rwp1.quantile(0.15)
        wp_end = ps.rwp2.quantile(0.85)

        c = r ** 3 / 5.0 if x_is_mass else 1.0
        rd = rocky_radius_density(r)
        rocky_density = c * ps.mrr.median() * rd
        water_density = c * ps.mrw.median() * rd
        puffy_density = c * (ps.mrp.median() + (r - ps.rwp2.median()) * ps.drdr.median())

        if r <= rw_start:
            m1, m2 = rocky_density, None
        elif rw_start < r < rw_end:
            m1, m2 = water_density, rocky_density
        elif r <= wp_start:
            m1, m2 = water_density, None
        elif wp_start < r < wp_end:
            m1, m2 = puffy_density, water_density
        else:
            m1, m2 = puffy_density, None
        return m1, m2

    def _fit_distribution(self, x, y, m1, m2):
        if m2 is None:
            def dmodel(x, pv):
                return pv[0] * spdf(x, *pv[1:])

            def minfun(pv):
                if any(pv < 0.0) or pv[1] > x[-1]:
                    return inf
                return ((y - dmodel(x, pv)) ** 2).sum()

            res = minimize(minfun, [2.0, m1, 0.1, 1.0], method='bfgs')
            return dmodel, res.x, res.x[1], None
        else:
            def dmodel(x, pv):
                return pv[0] * spdf(x, *pv[1:4]) + pv[4] * spdf(x, *pv[5:])

            def minfun(pv):
                if any(pv < 0.0) or any(pv[[1, 5]] > x[-1]) or any(pv[[3, 7]] > 7.0) or (pv[1] > pv[5]):
                    return inf
                return ((y - dmodel(x, pv)) ** 2).sum()

            de = DE(minfun, bounds=[[0, 10], [x[0], x[-1]], [0.1, 3.0], [0.1, 7.0],
                                    [0, 10], [x[0], x[-1]], [0.1, 3.0], [0.1, 7.0]],
                    npop=100)
            de.optimize(100)
            res = minimize(minfun, de.minimum_location)
            return dmodel, res.x, res.x[1], res.x[5]

    def _plot_distribution(self, r, d, ax, plot_model: bool = True, plot_modes: bool = True, x_is_mass: bool = False):
        ps = percentile(d, [50, 16, 84, 2.5, 97.5])
        x, y = az.kde(d, adaptive=True)
        y /= y.max()
        il, iu = argmin(abs(x - ps[1])), argmin(abs(x - ps[2]))
        ill, iuu = argmin(abs(x - ps[3])), argmin(abs(x - ps[4]))
        ax.fill_between(x, y, alpha=0.5)
        ax.fill_between(x[ill:iuu], y[ill:iuu], lw=2, fc='k', alpha=0.25)
        ax.fill_between(x[il:iu], y[il:iu], lw=2, fc='k', alpha=0.25)
        ax.plot(x, y, 'k')
        if plot_model or plot_modes:
            m1, m2 = self._identify_modes(r, x, y, d, x_is_mass)
            model, pv, m1, m2 = self._fit_distribution(x, y, m1, m2)
            if plot_model:
                ax.plot(x, model(x, pv), '--k')
            if plot_modes:
                ax.axvline(m1, c='k', ls='--')
                if m2 is not None:
                    ax.axvline(m2, c='k', ls='--')

    def plot_density(self, r, re, relative_to_rocky: bool = False):
        density = sample_density((r, re), self.radii, self.probs, self.icdf)
        if relative_to_rocky:
            density /= rocky_radius_density(r)
        fig, ax = subplots()
        self._plot_distribution(r, density, ax)
        xlabel = r'Density [$\rho_{rocky}$]' if relative_to_rocky else 'Density [g cm$^{-3}$]'
        setp(ax, xlabel=xlabel, ylabel='Posterior probability', yticks=[])
        return density

    def plot_mass(self, r: float, re: float):
        mass = sample_mass((r, re), self.radii, self.probs, self.icdf)
        fig, ax = subplots()
        self._plot_distribution(r, mass, ax, x_is_mass=True)
        setp(ax, xlabel='Mass [M$_\oplus$]', ylabel='Posterior probability', yticks=[])


rmr = RadiusMassRelation()
d = rmr.plot_density(1.8, 0.1, False)
# rmr.plot_mass(1.6, 0.1)