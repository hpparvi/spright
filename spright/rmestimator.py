#  MOOT
#  Copyright (C) 2022 Hannu Parviainen.
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

from pathlib import Path
from typing import Optional

import astropy.io.fits as pf
from astropy.table import Table
from astropy.time import Time
from matplotlib.pyplot import subplots, setp
from numpy import pi, diag, array, full, linspace, meshgrid, asarray, zeros, argmin, sort, ones, squeeze, isfinite, \
    ndarray, nan, clip, inf
from numpy.random import multivariate_normal, permutation, seed, normal
from scipy.optimize import minimize

from .rdmodel import RadiusDensityModel
from .version import version
from .core import mearth, rearth
from .model import create_radius_mass_map, create_radius_density_map
from .analytical_model import model
from .relationmap import RMRelationMap, RDRelationMap
from .lpf import LPF


class RMEstimator:
    """A class that computes a numerical radius-density relation.

    """
    def __init__(self, nsamples: int = 50,
                 names: Optional[ndarray] = None,
                 radii: Optional[tuple[ndarray, ndarray]] = None,
                 masses: Optional[tuple[ndarray, ndarray]] = None,
                 densities: Optional[tuple[ndarray, ndarray]] = None,
                 rock: str = 'z19',
                 water: str = 'z19',
                 seed: Optional[int] = None):

        self.radius_means: Optional[ndarray] = None
        self.radius_uncertainties: Optional[ndarray] = None
        self.mass_means: Optional[ndarray] = None
        self.mass_uncertainties: Optional[ndarray] = None
        self.density_means: Optional[ndarray] = None
        self.density_uncertainties: Optional[ndarray] = None

        self.radius_samples: Optional[ndarray] = None
        self.mass_samples: Optional[ndarray] = None
        self.density_samples: Optional[ndarray] = None

        self.seed: Optional[int] = seed
        self.nplanets: int = 0
        self.nsamples: int = 0

        self._init_data(names, radii, masses, densities)
        self._create_samples(nsamples)

        self.rdmodel = RadiusDensityModel(rock, water)
        self.lpf = LPF(self.radius_samples, self.density_samples, self.rdmodel)

        self.rdmap: Optional[RDRelationMap] = None
        self.rmmap: Optional[RMRelationMap] = None

        self._optimization_result: ndarray | None = None
        self._posterior_sample: ndarray | None = None

    def _init_data(self, names: ndarray,
                   radii: tuple[ndarray, ndarray],
                   masses: tuple[ndarray, ndarray],
                   densities: tuple[ndarray, ndarray]):
        self.planet_names = names
        self.radius_means, self.radius_uncertainties = radii
        if masses is not None:
            self.mass_means, self.mass_uncertainties = masses
        else:
            self.density_means, self.density_uncertainties = densities
        self.nplanets = self.planet_names.size

    def _create_samples(self, nsamples: int):
        seed(self.seed)
        self.nsamples = nsamples
        self.radius_samples = r = zeros((nsamples, self.nplanets))
        self.mass_samples = m = zeros((nsamples, self.nplanets))
        self.density_samples = zeros((nsamples, self.nplanets))
        for i in range(self.nplanets):
            self.radius_samples[:, i] = clip(normal(self.radius_means[i], self.radius_uncertainties[i], size=nsamples), 0, inf)
        if self.density_means is None:
            for i in range(self.nplanets):
                self.mass_samples[:, i] = clip(normal(self.mass_means[i], self.mass_uncertainties[i], size=nsamples), 0, inf)
            self.density_samples[:] = ((m * mearth) / (4 / 3 * pi * (r * rearth) ** 3))
            self.density_means = self.density_samples.mean(0)
            self.density_uncertainties = self.density_samples.std(0)
        else:
            self.mass_samples[:] = nan
            for i in range(self.nplanets):
                self.density_samples[:, i] = clip(normal(self.density_means[i], self.density_uncertainties[i], size=nsamples), 0, inf)

    def add_lnprior(self, lnprior):
        self.lpf._additional_log_priors.append(lnprior)

    def model(self, rho, radius, pv, components = None):
        return self.lpf.model(rho, radius, pv, ones(3) if components is None else components)

    def optimize(self, niter: int = 500, npop: int = 150):
        self.lpf.optimize_global(niter, npop, plot_convergence=False)
        self._optimization_result = self.lpf.de.minimum_location.copy()

    def sample(self, niter: int = 500, thin: int = 5, repeats: int = 1, population=None):
        if population is None:
            if self.lpf.sampler is None:
                population = self.lpf.de.population.copy()
            else:
                population = self.lpf.sampler.chain[:, -1, :].copy()
        self.lpf.sample_mcmc(niter, thin, repeats, population.shape[0], population=population, save=False, vectorize=True)

    def posterior_samples(self, burn: int = 0, thin: int = 1):
        return self.lpf.posterior_samples(burn, thin)

    def compute_maps(self, nsamples: int = 1500,
                     rres: int = 200, dres: int = 100, pres: int = 100,
                     rlims: tuple[float, float] = (0.5, 6.0),
                     dlims: tuple[float, float] = (0, 12),
                     mlims: tuple[float, float] = (0, 25),
                     rseed: int = 0):
        seed(rseed)
        rd = self.rdmodel
        df = self.lpf.posterior_samples()
        self._posterior_sample = pvs = df.iloc[permutation(df.shape[0])[:nsamples]]
        radii, densities, rdm = create_radius_density_map(pvs.values, rd, dres=dres, rres=rres, dlims=dlims, rlims=rlims)
        radii, masses, rmm = create_radius_mass_map(pvs.values, rd, mres=dres, rres=rres, mlims=mlims, rlims=rlims)
        self.rdmap = RDRelationMap(rdm, radii, densities, pres)
        self.rmmap = RMRelationMap(rmm, radii, masses, pres)

    def save(self, filename: Optional[Path] = None):
        if self.lpf.sampler is None or self.rdmap is None:
            raise ValueError("Cannot save before computing the maps")

        rdh = pf.PrimaryHDU(self.rdmap._pmapc)
        rdc = pf.ImageHDU(self.rdmap.xy_cdf, name='rd_cdf')
        rdi = pf.ImageHDU(self.rdmap.xy_icdf, name='rd_icdf')
        drc = pf.ImageHDU(self.rdmap.yx_cdf, name='dr_cdf')
        dri = pf.ImageHDU(self.rdmap.yx_icdf, name='dr_icdf')

        rmr = pf.ImageHDU(self.rmmap._pmapc, name='rmr')
        rmc = pf.ImageHDU(self.rmmap.xy_cdf, name='rm_cdf')
        rmi = pf.ImageHDU(self.rmmap.xy_icdf, name='rm_icdf')
        mrc = pf.ImageHDU(self.rmmap.yx_cdf, name='mr_cdf')
        mri = pf.ImageHDU(self.rmmap.yx_icdf, name='mr_icdf')

        smh = pf.BinTableHDU(Table.from_pandas(self._posterior_sample), name='samples')

        d = self.rdmap.y
        m = self.rmmap.y
        r = self.rdmap.x
        p = self.rdmap.probs

        def set_axes(h, xname, yname, x, y):
            h.header['CTYPE1'] = yname
            h.header['CRPIX1'] = 1
            h.header['CRVAL1'] = y[0]
            h.header['CDELT1'] = y[1] - y[0]

            h.header['CTYPE2'] = xname
            h.header['CRPIX2'] = 1
            h.header['CRVAL2'] = x[0]
            h.header['CDELT2'] = x[1] - x[0]

        set_axes(rdh, 'radius', 'density', r, d)
        set_axes(rdc, 'radius', 'density', r, d)
        set_axes(rdi, 'radius', 'icdf', r, p)
        set_axes(drc, 'density', 'radius', d, r)
        set_axes(dri, 'density', 'icdf', d, p)

        set_axes(rmr, 'radius', 'mass', r, m)
        set_axes(rmc, 'radius', 'mass', r, m)
        set_axes(rmi, 'radius', 'icdf', r, p)
        set_axes(mrc, 'mass', 'radius', m, r)
        set_axes(mri, 'mass', 'icdf', m, p)

        rdh.header['CREATOR'] = f'Spright v{str(version)} '
        rdh.header['CREATED'] = Time.now().to_value('fits', 'date')

        # Catalog
        tbs = Table(data=[self.planet_names.astype(str),
                          self.radius_means.astype('d'), self.radius_uncertainties.astype('d'),
                          self.mass_means.astype('d'), self.mass_uncertainties.astype('d'),
                          self.density_means.astype('d'), self.density_uncertainties.astype('d')],
                    names=['name', 'radius', 'radius_e', 'mass', 'mass_e', 'density', 'density_e'],
                    units=[None, 'R_Earth', 'R_Earth', 'M_Earth', 'M_Earth', 'g cm^-3', 'g cm^-3'])
        cat = pf.BinTableHDU(tbs, name='catalog')

        # Radius, mass, and density samples
        tbs = Table(data=[self.radius_samples.ravel(),
                          self.mass_samples.ravel(),
                          self.density_samples.ravel()],
                    names=['radius', 'mass', 'density'],
                    units=['R_Earth', 'M_Earth', 'g cm^-3'])
        rms = pf.BinTableHDU(tbs, name='rmsamples')

        hdul = pf.HDUList([rdh, rdc, rdi, drc, dri, rmr, rmc, rmi, mrc, mri, smh, cat, rms])
        filename = filename or Path('rdmap.fits')
        hdul.writeto(filename, overwrite=True)

    def plot_radius_density(self, pv=None, rhores: int = 200, radres: int = 200, ax=None,
                            max_samples: int = 500, cmap=None, components=None, plot_contours: bool = False,
                            rholim: tuple[float, float] = (0, 15), radlim: tuple[float, float] =(0.5, 5.5)):

        if ax is None:
            fig, ax = subplots()
        else:
            fig = ax.figure

        arho = linspace(*rholim, num=rhores)
        arad = linspace(*radlim, num=radres)
        xrad, xrho = meshgrid(arad, arho)
        if pv is None:
            if self.lpf.sampler is not None:
                pv = self.lpf.sampler.chain[:, :, :].reshape([-1, self.lpf._ndim])
            elif self._optimization_result is not None:
                pv = self._optimization_result
            else:
                raise ValueError('Need to give a parameter vector (population)')

        components = asarray(components) if components is not None else ones(4)
        pdf = zeros((3, rhores, radres))
        rd = self.lpf.rdm

        if pv.ndim == 1:
            pdf[:, :, :] = model(xrho.ravel(), xrad.ravel(), pv, components,
                                 rd._rr0, rd._rdr, rd._rx0, rd._rdx, rd.drocky,
                                 rd._wr0, rd._wdr, rd._wx0, rd._wdx, rd.dwater).reshape(pdf.shape)
        else:
            ns = min(pv.shape[0], max_samples)
            for x in permutation(pv)[:ns]:
                pdf += model(xrho.ravel(), xrad.ravel(), x, components,
                             rd._rr0, rd._rdr, rd._rx0, rd._rdx, rd.drocky,
                             rd._wr0, rd._wdr, rd._wx0, rd._wdx, rd.dwater).reshape(pdf.shape)
            pdf /= ns

        ax.imshow(pdf.mean(0), extent=(radlim[0], radlim[1], rholim[0], rholim[1]), origin='lower', cmap=cmap, aspect='auto')

        if plot_contours:
            quantiles = (0.5,)
            levels = []
            for i in range(3):
                rs = sort(pdf[i, :, :].ravel())
                crs = rs.cumsum()
                levels.append([rs[argmin(abs(crs / crs[-1] - (1.0 - q)))] for q in quantiles])
            for i in range(3):
                ax.contour(pdf[i, :, :], extent=(radlim[0], radlim[1], rholim[0], rholim[1]), levels=levels[i], colors='w')

        rhom, rhoe = self.lpf.density_samples.mean(0), self.lpf.density_samples.std(0)
        ax.errorbar(self.radius_means, rhom, xerr=self.radius_uncertainties, yerr=rhoe, fmt='ow', alpha=0.5)
        setp(ax, xlabel=r'Radius [R$_\oplus$]', ylabel=r'Density [g/cm$^3$]', ylim=(0, 15))
        if fig is not None:
            fig.tight_layout()
        return ax
