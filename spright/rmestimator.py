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
    ndarray, nan
from numpy.random import multivariate_normal, permutation, seed, normal
from scipy.optimize import minimize

from .rdmodel import RadiusDensityModel
from .version import version
from .core import mearth, rearth
from .model import model, lnlikelihood_vp, create_radius_density_icdf
from .lpf import LPF


class RMEstimator:
    """A class that computes a numerical radius-density relation.

    """
    def __init__(self, nsamples: int = 50,
                 tbl_file: Optional[Path] = None, use_tabulated_rho: bool = False, mask_bad: bool = True,
                 names: Optional[ndarray] = None,
                 radii: Optional[tuple[ndarray, ndarray]] = None,
                 masses: Optional[tuple[ndarray, ndarray]] = None,
                 densities: Optional[tuple[ndarray, ndarray]] = None):

        self.radius_means: Optional[ndarray] = None
        self.radius_uncertainties: Optional[ndarray] = None
        self.mass_means: Optional[ndarray] = None
        self.mass_uncertainties: Optional[ndarray] = None
        self.density_means: Optional[ndarray] = None
        self.density_uncertainties: Optional[ndarray] = None

        self.radius_samples: Optional[ndarray] = None
        self.mass_samples: Optional[ndarray] = None
        self.density_samples: Optional[ndarray] = None

        self.nplanets: int = 0
        self.nsamples: int = 0

        self._init_data(names, radii, masses, densities)
        self._create_samples(nsamples)

        self.rdm = RadiusDensityModel()
        self.lpf = LPF(self.radius_samples, self.density_samples, self.rdm)

        self.rdmap: Optional[ndarray] = None
        self.icdf: Optional[ndarray] = None
        self._optimization_result = None
        self._posterior_sample = None
        self._ra = None
        self._da = None
        self._pa = None

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
        self.nsamples = nsamples
        self.radius_samples = r = zeros((nsamples, self.nplanets))
        self.mass_samples = m = zeros((nsamples, self.nplanets))
        self.density_samples = zeros((nsamples, self.nplanets))
        for i in range(self.nplanets):
            self.radius_samples[:, i] = normal(self.radius_means[i], self.radius_uncertainties[i], size=nsamples)
        if self.density_means is None:
            for i in range(self.nplanets):
                self.mass_samples[:, i] = normal(self.mass_means[i], self.mass_uncertainties[i], size=nsamples)
            self.density_samples[:] = ((m * mearth) / (4 / 3 * pi * (r * rearth) ** 3))
            self.density_means = self.density_samples.mean(0)
            self.density_uncertainties = self.density_samples.std(0)
        else:
            self.mass_samples[:] = nan
            for i in range(self.nplanets):
                self.density_samples[:, i] = normal(self.density_means[i], self.density_uncertainties[i], size=nsamples)

    def add_lnprior(self, lnprior):
        self.lpf._additional_log_priors.append(lnprior)

    def model(self, rho, radius, pv, components = None):
        return self.lpf.model(rho, radius, pv, ones(3) if components is None else components)

    def optimize(self, x0=None):
        x0 = x0 or array([1.4, 2.5, 0.0, 1.8,   0.25, 0.75, 2.5,   -2.0,   -0.5, -0.5, -0.5])
        self._optimization_result = minimize(lambda x: -self.lpf.lnposterior(x), x0, method='Powell')

    def sample(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = 150, population=None):
        if population is None:
            if self.lpf.sampler is None:
                population = multivariate_normal(self._optimization_result.x,
                                                 diag(full(self.lpf._ndim, 1e-3)),
                                                 size=npop)
            else:
                population = self.lpf.sampler.chain[:, -1, :].copy()
        self.lpf.sample_mcmc(niter, thin, repeats, npop, population=population, save=False, vectorize=True)

    def posterior_samples(self, burn: int = 0, thin: int = 1):
        return self.lpf.posterior_samples(burn, thin)

    def compute_maps(self, nsamples: int = 1500,
                     rres: int = 200, dres: int = 100, pres: int = 100,
                     rlims: tuple[float, float] = (0.5, 6.0),
                     dlims: tuple[float, float] = (0, 12),
                     rseed: int = 0):
        seed(rseed)
        df = self.lpf.posterior_samples()
        xi = permutation(df.shape[0])[:nsamples]
        self._posterior_sample = pvs = df.iloc[xi]
        rd = self.lpf.rdm
        self._ra, self._da, self._pa, self.rdmap, self.icdf = create_radius_density_icdf(pvs.values,
                                                                                         rd._r0, rd._dr, rd.drocky, rd.dwater,
                                                                                         pres, rlims, dlims, rres, dres)

    def save(self, filename: Optional[Path] = None):
        if self.lpf.sampler is None or self.rdmap is None or self.icdf is None:
            raise ValueError("Cannot save before computing the idcf")

        rdh = pf.PrimaryHDU(self.rdmap)
        ich = pf.ImageHDU(self.icdf, name='icdf')
        smh = pf.BinTableHDU(Table.from_pandas(self._posterior_sample), name='samples')

        d = self._da
        r = self._ra
        p = self._pa

        rdh.header['CTYPE1'] = 'density'
        rdh.header['CRPIX1'] = 1
        rdh.header['CRVAL1'] = d[0]
        rdh.header['CDELT1'] = d[1] - d[0]

        rdh.header['CTYPE2'] = 'radius'
        rdh.header['CRPIX2'] = 1
        rdh.header['CRVAL2'] = r[0]
        rdh.header['CDELT2'] = r[1] - r[0]

        rdh.header['CREATOR'] = f'Moot v{str(version)} '
        rdh.header['CREATED'] = Time.now().to_value('fits', 'date')

        ich.header['CTYPE1'] = 'icdf'
        ich.header['CRPIX1'] = 1
        ich.header['CRVAL1'] = p[0]
        ich.header['CDELT1'] = p[1] - p[0]

        ich.header['CTYPE2'] = 'radius'
        ich.header['CRPIX2'] = 1
        ich.header['CRVAL2'] = r[0]
        ich.header['CDELT2'] = r[1] - r[0]

        # Catalog
        tbs = Table(data=[self.planet_names.astype('a20'),
                          self.radius_means, self.radius_uncertainties,
                          self.mass_means, self.mass_uncertainties,
                          self.density_means, self.density_uncertainties],
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

        hdul = pf.HDUList([rdh, ich, smh, cat, rms])
        filename = filename or Path('rdmap.fits')
        hdul.writeto(filename, overwrite=True)


    def plot_radius_density(self, pv=None, rhores: int = 200, radres: int = 200, ax=None,
                            max_samples: int = 500, cmap=None, components=None, plot_contours=False):
        arho = linspace(0, 15, rhores)
        arad = linspace(0.5, 5.5, radres)
        xrho, xrad = meshgrid(arho, arad)

        if pv is None:
            if self.lpf.sampler is not None:
                pv = self.lpf.sampler.chain[:, :, :].reshape([-1, self.lpf._ndim])
            elif self._optimization_result is not None:
                pv = self._optimization_result.x
            else:
                raise ValueError('Need to give a parameter vector (population)')

        components = asarray(components) if components is not None else ones(4)
        pdf = zeros((rhores, radres, 3))
        rd = self.lpf.rdm

        if pv.ndim == 1:
            for i in range(3):
                if components[i] != 0:
                    cs = zeros(3)
                    cs[i] = 1.0
                    pdf[:, :, i] = model(xrho.ravel(), xrad.ravel(), pv, cs, rd._r0, rd._dr, rd.drocky, rd.dwater).reshape(xrho.shape).T
        else:
            for i in range(3):
                if components[i] != 0:
                    cs = zeros(3)
                    cs[i] = 1.0
                    m = array([model(xrho.ravel(), xrad.ravel(), x, cs, rd._r0, rd._dr, rd.drocky, rd.dwater).reshape(xrho.shape) for x in
                               permutation(pv)[:max_samples]])
                    pdf[:, :, i] = m.mean(0).T

        fig = None
        if ax is None:
            fig, ax = subplots()

        ax.imshow(pdf.mean(-1), extent=(0.5, 5.5, 0.0, 15), origin='lower', cmap=cmap, aspect='auto')

        if plot_contours:
            quantiles = (0.5,)
            levels = []
            for i in range(3):
                rs = sort(pdf[:, :, i].ravel())
                crs = rs.cumsum()
                levels.append([rs[argmin(abs(crs / crs[-1] - (1.0 - q)))] for q in quantiles])
            for i in range(3):
                ax.contour(pdf[:, :, i], extent=(0.5, 5.5, 0.0, 15), levels=levels[i], colors='w')

        rhom, rhoe = self.lpf.density_samples.mean(0), self.lpf.density_samples.std(0)
        ax.errorbar(self.radius_means, rhom, xerr=self.radius_uncertainties, yerr=rhoe, fmt='ow', alpha=0.5)
        setp(ax, xlabel=r'Radius [R$_\oplus$]', ylabel=r'Density [g/cm$^3$]', ylim=(0, 15))
        if fig is not None:
            fig.tight_layout()
        return ax
