from pathlib import Path
from typing import Union, Optional

import warnings
import astropy.units as u
import astropy.io.fits as pf
import numpy as np

import mpltern

from astropy.units import UnitsWarning
from astropy.constants import M_sun
from astropy.table import Table
from matplotlib import cm
from matplotlib.pyplot import figure
from numpy import arange, array, ndarray, pi, median, zeros, sqrt, concatenate
from numpy.random import normal
from pandas import DataFrame
from scipy.constants import G
from uncertainties import ufloat
from uncertainties.core import Variable as UVar

from .distribution import Distribution
from .model import sample_density, sample_mass
from .analytical_model import map_pv, map_r_to_xy, mixture_weights
from .rdmodel import RadiusDensityModel
from .relationmap import RDRelationMap, RMRelationMap
from .util import sample_distribution

np.seterr(invalid='ignore')

prq = Union[UVar, float, tuple[float, float]]
oprq = Optional[Union[UVar, float, tuple[float, float]]]


def unpack_value(v: prq) -> tuple:
    if isinstance(v, UVar):
        r, re = v.n, v.s
    elif isinstance(v, (tuple, list)):
        r, re = v
    else:
        r, re = v, 1e-4
    return r, re


class RMRelation:
    """Numerical radius-mass relation for small short-period planets around M dwarfs.

    """

    models = {'stpm': 'stpm.fits',
              'tepcat_m': 'tepcat_m_z19.fits',
              'tepcat_m_z19': 'tepcat_m_z19.fits',
              'tepcat_m_a21': 'tepcat_m_a21.fits',
              'tepcat_fgk': 'tepcat_fgk_z19.fits',
              'tepcat_fgk_z19': 'tepcat_fgk_z19.fits',
              'tepcat_fgk_a21': 'tepcat_fgk_a21.fits',
              'exoeu_m_z19': 'exoeu_m_z19.fits',
              'exoeu_m_a21': 'exoeu_m_a21.fits',
              'exoeu_fgk_z19': 'exoeu_fgk_z19.fits',
              'exoeu_fgk_a21': 'exoeu_fgk_a21.fits'}

    def __init__(self, fname: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        fname
            RMRelation file.
        """

        self.rdm = RadiusDensityModel()

        if fname is None or not Path(fname).exists():
            fname = Path(__file__).parent / 'data' / self.models[fname or 'stpm']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnitsWarning)
            with pf.open(fname) as f:
                self.rdmap = RDRelationMap.load(fname)
                self.rmmap = RMRelationMap.load(fname)
                self.posterior_samples = Table.read(fname, 10).to_pandas()
                self.catalog = Table.read(fname, 11).to_pandas()
                self.rdsamples = Table.read(fname, 12).to_pandas()

    def sample(self, quantity: str, radius: oprq = None, mass: oprq = None,
               mstar: oprq = None, period: oprq = None, eccentricity: oprq = None,
               nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        quantity: {'radius', 'density', 'mass', 'k'}
            Returned quantity.
        radius
            Planet radius in Earth radii.
        mass
            Planet mass in Earth masses.
        mstar
            Stellar mass in Solar masses.
        period
            Orbital period of the planet in days.
        eccentricity
            Planet's orbital eccentricity.
        nsamples
            Number of samples to return.

        Notes
        -----
        The radius, period, mstar, and eccentricity can be given as floats, tuples of two floats (mean, sigma),
        uncertainties ufloats, or frozen scipy.stats distributions.

        Returns
        -------
        Distribution
            Samples from either the density or mass posterior given the planet radius.
        """
        qs = ('radius', 'density', 'mass', 'k', 'rv')
        if quantity not in qs:
            raise ValueError(f"Quantity has to be one of {qs}")

        if quantity == 'density':
            return self.predict_density(radius, nsamples)
        elif quantity == 'mass':
            return self.predict_mass(radius, nsamples)
        elif quantity in ('k', 'rv'):
            return self.predict_rv_semi_amplitude(radius, period, mstar, eccentricity, nsamples)
        if quantity == 'radius':
            return self.predict_radius(mass, nsamples)

    def predict_density(self, radius: prq, nsamples: int = 5000) -> Distribution:
        """Predicts the bulk density of the planet given its radius.

        Parameters
        ----------
        radius
            Planet radius in Earth radii. Can be either a float, a tuple of two floats (mean, sigma), an uncertainties
            ufloat, or a frozen scipy.stats distribution.
        nsamples
            Number of samples to return.

        Returns
        -------
        Distribution
            Predicted bulk density distribution in g/cm^3.
        """
        rs, rho = self.rdmap.sample(radius, 'rd', nsamples)
        return Distribution(rho, 'density', self._identify_modes(rho.mean(), 'density'))

    def predict_mass(self, radius: prq, nsamples: int = 5000) -> Distribution:
        """Predicts the mass of the planet given its radius.

        Parameters
        ----------
        radius
            Planet radius in Earth radii. Can be either a float, a tuple of two floats (mean, sigma), an uncertainties
            ufloat, or a frozen scipy.stats distribution.
        nsamples
            Number of samples to return.

        Returns
        -------
        Distribution
            Predicted mass distribution in Earth masses.
        """
        rs, s = self.rdmap.sample(radius, 'rd', nsamples)
        v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
        m_g = v * s * (u.g / u.cm ** 3)
        return Distribution(m_g.to(u.M_earth).value, 'mass', self._identify_modes(rs.mean(), 'mass'))

    def predict_rv_semi_amplitude(self, radius, period, mstar, eccentricity=0.0, nsamples: int = 5000) -> Distribution:
        """Predicts the RV semi-amplitude given planet radius, orbital period, stellar mass, and orbital eccentricity.

        Parameters
        ----------
        radius
            Planet radius in Earth radii.
        period
            Plaent's orbital period in days.
        mstar
            Mass of the host star in Solar masses.
        eccentricity
            Planet's orbital eccentricity.
        nsamples
            Number of samples to return.

        Notes
        -----
        The radius, period, mstar, and eccentricity can be given as floats, tuples of two floats (mean, sigma),
        uncertainties ufloats, or frozen scipy.stats distributions.

        Returns
        -------
        Distribution
            Predicted RV semiamplitude distribution in m/s.
        """
        rs, s = self.rdmap.sample(radius, 'rd', nsamples)
        v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
        mpl = (v * s * (u.g / u.cm ** 3)).to(u.kg)

        ecc = sample_distribution(eccentricity, nsamples)
        mst = (sample_distribution(mstar, nsamples) * u.M_sun).to(u.kg)
        prd = (sample_distribution(period, nsamples) * u.d).to(u.s)
        k = ((2 * pi * G / prd) ** (1 / 3) * mpl / mst ** (2 / 3) * (1 / sqrt(1 - ecc ** 2))).value
        return Distribution(k, 'k', (float(median(k)), None), False)

    def predict_radius(self, mass: prq, nsamples: int = 5000) -> Distribution:
        """Predicts the radius of the planet given its mass.

        Parameters
        ----------
        mass
            Planet mass in Earth masses. Can be either a float, a tuple of two floats (mean, sigma), an uncertainties
            ufloat, or a frozen scipy.stats distribution.
        nsamples
            Number of samples to return.

        Returns
        -------
        distribution
            Predicted radius distribution in Earth radii.
        """
        ms, s = self.rmmap.sample(mass, 'mr', nsamples)
        return Distribution(s, 'radius', (float(median(s)), None), False)

    def predict_class(self, radius, mass=None, max_samples: int = 5000):
        pvs = self.posterior_samples.values[:max_samples]
        ns = pvs.shape[0]
        rs = sample_distribution(radius, ns)

        weights = zeros((ns, 3))
        for i, pv in enumerate(pvs):
            pvm = map_pv(pv)
            x, y = map_r_to_xy(rs[i:i+1], *pvm[:4])
            weights[i] = concatenate(mixture_weights(x, y))
        return DataFrame(weights, columns=['rocky', 'water', 'puffy'])

    def plot_class(self, radius: None, weights: DataFrame = None, figsize=None):
        if weights is None:
            weights = self.predict_class(radius)
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(projection="ternary")
        ax.hexbin(weights.water, weights.rocky, weights.puffy, edgecolors="0.75", linewidths=0.5, gridsize=10,
                  cmap=cm.Oranges)
        ax.set_tlabel(f"Water world\n(P={1e2*weights.water.mean():2.0f}%)")
        ax.set_llabel(f"Rocky planet\n(P={1e2*weights.rocky.mean():2.0f}%)")
        ax.set_rlabel(f"Sub-Neptune\n(P={1e2*weights.puffy.mean():2.0f}%)")
        ax.taxis.set_ticks([])
        ax.laxis.set_ticks([])
        ax.raxis.set_ticks([])
        return ax

    def _identify_modes(self, r, quantity: str = 'density'):
        assert quantity in ('density', 'relative density', 'mass')
        ps = self.posterior_samples
        pv = ps.median()

        t = ps.r4 - ps.r1
        a = 0.5 - abs(ps.ww - 0.5)
        r2 = ps.r1 + t * (1.0 - ps.ww + ps.ws * a)
        r3 = ps.r1 + t * (ps.ww + ps.ws * a)

        rw_start = ps.r1.quantile(0.15)
        rw_end = r2.quantile(0.85)
        wp_start = r3.quantile(0.15)
        wp_end = ps.r4.quantile(0.85)

        if quantity == 'density':
            c = 1.0
        else:
            c = r ** 3 / 5.0

        rocky_density = c * self.rdm.evaluate_rocky(pv.cr, array([r]))[0]
        water_density = c * self.rdm.evaluate_water(pv.cw, array([r]))[0]
        puffy_density = c * (pv.ip * r ** pv.sp / 2.0 ** pv.sp)

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
