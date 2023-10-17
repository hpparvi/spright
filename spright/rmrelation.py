from pathlib import Path
from typing import Union, Optional

import astropy.units as u
import astropy.io.fits as pf
import numpy as np
from astropy.constants import M_sun
from astropy.table import Table
from numpy import arange, array, ndarray, pi, median, zeros, sqrt
from numpy.random import normal
from scipy.constants import G
from uncertainties import ufloat
from uncertainties.core import Variable as UVar

from .distribution import Distribution
from .model import sample_density, sample_mass
from .rdmodel import RadiusDensityModel
from .relationmap import RDRelationMap, RMRelationMap

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

    def __init__(self, fname: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        fname
            RMRelation file.
        """

        self.rdm = RadiusDensityModel()
        if fname is None:
            fname = Path(__file__).parent / 'data' / 'stpm.fits'
        elif Path(fname).exists():
            fname = Path(fname)
        elif isinstance(fname, str):
            if fname.lower() == 'stpm':
                fname = Path(__file__).parent / 'data' / 'stpm.fits'
            elif fname.lower() == 'tepcat_m':
                fname = Path(__file__).parent / 'data' / 'tepcat_m.fits'
            elif fname.lower() == 'tepcat_fgk':
                fname = Path(__file__).parent / 'data' / 'tepcat_fgk.fits'
            else:
                raise ValueError()

        with pf.open(fname) as f:
            self.rdmap = RDRelationMap.load(fname)
            self.rmmap = RMRelationMap.load(fname)
            self.posterior_samples = Table.read(fname, 10).to_pandas()
            self.catalog = Table.read(fname, 11).to_pandas()
            self.rdsamples = Table.read(fname, 12).to_pandas()


    def sample(self, quantity: str, radius: oprq = None, mass: oprq = None,
               mstar: oprq = None, period: oprq = None, ecc: oprq = None,
               nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        quantity: {'radius', 'density', 'mass', 'k'}
            Returned quantity.
        radius
            Planet radius either as a single float or as a tuple with radius and its uncertainty.
        mass
            Planet mass either as a single float or as a tuple with radius and its uncertainty.
        mstar
            Stellar mass either as a single float or as a tuple with radius and its uncertainty.
        period
            Orbital period of the planet either as a single float or as a tuple with radius and its uncertainty.
        ecc
            Planet's orbital eccentricity either as a single float or as a tuple with radius and its uncertainty.
        nsamples
            Number of samples to return.

        Returns
        -------
        ndarray
            Samples from either the density or mass posterior given the planet radius.
        """
        qs = ('radius', 'density', 'mass', 'k')
        if quantity not in qs:
            raise ValueError(f"Quantity has to be one of {qs}")

        if quantity == 'density':
            return self.predict_density(radius, nsamples)
        elif quantity == 'mass':
            return self.predict_mass(radius, nsamples)
        elif quantity == 'k':
            return self.predict_rv_semi_amplitude(radius, period, mstar, ecc, nsamples)
        if quantity == 'radius':
            return self.predict_radius(mass, nsamples)

    def predict_density(self, radius: prq, nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        radius
            Planet radius either as a ufloat, a single float or as a tuple with radius and its uncertainty.
        nsamples
            Number of samples to return.

        Returns
        -------
        distribution
            Predicted bulk density distribution.
        """
        r, re = unpack_value(radius)
        rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
        return Distribution(s, 'density', self._identify_modes(r, 'density'))

    def predict_mass(self, radius: prq, nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        radius
            Planet radius either as a ufloat, a single float or as a tuple with radius and its uncertainty.
        nsamples
            Number of samples to return.

        Returns
        -------
        distribution
            Predicted mass distribution.
        """
        r, re = unpack_value(radius)
        rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
        v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
        m_g = v * s * (u.g / u.cm ** 3)
        return Distribution(m_g.to(u.M_earth).value, 'mass', self._identify_modes(r, 'mass'))

    def predict_rv_semi_amplitude(self, radius: prq, period: prq, mstar: prq, ecc: oprq = None, nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        radius
            Planet radius either as a ufloat, a single float or as a tuple with radius and its uncertainty.
        nsamples
            Number of samples to return.

        Returns
        -------
        distribution
            Predicted RV semiamplitude distribution.
        """
        r, re = unpack_value(radius)
        rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
        v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
        mpl = (v * s * (u.g / u.cm ** 3)).to(u.kg)
        if ecc is None:
            ecc = zeros(nsamples)
        else:
            ecc = normal(*unpack_value(ecc), size=nsamples)
        mst = (normal(*unpack_value(mstar), size=nsamples) * M_sun).to(u.kg)
        prd = (normal(*unpack_value(period), size=nsamples) * u.d).to(u.s)
        k = ((2*pi*G/prd)**(1/3) * mpl / mst**(2/3) * (1/sqrt(1-ecc**2))).value
        return Distribution(k, 'k', (float(median(k)), None), False)

    def predict_radius(self, mass: prq, nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        mass
            Planet mass either as a ufloat, a single float or as a tuple with radius and its uncertainty.
        nsamples
            Number of samples to return.

        Returns
        -------
        distribution
            Predicted radius distribution.
        """
        m, me = unpack_value(mass)
        ms, s = self.rmmap.sample((m, me), 'mr', nsamples)
        return Distribution(s, 'radius', (float(median(s)), None), False)

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
