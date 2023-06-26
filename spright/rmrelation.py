from pathlib import Path
from typing import Union, Optional

import astropy.units as u
import astropy.io.fits as pf
import numpy as np
from astropy.table import Table
from numpy import arange, array, ndarray, pi
from uncertainties import ufloat

from .distribution import Distribution
from .model import sample_density, sample_mass
from .rdmodel import RadiusDensityModel
from .relationmap import RDRelationMap, RMRelationMap

np.seterr(invalid='ignore')


class RMRelation:
    """Numerical radius-mass relation for small short-period planets around M dwarfs.

    """
    def __init__(self, fname: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        fname
            Radius-density table file.
        """

        self.rdm = RadiusDensityModel()
        if fname is None:
            fname = Path(__file__).parent / 'data' / 'stpm.fits'
        with pf.open(fname) as f:
            self.rdmap = RDRelationMap.load(fname)
            self.rmmap = RMRelationMap.load(fname)
            self.posterior_samples = Table.read(fname, 10).to_pandas()
            self.catalog = Table.read(fname, 11).to_pandas()
            self.rdsamples = Table.read(fname, 12).to_pandas()

    def sample(self, quantity: str,
               radius: Optional[Union[float, tuple[float, float]]] = None,
               mass: Optional[Union[float, tuple[float, float], ufloat]] = None,
               mstar: Optional[Union[float, tuple[float, float], ufloat]] = None,
               nsamples: int = 5000) -> Distribution:
        """

        Parameters
        ----------
        quantity: {'radius', 'density', 'mass', 'k'}
            Returned quantity.
        radius
            Planet radius either as a single float or as a tuple with radius and its uncertainty.
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
            if radius is None:
                raise ValueError("The planet radius should be given for density prediction.")
            elif isinstance(radius, (tuple, list)):
                r, re = radius
            else:
                r, re = radius, 1e-4
            rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
            return Distribution(s, quantity, self._identify_modes(r, quantity))

        elif quantity == 'mass':
            if radius is None:
                raise ValueError("The planet radius should be given for mass prediction.")
            elif isinstance(radius, (tuple, list)):
                r, re = radius
            else:
                r, re = radius, 1e-4
            rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
            v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
            m_g = v * s * (u.g / u.cm ** 3)
            return Distribution( m_g.to(u.M_earth).value, quantity, self._identify_modes(r, quantity))

        elif quantity == 'k':
            if radius is None or mstar is None:
                raise ValueError("The planet radius and stellar mass should be given for RV semi-amplitude prediction.")
            elif isinstance(radius, (tuple, list)):
                r, re = radius
            else:
                r, re = radius, 1e-4
            rs, s = self.rdmap.sample((r, re), 'rd', nsamples)
            v = 4 / 3 * pi * (rs * u.R_earth).to(u.cm) ** 3
            m_g = v * s * (u.g / u.cm ** 3)
            #TODO: Finish the K calculation
            return Distribution( m_g.to(u.M_earth).value, quantity, self._identify_modes(r, quantity))

        if quantity == 'radius':
            if mass is None:
                raise ValueError("The planet mass should be given for density prediction.")
            elif isinstance(mass, (tuple, list)):
                m, me = mass
            else:
                m, me = mass, 1e-4
            ms, s = self.rmmap.sample((m, me), 'mr', nsamples)
            return Distribution(s, quantity, self._identify_modes(m, quantity))

    def sample_density(self, r: float, re: float, nsamples: int = 5000):
        return sample_density((r, re), self.radii, self.probs, self.icdf, nsamples)

    def sample_mass(self, r: float, re: float, nsamples: int = 5000):
        return sample_mass((r, re), self.radii, self.probs, self.icdf, nsamples)

    def _identify_modes(self, r, quantity: str = 'density'):
        assert quantity in ('density', 'relative density', 'mass')
        ps = self.posterior_samples
        pv = ps.median()

        rw_start = ps.r1.quantile(0.15)
        rw_end = (ps.wc - 0.5*ps.ww*(ps.r4 - ps.r1)).quantile(0.85)
        wp_start = (ps.wc + 0.5*ps.ww*(ps.r4 - ps.r1)).quantile(0.15)
        wp_end = ps.r4.quantile(0.85)

        if quantity == 'density':
            c = 1.0
        else:
            c = r ** 3 / 5.0

        rocky_density = c * self.rdm.evaluate_rocky(pv.cr, array([r]))[0]
        water_density = c * self.rdm.evaluate_water(pv.cw, array([r]))[0]
        puffy_density = c * (pv.ip * r**pv.sp / 2.0**pv.sp)

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
