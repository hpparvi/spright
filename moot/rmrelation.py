from pathlib import Path
from typing import Union, Optional

import astropy.io.fits as pf
import numpy as np
from astropy.table import Table
from numpy import arange

from .distribution import Distribution
from .model import sample_density, sample_mass, rocky_radius_density

np.seterr(invalid='ignore')


class RMRelation:
    def __init__(self, fname: Optional[Union[str, Path]] = None):
        if fname is None:
            fname = Path(__file__).parent / 'data' / 'rdmap.fits'
        with pf.open(fname) as f:
            self.rd_posterior = f[0].data.copy()
            self.icdf = f[1].data.copy()
            h0 = f[0].header
            self.densities = h0['CRVAL1'] + arange(h0['NAXIS1']) * h0['CDELT1']
            h1 = f[1].header
            self.radii = h1['CRVAL2'] + arange(h1['NAXIS2']) * h1['CDELT2']
            self.probs = h1['CRVAL1'] + arange(h1['NAXIS1']) * h1['CDELT1']
            self.posterior_samples = Table.read(fname).to_pandas()
            self.catalog = Table.read(fname, 3).to_pandas()
            self.rdsamples = Table.read(fname, 4).to_pandas()

    def sample(self, quantity, radius, nsamples: int = 5000):
        try:
            r, re = radius
        except TypeError:
            r, re = radius, 1e-4
        if quantity == 'density':
            s = sample_density((r, re), self.radii, self.probs, self.icdf, nsamples)
        else:
            s = sample_mass((r, re), self.radii, self.probs, self.icdf, nsamples)
        return Distribution(s, quantity, self._identify_modes(r, quantity))

    def sample_density(self, r: float, re: float, nsamples: int = 5000):
        return sample_density((r, re), self.radii, self.probs, self.icdf, nsamples)

    def sample_mass(self, r: float, re: float, nsamples: int = 5000):
        return sample_mass((r, re), self.radii, self.probs, self.icdf, nsamples)

    def _identify_modes(self, r, quantity: str = 'density'):
        assert quantity in ('density', 'relative density', 'mass')
        ps = self.posterior_samples

        rw_start = ps.rrw1.quantile(0.15)
        rw_end = ps.rrw2.quantile(0.85)
        wp_start = ps.rwp1.quantile(0.15)
        wp_end = ps.rwp2.quantile(0.85)

        if quantity == 'density':
            c = 1.0
        else:
            c = r ** 3 / 5.0

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
