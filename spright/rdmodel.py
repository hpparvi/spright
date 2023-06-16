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


import pandas as pd

from numpy import floor, nan, zeros, newaxis, linspace, ndarray
from scipy.interpolate import interp1d

from .core import mearth, rearth, rho, root
from .model import bilerp_vrvc, bilerp_vr


def read_mr():
    mr = pd.read_csv(root / 'data/mrtable3.txt', delim_whitespace=True, header=0, skiprows=[1], index_col=0)
    mr.index.name = 'mass'
    mr.drop(['cold_h2/he', 'max_coll_strip'], axis=1, inplace=True)
    md = pd.DataFrame(rho(mr.values * rearth, mr.index.values[:, newaxis] * mearth))
    md.columns = mr.columns
    md.set_index(mr.index, inplace=True)
    return mr, md


class RadiusDensityModel:
    """A utility class to use the Radius-density models for rocky planets and water worlds by Li Zeng (PNAS, 2019).

    Attributes
    ----------
    models:
    radius:
    nm:
    nr:
    drocky: ndarray
        Table containing radius-density models for rocky planets
    dwater: ndarray
        Table containing radius-density models for water worlds

    """
    def __init__(self, nr: int = 200):
        """
        Parameters
        ----------
        nr
            The resolution of the radius array.
        """
        self.mr, self.md = mr, md = read_mr()
        self.models = self.mr.columns.values.copy()
        self.radius = linspace(0.5, 3.5, nr)
        self.nm = self.models.size
        self.nr = nr
        self._r0 = self.radius[0]
        self._dr = self.radius[1] - self.radius[0]

        density = zeros((self.nm, self.nr))
        for i in range(self.nm):
            ip = interp1d(mr.iloc[:, i].values, md.iloc[:, i].values, 1, bounds_error=False)
            density[i] = ip(self.radius)
        self.drocky = density[:21][::-1]
        self.dwater = density[21:]

    def evaluate_rocky(self, c, r):
        if isinstance(r, ndarray) and isinstance(c, ndarray):
            return bilerp_vrvc(r, c, self.radius[0], self._dr, 0.0, 0.05, self.drocky)
        elif isinstance(r, ndarray):
            return bilerp_vr(r, c, self.radius[0], self._dr, 0.0, 0.05, self.drocky)
        else:
            raise NotImplementedError

    def evaluate_water(self, c, r):
        if isinstance(r, ndarray) and isinstance(c, ndarray):
            return bilerp_vrvc(r, c, self.radius[0], self._dr, 0.05, 0.05, self.dwater)
        elif isinstance(r, ndarray):
            return bilerp_vr(r, c, self.radius[0], self._dr, 0.05, 0.05, self.dwater)
        else:
            raise NotImplementedError