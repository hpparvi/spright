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
from typing import Union

import astropy.io.fits as pf
from numpy import ndarray

from .core import root
from .lerp import bilerp_vr, bilerp_vrvc


def read_rdmodel(fname: Union[Path, str]):
    with pf.open(root / 'data' / fname) as hdul:
        dd = hdul[0].data.astype('d')
        dh = hdul[0].header
        r0 = dh['CRVAL1']
        dr = dh['CDELT1']
        xw0 = dh['CRVAL2']
        dxw = dh['CDELT2']
    return dd, r0, dr, xw0, dxw


def read_rocky_zeng19():
    return read_rdmodel('rocky_zeng19.fits')


def read_water_zeng19():
    return read_rdmodel('water_zeng19.fits')


def read_water_ag21():
    return read_rdmodel('water_aguichine21.fits')


class RadiusDensityModel:
    """A utility class to use the Radius-density models for rocky planets and water worlds by Li Zeng (PNAS, 2019).

    Attributes
    ----------
    drocky: ndarray
        Table containing radius-density models for rocky planets
    dwater: ndarray
        Table containing radius-density models for water worlds

    """
    def __init__(self, rocky: str = 'z19', water: str = 'z19'):
        """
        Parameters
        ----------
        rocky
            The theoretical radius-density model to use for rocky planets.
        water
            The theoretical radius-density model to use for water-rich planets.
        """

        rocky_models = {'z19': read_rocky_zeng19}
        water_models = {'z19': read_water_zeng19, 'a21': read_water_ag21}

        if rocky not in rocky_models.keys():
            raise ValueError()

        if water not in water_models.keys():
            raise ValueError()

        self.drocky, self._rr0, self._rdr, self._rx0, self._rdx = rocky_models[rocky]()
        self.dwater, self._wr0, self._wdr, self._wx0, self._wdx = water_models[water]()

    def evaluate_rocky(self, c, r):
        if isinstance(r, ndarray) and isinstance(c, ndarray):
            return bilerp_vrvc(r, c, self._rr0, self._rdr, self._rx0, self._rdx, self.drocky)
        elif isinstance(r, ndarray):
            return bilerp_vr(r, c, self._rr0, self._rdr, self._rx0, self._rdx, self.drocky)
        else:
            raise NotImplementedError

    def evaluate_water(self, c, r):
        if isinstance(r, ndarray) and isinstance(c, ndarray):
            return bilerp_vrvc(r, c, self._wr0, self._wdr, self._wx0, self._wdx, self.dwater)
        elif isinstance(r, ndarray):
            return bilerp_vr(r, c, self._wr0, self._wdr, self._wx0, self._wdx, self.dwater)
        else:
            raise NotImplementedError
