#  Spright
#  Copyright (C) 2022-2026 Hannu Parviainen.
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

from astropy import units as u
from numpy import ndarray, atleast_2d, pi
from numpy.random import uniform
from scipy.interpolate import RegularGridInterpolator

from .model import create_radius_density_icdf
from .rdmodel import RadiusDensityModel


def create_mock_sample(r: ndarray, pv: ndarray, quantity: str = 'mass') -> ndarray:
    rdm = RadiusDensityModel()
    radii, densities, probs, rdmap, icdf = create_radius_density_icdf(atleast_2d(pv),
                                                                      rdm._r0, rdm._dr, rdm.drocky, rdm.dwater,
                                                                      pres=300, rres=300, dres=300)
    rgi = RegularGridInterpolator((radii, probs), icdf, bounds_error=False)
    if quantity == 'density':
        return rgi((r, uniform(size=r.size)))
    elif quantity == 'mass':
        v = 4/3 * pi * (r*u.R_earth).to(u.cm)**3
        m_g = v * rgi((r, uniform(size=r.size))) * (u.g / u.cm**3)
        return m_g.to(u.M_earth).value
    else:
        raise ValueError("Quantity has to be either 'mass' or 'density'.")
