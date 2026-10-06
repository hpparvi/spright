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

from typing import Optional

import astropy.units as u
import pandas as pd

from pathlib import Path
from numpy import pi, newaxis

root = Path(__file__).parent.resolve()

# Earth mass [g] and radius [cm]
mearth = (5.9742e24 * u.kg).to(u.g).value
rearth = (6.371e6 * u.m).to(u.cm).value

def rho(r, m):
    """Calculate the bulk density of a spherical body.

    Parameters
    ----------
    r : float or ndarray
        Radius.
    m : float or ndarray
        Mass.

    Returns
    -------
    float or ndarray
        Bulk density in the units of the inputs (g/cm^3 if the radius is given in
        centimetres and the mass in grams).
    """
    return m/(4/3*pi*r**3)


def read_mr():
    """Read the theoretical mass-radius table and convert it to a mass-density table.

    The table gives the planet radius as a function of mass for a set of planet
    compositions. The ``cold_h2/he`` and ``max_coll_strip`` columns are dropped.

    Returns
    -------
    mr : DataFrame
        Planet radii [R_earth] indexed by the planet mass [M_earth], with one column
        per composition.
    md : DataFrame
        Planet bulk densities [g/cm^3] with the same index and columns as ``mr``.

    Notes
    -----
    This function does not currently work: the ``data/mrtable3.txt`` file it reads is
    not shipped with the package, and it uses the ``delim_whitespace`` argument that
    has been removed from `pandas.read_csv`.
    """
    mr = pd.read_csv(root / 'data/mrtable3.txt', delim_whitespace=True, header=0, skiprows=[1], index_col=0)
    mr.index.name = 'mass'
    mr.drop(['cold_h2/he', 'max_coll_strip'], axis=1, inplace=True)
    md = pd.DataFrame(rho(mr.values*rearth, mr.index.values[:,newaxis] * mearth))
    md.columns = mr.columns
    md.set_index(mr.index, inplace=True)
    return mr, md