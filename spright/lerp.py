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

from numba import njit
from numpy import clip, floor, nan, zeros


@njit(cache=True)
def lerp(x, a, b):
    """Calculate the normalised position of a value between two limits.

    Parameters
    ----------
    x : float or ndarray
        Value to map.
    a : float
        Lower limit, mapped to zero.
    b : float
        Upper limit, mapped to one.

    Returns
    -------
    float or ndarray
        Position of ``x`` between ``a`` and ``b``, clipped to [0, 1].
    """
    return clip((x - a)/(b - a), 0.0, 1.0)


@njit(cache=True)
def bilerp_s(r, c, r0, dr, c0, dc, data):
    """Interpolate a regularly gridded table bilinearly for a scalar radius and composition.

    Parameters
    ----------
    r : float
        Radius.
    c : float
        Composition.
    r0 : float
        Radius of the first table column.
    dr : float
        Radius step of the table.
    c0 : float
        Composition of the first table row.
    dc : float
        Composition step of the table.
    data : ndarray
        Table to interpolate with a shape ``(n_composition, n_radius)``.

    Returns
    -------
    float
        Interpolated table value, or NaN if the point falls outside the table.

    Notes
    -----
    The radius must be smaller than the radius of the last table column. Compositions
    from the last table row up to one composition step above it evaluate to the last
    row.
    """
    nr = (r - r0) / dr
    ir = int(floor(nr))
    ar1 = nr - ir
    ar2 = 1.0 - ar1

    nc = (c - c0) / dc
    ic = int(floor(nc))
    ac1 = nc - ic
    ac2 = 1.0 - ac1

    if ic < 0 or ir < 0 or ic > data.shape[0] - 1 or ir >= data.shape[1] - 1:
        return nan

    if ic == data.shape[0] - 1:
        ic -= 1
        ac1 = 1.0
        ac2 = 0.0

    l00 = data[ic, ir]
    l01 = data[ic, ir + 1]
    l10 = data[ic + 1, ir]
    l11 = data[ic + 1, ir + 1]

    return (l00 * ac2 * ar2
            + l10 * ac1 * ar2
            + l01 * ac2 * ar1
            + l11 * ac1 * ar1)


@njit(cache=True)
def bilerp_vr(r, c, r0, dr, c0, dc, data):
    """Interpolate a regularly gridded table bilinearly for a radius vector and a composition.

    Parameters
    ----------
    r : ndarray
        Radii.
    c : float
        Composition.
    r0 : float
        Radius of the first table column.
    dr : float
        Radius step of the table.
    c0 : float
        Composition of the first table row.
    dc : float
        Composition step of the table.
    data : ndarray
        Table to interpolate with a shape ``(n_composition, n_radius)``.

    Returns
    -------
    ndarray
        Interpolated table values, with NaNs for the radii that fall outside the table.

    See Also
    --------
    bilerp_s : Scalar version doing the actual interpolation.
    """
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c, r0, dr, c0, dc, data)
    return d


@njit(cache=True)
def bilerp_vrvc(r, c, r0, dr, c0, dc, data):
    """Interpolate a regularly gridded table bilinearly for radius and composition vectors.

    Parameters
    ----------
    r : ndarray
        Radii.
    c : ndarray
        Compositions, one for each radius.
    r0 : float
        Radius of the first table column.
    dr : float
        Radius step of the table.
    c0 : float
        Composition of the first table row.
    dc : float
        Composition step of the table.
    data : ndarray
        Table to interpolate with a shape ``(n_composition, n_radius)``.

    Returns
    -------
    ndarray
        Interpolated table values, with NaNs for the points that fall outside the table.

    See Also
    --------
    bilerp_s : Scalar version doing the actual interpolation.
    """
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c[i], r0, dr, c0, dc, data)
    return d
