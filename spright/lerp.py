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
    return clip((x - a)/(b - a), 0.0, 1.0)


@njit(cache=True)
def bilerp_s(r, c, r0, dr, c0, dc, data):
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
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c, r0, dr, c0, dc, data)
    return d


@njit(cache=True)
def bilerp_vrvc(r, c, r0, dr, c0, dc, data):
    npt = r.size
    d = zeros(npt)
    for i in range(npt):
        d[i] = bilerp_s(r[i], c[i], r0, dr, c0, dc, data)
    return d
