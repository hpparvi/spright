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

from numpy import ndarray, iterable, isscalar, full
from numpy.random import normal
from uncertainties.core import Variable as UVar


def sample_distribution(d, nsamples: int) -> ndarray:
    if hasattr(d, 'rvs'):
        return d.rvs(size=nsamples)
    elif isinstance(d, UVar):
        return normal(d.n, d.s, size=nsamples)
    elif iterable(d) and len(d) == 2:
        return normal(d[0], d[1], size=nsamples)
    elif isscalar(d):
        return full(nsamples,  d)
    else:
        raise ValueError("Could not interpret 'd' as a distribution")