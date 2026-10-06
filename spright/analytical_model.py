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

from math import gamma

from numba import njit, prange
from numpy import sqrt, pi, clip, where, isfinite, zeros_like, atleast_1d, zeros, ndarray

from spright.lerp import bilerp_vr


@njit(cache=True)
def map_pv(pv):
    """Map a sampling-space parameter vector to the internal model parameterisation.

    The sampler works with the outer transition radii (``r1``, ``r4``) and the relative
    width and shape of the water-world population (``ww``, ``ws``), while the model needs
    the four transition radii explicitly. This function derives the inner transition radii
    ``r2`` and ``r3`` and converts the log10 density-PDF scales to linear scales.

    Parameters
    ----------
    pv : ndarray
        Parameter vector with at least 11 elements, ordered as
        ``[r1, r4, ww, ws, cr, cw, ip, sp, log10_sr, log10_sw, log10_sp]``.

    Returns
    -------
    ndarray
        Mapped parameter vector with 11 elements, ordered as
        ``[r1, r2, r3, r4, cr, cw, ip, sp, sr, sw, sp]``, where ``r1`` and ``r2`` are the
        start and end of the rocky-to-water transition, ``r3`` and ``r4`` are the start and
        end of the water-to-puffy transition, and the last three elements are the linear
        density-PDF scales of the rocky, water-world, and puffy components.

    Notes
    -----
    The inner transition radii are

    .. math::

        r_2 = r_1 + d (1 - w + p a), \\qquad r_3 = r_1 + d (w + p a),

    where :math:`d = r_4 - r_1`, :math:`w` is the population width, :math:`p` is the
    population shape, and :math:`a = 0.5 - |w - 0.5|`. This gives
    :math:`r_3 - r_2 = d (2w - 1)`, so the two transitions are separated by a pure
    water-world region when :math:`w > 0.5` and overlap when :math:`w < 0.5`. The shape
    parameter shifts both radii together, and the factor :math:`a` keeps them inside
    :math:`[r_1, r_4]` for :math:`p \\in [-1, 1]`.
    """
    pv_mapped = pv[:11].copy()
    r1 = pv_mapped[0] = pv[0]
    r4 = pv_mapped[3] = pv[1]
    d = r4 - r1
    w = pv[2]   # WW population width
    p = pv[3]   # WW population shape
    a = 0.5 - abs(w - 0.5)
    r2 = pv_mapped[1] = r1 + d * (1.0 - w + p * a)
    r3 = pv_mapped[2] = r1 + d * (w + p * a)
    pv_mapped[8:11] = 10 ** pv[8:11]
    return pv_mapped


@njit(cache=True)
def spdf(x, m, s, l):
    """Non-standardised Student's t-distribution PDF.

    Parameters
    ----------
    x : float or ndarray
        Values at which to evaluate the PDF.
    m : float or ndarray
        Location (centre) of the distribution.
    s : float
        Scale of the distribution.
    l : float
        Degrees of freedom.

    Returns
    -------
    float or ndarray
        Probability density evaluated at ``x``.
    """
    return gamma(0.5*(l + 1))/(sqrt(l*pi)*s*gamma(l/2))*(1 + ((x - m)/s)**2/l)**(-0.5*(l + 1))


def weights_full(x, y, x1, x2, x3, y1, y2, y3):
    """Calculate the barycentric coordinates of a point inside an arbitrary triangle.

    This is the general form of `mixture_weights`, which assumes a triangle with its
    vertices fixed at (0, 0), (0, 1), and (1, 0).

    Parameters
    ----------
    x, y : float or ndarray
        Coordinates of the point.
    x1, x2, x3 : float
        The x coordinates of the three triangle vertices.
    y1, y2, y3 : float
        The y coordinates of the three triangle vertices.

    Returns
    -------
    w1, w2, w3 : float or ndarray
        Weights of the three vertices. The weights sum to unity, and are all within
        [0, 1] when the point lies inside the triangle.
    """
    w1 = ((y2-y3)*(x - x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w2 = ((y3-y1)*(x - x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + ((x3-x2)*(y1-y3)))
    w3 = 1. - w1 - w2
    return w1, w2, w3


@njit(cache=True)
def map_r_to_xy(r, a1, a2, b1, b2):
    """Map the planet radius to the mixture triangle (x, y) coordinates.

    The radius traces a path along the edges of a triangle whose vertices stand for the
    rocky (0, 0), water-world (0, 1), and puffy (1, 0) populations. The resulting
    coordinates are turned into mixture weights by `mixture_weights`.

    Parameters
    ----------
    r : float or ndarray
        Planet radius [R_earth].
    a1 : float
        Start of the rocky-to-water transition [R_earth].
    a2 : float
        End of the rocky-to-water transition [R_earth].
    b1 : float
        Start of the water-to-puffy transition [R_earth].
    b2 : float
        End of the water-to-puffy transition [R_earth].

    Returns
    -------
    x, y : float or ndarray
        Mixture triangle coordinates, both within [0, 1] and with ``x + y <= 1``.

    Notes
    -----
    The transition widths are floored at 1e-4 to avoid division by zero. If the
    transitions overlap (``b1 < a2``), the path cuts across the interior of the triangle
    and the water-world weight never reaches unity.
    """
    db = max(b2-b1, 1e-4)
    x = clip((r-b1)/db, 0.0, 1.0)
    da = max(a2-a1, 1e-4)
    y = clip(clip((r-a1)/da, 0.0, 1.0) - x, 0.0, 1.0)
    return x, y


@njit(cache=True)
def mixture_weights(x, y):
    """Calculate the mixture weights using interpolation inside a triangle.

    Parameters
    ----------
    x, y : float or ndarray
        Mixture triangle coordinates from `map_r_to_xy`.

    Returns
    -------
    w1, w2, w3 : float or ndarray
        Weights of the rocky, water-world, and puffy components. The weights sum to unity.
    """
    w1 = 1. - x - y
    w2 = y
    w3 = 1. - w1 - w2
    return w1, w2, w3


@njit(cache=True)
def model(density, radius, pv, component, rr0, rdr, rx0, rdx, drocky, wr0, wdr, wx0, wdx, dwater) -> ndarray:
    """Evaluate the three-component radius-density mixture model.

    Calculates the probability density of observing a bulk density given a planet radius
    separately for the rocky, water-world, and puffy (sub-Neptune) components. Each
    component is a non-standardised Student's t-distribution with five degrees of freedom
    multiplied by its radius-dependent mixture weight.

    Parameters
    ----------
    density : float or ndarray
        Planet bulk densities [g/cm^3].
    radius : float or ndarray
        Planet radii [R_earth]. Must either have the same size as ``density`` or contain
        a single value.
    pv : ndarray
        Parameter vector in the sampling-space parameterisation (see `map_pv`).
    component : ndarray
        Multipliers for the rocky, water-world, and puffy components. Use ones to evaluate
        the full model, or set an element to zero to switch the component off.
    rr0 : float
        Radius of the first rocky-planet density table column [R_earth].
    rdr : float
        Radius step of the rocky-planet density table [R_earth].
    rx0 : float
        Iron ratio of the first rocky-planet density table row.
    rdx : float
        Iron ratio step of the rocky-planet density table.
    drocky : ndarray
        Rocky-planet density table with a shape ``(n_iron_ratio, n_radius)``.
    wr0 : float
        Radius of the first water-world density table column [R_earth].
    wdr : float
        Radius step of the water-world density table [R_earth].
    wx0 : float
        Water ratio of the first water-world density table row.
    wdx : float
        Water ratio step of the water-world density table.
    dwater : ndarray
        Water-world density table with a shape ``(n_water_ratio, n_radius)``.

    Returns
    -------
    ndarray
        Array with a shape ``(3, density.size)`` containing the weighted probability
        densities of the rocky, water-world, and puffy components. The full mixture
        model is the sum over the first axis.

    Notes
    -----
    The mean densities of the rocky and water-world components are interpolated from the
    theoretical density tables, while the mean density of the puffy component follows a
    power law, ``ip * (radius / 2)**sp``. The rocky and water-world components are set to
    zero wherever the radius or composition falls outside the density table.
    """
    density =atleast_1d(density)
    radius = atleast_1d(radius)
    pvm = map_pv(pv)
    model = zeros((3, density.size))

    rwstart, rwend, wpstart, wpend = pvm[0:4]
    crocky, cwater, mpuffy, dpuffy = pvm[4:8]
    srocky, swater, spuffy = pvm[8:]

    mrocky = bilerp_vr(radius, crocky, rr0, rdr, rx0, rdx, drocky)
    mwater = bilerp_vr(radius, cwater, wr0, wdr, wx0, wdx, dwater)
    mpuffy = mpuffy * radius**dpuffy / 2.0**dpuffy

    tx, ty = map_r_to_xy(radius, rwstart, rwend, wpstart, wpend)
    w1, w2, w3 = mixture_weights(tx, ty)

    procky = component[0] * w1 * spdf(density, mrocky, srocky, 5.0)
    pwater = component[1] * w2 * spdf(density, mwater, swater, 5.0)
    ppuffy = component[2] * w3 * spdf(density, mpuffy, spuffy, 5.0)

    model[0, :] = where(isfinite(procky), procky, 0.0)
    model[1, :] = where(isfinite(pwater), pwater, 0.0)
    model[2, :] = ppuffy
    return model


@njit(parallel=True)
def average_model(samples, density, radius, components, rr0, rdr, rx0, rdx, drocky,  wr0, wdr, wx0, wdx, dwater):
    """Average the radius-density mixture model over a set of parameter vectors.

    Evaluates `model` for each parameter vector in parallel and returns the mean, giving
    the posterior-averaged model when ``samples`` contains posterior samples.

    Parameters
    ----------
    samples : ndarray
        Parameter vectors in the sampling-space parameterisation with a shape
        ``(n_samples, n_parameters)``.
    density : ndarray
        Planet bulk densities [g/cm^3].
    radius : ndarray
        Planet radii [R_earth]. Must either have the same size as ``density`` or contain
        a single value.
    components : ndarray
        Multipliers for the rocky, water-world, and puffy components.
    rr0, rdr, rx0, rdx, drocky
        Rocky-planet density table and its grid definition (see `model`).
    wr0, wdr, wx0, wdx, dwater
        Water-world density table and its grid definition (see `model`).

    Returns
    -------
    ndarray
        Array with a shape ``(3, density.size)`` containing the averaged weighted
        probability densities of the rocky, water-world, and puffy components.
    """
    npv =samples.shape[0]
    t = zeros((3, density.size))
    for i in prange(npv):
        t += model(density, radius, samples[i], components,
                   rr0, rdr, rx0, rdx, drocky,
                   wr0, wdr, wx0, wdx, dwater)
    return t/npv
