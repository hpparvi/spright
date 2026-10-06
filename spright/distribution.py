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

from typing import Optional, Callable

import numpy as np
import arviz as az
from matplotlib.pyplot import subplots, setp
from numpy import inf, log, array, percentile, argmin, ndarray, median
from scipy.optimize import minimize

from .analytical_model import spdf

np.seterr(invalid='ignore')


class Distribution:
    """Posterior distribution of a predicted planet property.

    Stores the samples of a predicted density, mass, radius, or RV semi-amplitude, and
    optionally approximates the distribution with a uni- or bimodal Student's
    t-distribution model.

    Parameters
    ----------
    samples : ndarray
        Samples drawn from the distribution.
    quantity : {'density', 'mass', 'radius', 'k'}
        Quantity the samples represent, where ``'k'`` stands for the RV semi-amplitude.
    modes : tuple of (float, float or None)
        Initial guesses for the locations of the two distribution modes. The distribution
        is treated as unimodal if the second element is ``None``.
    fit : bool, optional
        Whether to fit the Student's t-distribution model to the samples.

    Attributes
    ----------
    quantity : {'density', 'mass', 'radius', 'k'}
        Stored quantity.
    samples : ndarray
        Array of samples.
    size : int
        Number of samples.
    units : str
        Units of the stored quantity.
    is_bimodal : bool
        Whether the distribution is modelled as bimodal.
    model : callable or None
        Fitted distribution model with a signature ``model(x, pv)``, or ``None`` if the
        model has not been fitted.
    model_pars : ndarray or None
        Fitted model parameters. These are ``[m, s, l]`` for a unimodal model and
        ``[w, m1, s1, l1, m2, s2, l2]`` for a bimodal model, where ``m`` is the location,
        ``s`` is the scale, ``l`` is the degrees of freedom, and ``w`` is the weight of
        the second mode.

    Raises
    ------
    ValueError
        If ``quantity`` is not one of the supported quantities.
    """

    _quantities = ('density', 'mass', 'radius', 'k')

    def __init__(self, samples: ndarray, quantity: str, modes: tuple[float, Optional[float]], fit: bool = True):
        if quantity not in self._quantities:
            raise ValueError(f'Unsupported quantity, quantity must be one of {self._quantities}')

        self.quantity: str = quantity
        self.samples: ndarray = samples
        self.size: int = samples.size

        if quantity == 'density':
            self.units = 'g/cm^3'
        elif quantity == 'mass':
            self.units = 'M_Earth'
        elif quantity == 'radius':
            self.units = 'R_Earth'
        elif quantity == 'k':
            self.units = 'm/s'
        else:
            self.units = None

        self.is_bimodal: bool = modes[1] is not None

        self.model: Optional[Callable] = None
        self.model_pars: Optional[ndarray] = None
        self._m1: Optional[float] = None
        self._m2: Optional[float] = None

        if fit:
            self._fit_distribution(*modes)

    def __repr__(self):
        """Summarise the distribution and the fitted distribution model."""
        p = self.model_pars
        c = (f"{self.quantity.capitalize()} distribution\nsize: {self.size}\nis bimodal: {self.is_bimodal}\n\n"
             f"Median: {median(self.samples):4.2f},\n"
             f"68% limits: {percentile(self.samples, [16,84]).round(1)},\n"
             f"95% limits: {percentile(self.samples, [2.5,97.5]).round(1)}\n\n"
             f"Distribution model:\n")

        s = ""
        if p is not None:
            if self.is_bimodal:
                s = f"  {1 - p[0]:3.2f} × T(m={p[1]:3.2f}, σ={p[2]:3.2f}, λ={p[3]:3.2f})\n+ {p[0]:3.2f} × T(m={p[4]:3.2f}, σ={p[5]:3.2f}, λ={p[6]:3.2f})"
            else:
                s = f"  T(m={p[0]:3.2f}, σ={p[1]:3.2f}, λ={p[2]:3.2f})"
        return c + s

    def _fit_kde(self, bw_fct: float = 1, lims=(-inf, inf)) -> tuple[ndarray, ndarray]:
        """Estimate the probability density of the samples using an adaptive KDE.

        Parameters
        ----------
        bw_fct : float, optional
            Bandwidth multiplier. Values larger than one give a smoother estimate.
        lims : tuple of (float, float), optional
            Lower and upper limits for the samples included in the estimate.

        Returns
        -------
        x : ndarray
            Grid over which the density is evaluated.
        y : ndarray
            Estimated probability density.
        """
        m = (self.samples > lims[0]) & (self.samples < lims[1])
        return az.kde(self.samples[m], adaptive=True, bw_fct=bw_fct)

    def _fit_distribution(self, m1: float, m2: Optional[float]) -> tuple[Callable, ndarray, float, Optional[float]]:
        """Fit a uni- or bimodal Student's t-distribution model to the samples.

        The model is fitted by maximising the likelihood with Powell's method, and the
        results are stored in the ``model`` and ``model_pars`` attributes.

        Parameters
        ----------
        m1 : float
            Initial guess for the location of the first mode.
        m2 : float or None
            Initial guess for the location of the second mode. A unimodal model is fitted
            if ``None``.

        Returns
        -------
        model : callable
            Distribution model with a signature ``model(x, pv)``.
        pv : ndarray
            Fitted model parameters (see the ``model_pars`` attribute).
        m1 : float
            Location of the first mode.
        m2 : float or None
            Location of the second mode, or ``None`` for a unimodal model.

        Notes
        -----
        All the model parameters are constrained to be non-negative and the degrees of
        freedom to be at most seven. The bimodal model additionally requires the weight
        of the second mode to be at most one and the first mode to be located below the
        second one.
        """
        if m2 is None:
            def dmodel(x, pv):
                return spdf(x, *pv)

            def minfun(pv):
                if any(pv < 0.0) or pv[2] > 7.0:
                    return inf
                return -log(dmodel(self.samples, pv)).sum() / self.size

            self._minimization_result = res = minimize(minfun, array([m1, 0.1, 1.0]), method='powell')
            self.model, self.model_pars, self._m1, self._m2 = dmodel, res.x, res.x[0], None
            return dmodel, res.x, res.x[0], None
        else:
            def dmodel(x, pv):
                return (1 - pv[0]) * spdf(x, *pv[1:4]) + pv[0] * spdf(x, *pv[4:])

            def minfun(pv):
                if any(pv < 0.0) or pv[0] > 1 or any(pv[[3, 6]] > 7.0) or (pv[1] > pv[4]):
                    return inf
                return -log(dmodel(self.samples, pv)).sum() / self.size

            self._minimization_result = res = minimize(minfun, array([0.5, m1, 0.5, 1.5, m2, 0.5, 1.5]),
                                                       method='powell')
            self.model, self.model_pars, self._m1, self._m2 = dmodel, res.x, res.x[1], res.x[4]
            return dmodel, res.x, res.x[1], res.x[4]

    def plot(self, plot_model: bool = True, plot_modes: bool = True, ax = None, bw_fct: float = 1, lims=(-inf, inf)):
        """Plot the distribution.

        Plots a kernel density estimate of the samples with the central 68% and 95%
        intervals shaded.

        Parameters
        ----------
        plot_model : bool, optional
            Whether to plot the fitted distribution model. Ignored if the model has not
            been fitted.
        plot_modes : bool, optional
            Whether to mark the mode locations of the fitted distribution model.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into. A new figure is created if ``None``.
        bw_fct : float, optional
            Bandwidth multiplier for the kernel density estimate.
        lims : tuple of (float, float), optional
            Lower and upper limits for the plotted quantity.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """
        plot_model &= self.model is not None
        ps = percentile(self.samples, [50, 16, 84, 2.5, 97.5])
        x, y = self._fit_kde(bw_fct=bw_fct, lims=lims)
        il, iu = argmin(abs(x - ps[1])), argmin(abs(x - ps[2]))
        ill, iuu = argmin(abs(x - ps[3])), argmin(abs(x - ps[4]))

        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        ax.fill_between(x, y, alpha=0.5)
        ax.fill_between(x[ill:iuu], y[ill:iuu], lw=2, fc='k', alpha=0.25)
        ax.fill_between(x[il:iu], y[il:iu], lw=2, fc='k', alpha=0.25)
        ax.plot(x, y, 'k')
        if plot_model:
            ax.plot(x, self.model(x, self.model_pars), '--k')
            if plot_modes:
                ax.axvline(self._m1, c='k', ls='--')
                if self._m2 is not None:
                    ax.axvline(self._m2, c='k', ls='--')

        xlabel = {'density': r'Density [g cm$^{-3}$]',
                  'relative density': r'Density [$\rho_{rocky}$]',
                  'mass': r'Mass [M$_\oplus$]',
                  'radius': r'Radius [R$_\oplus$]',
                  'k': r'RV semi-amplitude [m/s]'}[self.quantity]

        x1, x2 = percentile(self.samples, [1, 99])
        setp(ax, ylabel='Posterior probability', xlabel=xlabel, yticks=[], xlim=(max(x1, lims[0]), min(x2, lims[1])))
        if fig is not None:
            fig.tight_layout()
        return ax