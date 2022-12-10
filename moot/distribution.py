from typing import Optional, Callable

import numpy as np
import arviz as az
from matplotlib.pyplot import subplots, setp
from numpy import inf, log, array, percentile, argmin, ndarray
from scipy.optimize import minimize

from .model import spdf

np.seterr(invalid='ignore')


class Distribution:
    """Mass or density distribution.


    Attributes
    ----------
    quantity: {'density', 'mass'}
        Stored quantity
    samples: ndarray
        Array of samples
    size: int
        Number of samples

    """
    def __init__(self, samples: ndarray, quantity: str, modes: tuple[float, Optional[float]], fit: bool = True):
        self.quantity: str = quantity
        self.samples: ndarray = samples
        self.size: int = samples.size

        self.is_bimodal: bool = modes[1] is not None

        self.model: Optional[Callable] = None
        self.model_pars: Optional[ndarray] = None
        self._m1: Optional[float] = None
        self._m2: Optional[float] = None

        if fit:
            self._fit_distribution(*modes)

    def __repr__(self):
        p = self.model_pars
        c = f"{self.quantity.capitalize()} distribution\nsize: {self.size}\nis bimodal: {self.is_bimodal}\n\nDistribution model:\n"
        if self.is_bimodal:
            s = f"  {1 - p[0]:3.2f} × T(m={p[1]:3.2f}, σ={p[2]:3.2f}, λ={p[3]:3.2f})\n+ {p[0]:3.2f} × T(m={p[4]:3.2f}, σ={p[5]:3.2f}, λ={p[6]:3.2f})"
        else:
            s = f"  T(m={p[0]:3.2f}, σ={p[1]:3.2f}, λ={p[2]:3.2f})"
        return c + s

    def _fit_kde(self) -> tuple[ndarray, ndarray]:
        return az.kde(self.samples, adaptive=True)

    def _fit_distribution(self, m1: float, m2: Optional[float]) -> tuple[Callable, ndarray, float, Optional[float]]:
        if m2 is None:
            def dmodel(x, pv):
                return spdf(x, *pv)

            def minfun(pv):
                if any(pv < 0.0) or pv[2] > 7.0:
                    return inf
                return -log(dmodel(self.samples, pv)).sum() / self.size

            self._minimization_result = res = minimize(minfun, array([m1, 0.1, 1.0]), method='powell')
            return dmodel, res.x, res.x[1], None
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

    def plot(self, plot_model: bool = True, plot_modes: bool = True, ax = None):
        ps = percentile(self.samples, [50, 16, 84, 2.5, 97.5])
        x, y = self._fit_kde()
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
                  'mass': r'Mass [M$_\oplus$]'}[self.quantity]

        setp(ax, ylabel='Posterior probability', xlabel=xlabel, yticks=[], xlim=percentile(self.samples, [1, 99]))
        if fig is not None:
            fig.tight_layout()