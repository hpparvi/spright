import astropy.units as u
from numpy import diag, inf, pi, atleast_2d, zeros, squeeze,  ndarray
from numpy.random import multivariate_normal
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.param import ParameterSet as PS, GParameter as GP, NormalPrior as NP, UniformPrior as UP

from .model import model, lnlikelihood_vp


def map_pv(pv):
    pv = atleast_2d(pv)
    pv_mapped = pv.copy()
    pv_mapped[:, 7:13] = 10 ** pv[:, 7:13]
    return pv_mapped


class LPF(LogPosteriorFunction):
    def __init__(self, r: ndarray, re: ndarray, mass: ndarray, masse: ndarray, nsamples: int = 50):
        super().__init__('lpf')
        self._init_parameters()

        self._model_dim = len(self.ps)
        self._radm, self._rade = r, re
        self._massm, self._masse = mass, masse

        self.nplanets = self._radm.size
        self.nsamples: int = 0
        self.radius_samples = None
        self.density_samples = None
        self.create_samples(nsamples)

    def create_samples(self, nsamples: int):
        self.nsamples = nsamples
        samples = zeros((nsamples, self.nplanets, 2))
        for i in range(self.nplanets):
            samples[:, i, :] = multivariate_normal([self._radm[i], self._massm[i]],
                                                         diag([self._rade[i]**2, self._masse[i]**2]),
                                                         size=nsamples)
        self.radius_samples = r = samples[:, :, 0].copy()
        self.mass_samples =  m = samples[:, :, 1].copy()
        self.density_samples = ((m * u.M_earth) / (4/3*pi*(r * u.R_earth)**3)).to(u.g/u.cm**3).value

    def _init_parameters(self):
        self.ps = PS([GP('rrw1', 'rocky-water transition start',    'R_earth',   UP( 1.0,  1.8),  (0.0, inf)),
            GP('rrw2', 'rocky-water transition end',     'R_earth',   UP( 1.6,  2.2),  (0.0, inf)),
            GP('rwp1', 'water-puffy transition start',    'R_earth',   UP(1.8,  2.6),  (0.0, inf)),
            GP('rwp2', 'water-puffy transition end',     'R_earth',   UP( 2.0,  2.8),  (0.0, inf)),
            GP('mrr', 'RP density pdf mean', 'rho_rocky', NP(0.9, 0.5), (0.0, inf)),
            GP('mrw', 'WW density pdf mean', 'rho_rocky', NP(0.4, 0.5), (0.0, inf)),
            GP('mrp', 'SN density pdf mean', 'rho_rocky', NP(0.2, 0.5), (0.0, inf)),
            GP('srr', 'RP density pdf scale', 'rho_rocky', UP(-3.0, 0.0), (-inf, inf)),
            GP('srw', 'WW density pdf scale', 'rho_rocky', UP(-3.0, 0.0), (-inf, inf)),
            GP('srp', 'SN density pdf scale', 'rho_rocky', UP(-3.0, 0.0), (-inf, inf)),
            GP('l1', 'RP density pdf dof', '', NP(0.0, 0.5), (-inf, inf)),
            GP('l2', 'WW density pdf dof', '', NP(0.0, 0.5), (-inf, inf)),
            GP('l3', 'SN density pdf dof', '', NP(0.0, 0.5), (-inf, inf)),
            GP('drdr', 'SN density slope', 'drho/drad', NP(0.0, 1.0), (-inf, inf))])
        self.ps.freeze()

    @staticmethod
    def model(rho, radius, pv, component):
        return model(rho, radius, squeeze(map_pv(pv)), component)

    def lnlikelihood(self, pv):
        return lnlikelihood_vp(map_pv(pv), self.density_samples, self.radius_samples)
