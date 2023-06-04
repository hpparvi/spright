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

from numpy import inf, atleast_2d, squeeze, ndarray, clip
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.param import ParameterSet as PS, GParameter as GP, NormalPrior as NP, UniformPrior as UP
from scipy.stats import beta

from .model import model, lnlikelihood_vp
from .rdmodel import RadiusDensityModel


def map_pv(pv):
    pv = atleast_2d(pv)
    pv_mapped = pv.copy()
    pv_mapped[:, 8:] = 10 ** pv[:, 8:]
    return pv_mapped


class LPF(LogPosteriorFunction):
    def __init__(self, radius_samples: ndarray, density_samples: ndarray, rdm: RadiusDensityModel):
        super().__init__('RadiusDensityLPF')
        self._init_parameters()
        self._ndim: int = len(self.ps)
        self.nsamples: int = radius_samples.shape[0]
        self.nplanets: int = radius_samples.shape[1]
        self.radius_samples: ndarray = radius_samples
        self.density_samples: ndarray = density_samples
        self.rdm = rdm

        p = beta(0.1, 0.1, loc=-1, scale=1.2)
        def lnprior_water(pvp):
            pvp = atleast_2d(pvp)
            wpw = clip((pvp[:, 2] - pvp[:, 1]) / (pvp[:, 3] - pvp[:, 0]), -0.99, 0.19)
            return p.logpdf(wpw)
        self._additional_log_priors.append(lnprior_water)

    def _init_parameters(self):
        self.ps = PS([GP('rrw1',     'rocky-water transition start',   'R_earth',   UP( 1.0, 2.0), ( 0.0, inf)),
                      GP('rrw2',     'rocky-water transition end',     'R_earth',   UP( 1.4, 2.6), ( 0.0, inf)),
                      GP('rwp1',     'water-puffy transition start',   'R_earth',   UP( 1.0, 2.5), ( 0.0, inf)),
                      GP('rwp2',     'water-puffy transition end',     'R_earth',   UP( 1.4, 5.0), ( 0.0, inf)),
                      GP('cr',       'rocky planet iron ratio',        '',          UP( 0.0, 1.0), ( 0.0, 1.0)),
                      GP('cw',       'water world water ratio',        '',          NP( 0.5, 0.1),( 0.0, 1.0)),
                      GP('ip',       'sub-Neptune density intercept',  'gcm^3',      NP( 2.0, 1.5), ( 0.0, inf)),
                      GP('sp',       'sub-Neptune density slope',      'drho/drad',    NP(0.0, 1.0), (-inf, inf)),
                      GP('log10_sr', 'log10 RP density pdf scale',         '',    NP( 0.0, 0.6), (-inf, inf)),
                      GP('log10_sw', 'log10 WW density pdf scale',         '',    NP( 0.0, 0.3), (-inf, inf)),
                      GP('log10_sp', 'log10 SN density pdf scale',         '',    NP( 0.0, 0.6), (-inf, inf))])
        self.ps.freeze()

    def model(self, rho, radius, pv, component):
        return model(rho, radius, squeeze(map_pv(pv)), component,
                     self.rdm._r0, self.rdm._dr, self.rdm.drocky, self.rdm.dwater)

    def lnlikelihood(self, pv):
        return lnlikelihood_vp(map_pv(pv), self.density_samples, self.radius_samples,
                               self.rdm._r0, self.rdm._dr, self.rdm.drocky, self.rdm.dwater)
