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

from numpy import inf, atleast_2d, squeeze, ndarray, clip, where, nan_to_num
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.param import ParameterSet as PS, GParameter as GP, NormalPrior as NP, UniformPrior as UP

from .lnlikelihood import lnlikelihood_vp
from .analytical_model import map_pv, model
from .rdmodel import RadiusDensityModel


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

        # Additional sub-Neptune density prior
        # ------------------------------------
        # This prior rejects any solutions where the beginning to the sub-Neptune population (r3)
        # would have a higher density than a rocky-world at the same radius.
        def sn_density_prior(pv):
            pv = atleast_2d(pv)
            d = pv[:, 1] - pv[:, 0]
            a = 0.5 - abs(pv[:, 2] - 0.5)
            r3 = pv[:,0] + d * (pv[:, 2] + pv[:, 3] * a)

            puffy_rho_at_r3 = pv[:, 6] * pv[:, 0] ** pv[:, 7] / 2.0 ** pv[:, 7]
            rocky_rho_at_r3 = nan_to_num(self.rdm.evaluate_rocky(0.0, r3), nan=inf)
            return where(puffy_rho_at_r3 > rocky_rho_at_r3, -inf, 0.)
        self._additional_log_priors.append(sn_density_prior)


    def _init_parameters(self):
        self.ps = PS([GP('r1',       'rocky transition start',   'R_earth',   UP( 0.5, 2.5), ( 0.0, inf)),
                      GP('r4',       'puffy transition end',     'R_earth',   UP( 1.0, 4.0), ( 0.0, inf)),
                      GP('ww',       'relative ww population width', '', UP(0.0, 1.0), (0.0, 1.0)),
                      GP('ws',       'water world population shape', 'R_earth', UP(-1.0, 1.0), (-1.0, 1.0)),
                      GP('cr',       'rocky planet iron ratio',        '',          UP( 0.0, 1.0), ( 0.0, 1.0)),
                      GP('cw',       'water world water ratio',        '',          NP( 0.5, 0.1),( 0.0, 1.0)),
                      GP('ip',       'SN density at r=2',  'gcm^3',            UP( 0.1, 7.0), ( 0.0, inf)),
                      GP('sp',       'SN density exponent',      'drho/drad',    NP(-0.5, 1.5), (-inf, inf)),
                      GP('log10_sr', 'log10 RP density pdf scale',         '',    NP( 0.0, 0.35), (-inf, inf)),
                      GP('log10_sw', 'log10 WW density pdf scale',         '',    NP( 0.0, 0.35), (-inf, inf)),
                      GP('log10_sp', 'log10 SN density pdf scale',         '',    NP( 0.0, 0.35), (-inf, inf))])
        self.ps.freeze()

    def model(self, rho, radius, pv, component):
        r = self.rdm
        return model(rho, radius, pv, component,
                     r._rr0, r._rdr, r._rx0, r._rdx, r.drocky,
                     r._wr0, r._wdr, r._wx0, r._wdx, r.dwater)

    def lnlikelihood(self, pv):
        r = self.rdm
        return lnlikelihood_vp(pv, self.density_samples, self.radius_samples,
                               r._rr0, r._rdr, r._rx0, r._rdx, r.drocky,
                               r._wr0, r._wdr, r._wx0, r._wdx, r.dwater)
