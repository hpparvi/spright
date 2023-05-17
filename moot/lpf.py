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

from numpy import inf, atleast_2d, squeeze, ndarray
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.param import ParameterSet as PS, GParameter as GP, NormalPrior as NP, UniformPrior as UP

from .model import model, lnlikelihood_vp
from .rdmodel import RadiusDensityModel


def map_pv(pv):
    pv = atleast_2d(pv)
    pv_mapped = pv.copy()
    pv_mapped[:, 7:10] = 10 ** pv[:, 7:10]
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

    def _init_parameters(self):
        self.ps = PS([GP('rrw1',   'rocky-water transition start',   'R_earth',   NP( 1.5, 0.3), ( 0.0, inf)),
                     GP('rrw2',    'rocky-water transition end',     'R_earth',   NP( 1.8, 0.3), ( 0.0, inf)),
                     GP('rwp1',    'water-puffy transition start',   'R_earth',   NP( 2.0, 0.3), ( 0.0, inf)),
                     GP('rwp2',    'water-puffy transition end',     'R_earth',   NP( 2.4, 0.3), ( 0.0, inf)),
                     GP('cr',      'Rocky planet iron ratio',        'rho_rocky', UP( 0.0, 1.0), ( 0.0, 1.0)),
                     GP('cw',      'Water world water ratio',        'rho_rocky', UP( 0.05, 1.0),( 0.0, 1.0)),
                     GP('ip',      'Sub-Neptune density intercept',  'gcm^3',     NP( 2.0, 1.5), ( 0.0, inf)),
                     GP('scaler',  'log10 RP density pdf scale',           '',          NP( 0.0, 0.6), (-inf, inf)),
                     GP('scalew',  'log10 WW density pdf scale',           '',          NP( 0.0, 0.3), (-inf, inf)),
                     GP('scalep',  'log10 SN density pdf scale',           '',          NP( 0.0, 0.6), (-inf, inf)),
                     GP('dofr',    'RP density pdf dof',             '',          NP( 5.0, 0.001), (-inf, inf)),
                     GP('dofw',    'WW density pdf dof',             '',          NP( 5.0, 0.001), (-inf, inf)),
                     GP('dofp',    'SN density pdf dof',             '',          NP( 5.0, 0.001), (-inf, inf)),
                     GP('dddrp',   'SN density slope',               'drho/drad', NP( 0.0, 1.0), (-inf, inf))])
        self.ps.freeze()

    def model(self, rho, radius, pv, component):
        return model(rho, radius, squeeze(map_pv(pv)), component,
                     self.rdm._r0, self.rdm._dr, self.rdm.drocky, self.rdm.dwater)

    def lnlikelihood(self, pv):
        return lnlikelihood_vp(map_pv(pv), self.density_samples, self.radius_samples,
                               self.rdm._r0, self.rdm._dr, self.rdm.drocky, self.rdm.dwater)
