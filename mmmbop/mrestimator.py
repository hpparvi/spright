from matplotlib.pyplot import subplots, setp
from numpy import diag, inf, atleast_2d, array, full, linspace, meshgrid, asarray, zeros, argmin, sort, ones
from numpy.random import multivariate_normal, permutation
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction as LPF
from pytransit.param import ParameterSet as PS, GParameter as GP, NormalPrior as NP, UniformPrior as UP
from scipy.optimize import minimize

from .core import read_stpm
from .model import model, lnlikelihood_vp


class MREstimator(LPF):
    def __init__(self, nsamples: int = 50):
        super().__init__('sptm')
        self._samples = None
        self._init_parameters()

        self._ndim = len(self.ps)
        self._df = df = read_stpm()
        self._m_good = m = (df.eM_relative <= 0.25) & (df.eR_relative <= 0.08)
        self._radm = df[m].R_Rterra.values.copy()
        self._rade = df[['edR_Rterra', 'euR_Rterra']].mean(axis=1).values[m]
        self._rhom = df[m].rho_rhoterra.values.copy()
        self._rhoe = df[['edrho_rhoterra', 'eurho_rhoterra']].mean(axis=1).values[m]

        self.nplanets = self._radm.size
        self._isample = 0
        self._nsamples = 0
        self._optimization_result = None
        self.sampler = None
        self._sampling_mode = 'optimize'

        self.create_samples(nsamples)

    def create_samples(self, nsamples: int):
        self._nsamples = nsamples
        self._samples = zeros((nsamples, self.nplanets, 2))
        for i in range(self.nplanets):
            self._samples[:, i, :] = multivariate_normal([self._radm[i], self._rhom[i]],
                                                         diag([self._rade[i]**2, self._rhoe[i]**2]),
                                                         size=nsamples)

    def _init_parameters(self):
        self.ps = PS([GP('rrw1', 'rocky-water transition start',    'R_earth',   UP( 1.0,  2.0),  (0.0, inf)),
            GP('rrw2', 'rocky-water transition end',     'R_earth',   UP( 1.0,  2.0),  (0.0, inf)),
            GP('rwp1', 'water-puffy transition start',    'R_earth',   UP( 2.0,  2.6),  (0.0, inf)),
            GP('rwp2', 'water-puffy transition end',     'R_earth',   UP( 2.0,  2.6),  (0.0, inf)),
            #GP('rrw1', 'rocky-water transition center', 'R_earth', NP(1.5, 0.3), (0.0, inf)),
            #GP('rrw2', 'rocky-water transition width', 'R_earth', NP(0.5, 0.2), (0.0, inf)),
            #GP('rwp1', 'water-puffy transition center', 'R_earth', NP(2.3, 0.3), (0.0, inf)),
            #GP('rwp2', 'water-puffy transition width', 'R_earth', NP(0.5, 0.2), (0.0, inf)),
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

    def sample(self):
        r, d = self._samples[self._isample].T
        self._isample = (self._isample + 1)%self._nsamples
        return r, d

    def model(self, rho, radius, pv, component):
        return model(rho, radius, self._map_pv(pv), component)

    @staticmethod
    def _map_pv(pv):
        pv = atleast_2d(pv)
        pv_mapped = pv.copy()
        #pv_mapped[:, 0] = pv[:, 0] - 0.5*pv[:, 1]
        #pv_mapped[:, 1] = pv[:, 0] + 0.5*pv[:, 1]
        #pv_mapped[:, 2] = pv[:, 2] - 0.5*pv[:, 3]
        #pv_mapped[:, 3] = pv[:, 2] + 0.5*pv[:, 3]
        pv_mapped[:, 7:13] = 10**pv[:, 7:13]
        return pv_mapped

    def lnlikelihood(self, pv):
        return lnlikelihood_vp(self._map_pv(pv), self._samples[:, :, 1], self._samples[:, :, 0])

    def optimize(self, x0=None):
        self.optimize_local(x0)

    def optimize_local(self, x0=None):
        #x0 = x0 or array([1.5, 0.2, 2.4, 0.2, 0.9, 0.4, 0.2, -1, -1, -1, 0.0, 0.0, 0.0, -0.001])
        x0 = x0 or array([1.4, 1.8, 2.2, 2.5, 0.9, 0.4, 0.2, -1, -1, -1, 0.0, 0.0, 0.0, -0.001])
        self._optimization_result = minimize(lambda x: -self.lnposterior(x), x0, method='Powell')

    def optimize_global(self, *nargs, **kwargs):
        raise NotImplementedError()

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = 150, population=None,
                    label='MCMC sampling', reset=True, leave=True, save=False, use_tqdm: bool = True):

        if population is None:
            if self.sampler is None:
                population = multivariate_normal(self._optimization_result.x,
                                                 diag(full(self._ndim, 1e-3)),
                                                 size=npop)
            else:
                population = self.sampler.chain[:, -1, :].copy()

        super().sample_mcmc(niter, thin, repeats, npop, population=population, label=label, reset=reset,
                            leave=leave, save=save, use_tqdm=use_tqdm, pool=None, lnpost=None, vectorize=True)

    def plot_radius_density(self, pv=None, rhores: int = 200, radres: int = 200, ax=None,
                            max_samples: int = 1500, cmap=None, components=None, plot_contours=False):
        arho = linspace(0, 1.7, rhores)
        arad = linspace(0.5, 5.5, radres)
        xrho, xrad = meshgrid(arho, arad)

        if pv is None:
            if self.sampler is not None:
                pv = self.sampler.chain[:, :, :].reshape([-1, self._ndim])
            elif self._optimization_result is not None:
                pv = self._optimization_result.x
            else:
                raise ValueError('Need to give a parameter vector (population)')

        pv = self._map_pv(pv)

        components = asarray(components) if components is not None else ones(3)

        pdf = zeros((rhores, radres, 3))

        if pv.ndim == 1:
            for i in range(3):
                if components[i] != 0:
                    cs = zeros(3)
                    cs[i] = 1.0
                    pdf[:, :, i] = model(xrho.ravel(), xrad.ravel(), pv, cs).reshape(xrho.shape).T
        else:
            for i in range(3):
                if components[i] != 0:
                    cs = zeros(3)
                    cs[i] = 1.0
                    m = array([model(xrho.ravel(), xrad.ravel(), x, cs).reshape(xrho.shape) for x in permutation(pv)[:max_samples]])
                    pdf[:, :, i] = m.mean(0).T

        fig = None
        if ax is None:
            fig, ax = subplots()

        ax.imshow(pdf.mean(-1), extent=(0.5, 5.5, 0.0, 1.7), origin='lower', cmap=cmap)

        if plot_contours:
            quantiles = (0.5,)
            levels = []
            for i in range(3):
                rs = sort(pdf[:, :, i].ravel())
                crs = rs.cumsum()
                levels.append([rs[argmin(abs(crs/crs[-1] - (1.0 - q)))] for q in quantiles])
            for i in range(3):
                ax.contour(pdf[:, :, i], extent=(0.5, 5.5, 0.0, 1.7), levels=levels[i], colors='w')

        ax.errorbar(self._radm, self._rhom, xerr=self._rade, yerr=self._rhoe, fmt='ow', alpha=0.5)
        setp(ax, xlabel=r'Radius [R$_\oplus$]', ylabel=r'Density [$\rho$ rocky]', ylim=(0, 1.7))
        if fig is not None:
            fig.tight_layout()
        return ax
