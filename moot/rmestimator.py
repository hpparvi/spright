import astropy.io.fits as pf
import astropy.units as u
import astropy.constants as c
import uncertainties.unumpy as un
from astropy.table import Table
from astropy.time import Time
from matplotlib.pyplot import subplots, setp
from numpy import pi, diag, array, full, linspace, meshgrid, asarray, zeros, argmin, sort, ones, squeeze
from numpy.random import multivariate_normal, permutation, seed
from scipy.optimize import minimize

#from . import version
from .core import read_stpm
from .model import model, lnlikelihood_vp, create_radius_density_icdf
from .lpf import LPF, map_pv


class RMEstimator:
    def __init__(self, nsamples: int = 50):
        self._df = df = read_stpm()
        self._m_good = m = (df.eM_relative <= 0.25) & (df.eR_relative <= 0.08)
        self._radm = r  = df[m].R_Rterra.values.copy()
        self._rade = re = df[['edR_Rterra', 'euR_Rterra']].mean(axis=1).values[m]
        self._massm = mm = df[m].M_Mterra.values.copy()
        self._masse = me = df[['edM_Mterra', 'euM_Mterra']].mean(axis=1).values[m]
        rhos = (un.uarray(mm, me) * c.M_earth.to(u.g).value) / (4/3 * pi * (un.uarray(r, re) * c.R_earth.to(u.cm).value)**3)
        self._rhom = un.nominal_values(rhos)

        self._rhoe = un.std_devs(rhos)

        self.lpf = LPF(r, re, mm, me, nsamples)
        self.nplanets: int = self.lpf.nplanets
        self.model_dim: int  = self.lpf._model_dim

        self._optimization_result = None
        self._posterior_sample = None
        self._ra = None
        self._da = None
        self.icdf = None
        self.rdmap = None
        self._pa = None

    def model(self, rho, radius, pv, component):
        return self.lpf.model(rho, radius, pv, component)

    def optimize(self, x0=None):
        x0 = x0 or array([1.4, 1.8, 2.2, 2.5, 0.9, 0.4, 0.2, -1, -1, -1, 0.0, 0.0, 0.0, -0.001])
        self._optimization_result = minimize(lambda x: -self.lpf.lnposterior(x), x0, method='Powell')

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = 150, population=None,
                    use_tqdm: bool = True):

        if population is None:
            if self.lpf.sampler is None:
                population = multivariate_normal(self._optimization_result.x,
                                                 diag(full(self.model_dim, 1e-3)),
                                                 size=npop)
            else:
                population = self.lpf.sampler.chain[:, -1, :].copy()

        self.lpf.sample_mcmc(niter, thin, repeats, npop, population=population, save=False, use_tqdm=use_tqdm,
                             pool=None, lnpost=None, vectorize=True)

    def posterior_samples(self, burn: int = 0, thin: int = 1):
        return self.lpf.posterior_samples(burn, thin)

    def compute_maps(self, nsamples: int = 1500,
                     rres: int = 200, dres: int = 100, pres: int = 100,
                     rlims: tuple[float, float] = (0.5, 6.0),
                     dlims: tuple[float, float] = (0, 12),
                     rseed: int = 0):
        seed(rseed)
        df = self.lpf.posterior_samples()
        xi = permutation(df.shape[0])[:nsamples]
        self._posterior_sample = pvs = df.iloc[xi]
        self._ra, self._da, self._pa, self.rdmap, self.icdf = create_radius_density_icdf(map_pv(pvs.values), pres,
                                                                                         rlims, dlims, rres, dres)

    def save(self):

        if self.lpf.sampler is None or self.rdmap is None or self.icdf is None:
            raise ValueError("Cannot save before computing the idcf")

        rdh = pf.PrimaryHDU(self.rdmap)
        ich = pf.ImageHDU(self.icdf, name='icdf')
        smh = pf.BinTableHDU(Table.from_pandas(self._posterior_sample), name='samples')

        d = self._da
        r = self._ra
        p = self._pa

        rdh.header['CTYPE1'] = 'density'
        rdh.header['CRPIX1'] = 1
        rdh.header['CRVAL1'] = d[0]
        rdh.header['CDELT1'] = d[1] - d[0]

        rdh.header['CTYPE2'] = 'radius'
        rdh.header['CRPIX2'] = 1
        rdh.header['CRVAL2'] = r[0]
        rdh.header['CDELT2'] = r[1] - r[0]

        rdh.header['CREATOR'] = f'Moot v{str(version)} '
        rdh.header['CREATED'] = Time.now().to_value('fits', 'date')

        ich.header['CTYPE1'] = 'icdf'
        ich.header['CRPIX1'] = 1
        ich.header['CRVAL1'] = p[0]
        ich.header['CDELT1'] = p[1] - p[0]

        ich.header['CTYPE2'] = 'radius'
        ich.header['CRPIX2'] = 1
        ich.header['CRVAL2'] = r[0]
        ich.header['CDELT2'] = r[1] - r[0]

        # Catalogue
        cat = pf.BinTableHDU(Table.from_pandas(self._df), name='catalogue')

        # Radius, mass, and density samples
        tbs = Table(data=[self.lpf.radius_samples.ravel(),
                          self.lpf.mass_samples.ravel(),
                          self.lpf.density_samples.ravel()],
                    names=['radius', 'mass', 'density'],
                    units=['R_Earth', 'M_Earth', 'g cm^-3'])
        rms = pf.BinTableHDU(tbs, name='rmsamples')

        hdul = pf.HDUList([rdh, ich, smh, cat, rms])
        hdul.writeto('rdmap.fits', overwrite=True)


    def plot_radius_density(self, pv=None, rhores: int = 200, radres: int = 200, ax=None,
                            max_samples: int = 500, cmap=None, components=None, plot_contours=False):
        arho = linspace(0, 15, rhores)
        arad = linspace(0.5, 5.5, radres)
        xrho, xrad = meshgrid(arho, arad)

        if pv is None:
            if self.lpf.sampler is not None:
                pv = self.lpf.sampler.chain[:, :, :].reshape([-1, self.lpf._model_dim])
            elif self._optimization_result is not None:
                pv = self._optimization_result.x
            else:
                raise ValueError('Need to give a parameter vector (population)')

        pv = map_pv(pv)
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
                    m = array([model(xrho.ravel(), xrad.ravel(), x, cs).reshape(xrho.shape) for x in
                               permutation(pv)[:max_samples]])
                    pdf[:, :, i] = m.mean(0).T

        fig = None
        if ax is None:
            fig, ax = subplots()

        ax.imshow(pdf.mean(-1), extent=(0.5, 5.5, 0.0, 15), origin='lower', cmap=cmap, aspect='auto')

        if plot_contours:
            quantiles = (0.5,)
            levels = []
            for i in range(3):
                rs = sort(pdf[:, :, i].ravel())
                crs = rs.cumsum()
                levels.append([rs[argmin(abs(crs / crs[-1] - (1.0 - q)))] for q in quantiles])
            for i in range(3):
                ax.contour(pdf[:, :, i], extent=(0.5, 5.5, 0.0, 15), levels=levels[i], colors='w')

        ax.errorbar(self._radm, self._rhom, xerr=self._rade, yerr=self._rhoe, fmt='ow', alpha=0.5)
        setp(ax, xlabel=r'Radius [R$_\oplus$]', ylabel=r'Density [g/cm$^3$]', ylim=(0, 15))
        if fig is not None:
            fig.tight_layout()
        return ax
