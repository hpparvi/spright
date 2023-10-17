import astropy.io.fits as pf

from pathlib import Path
from typing import Optional, Union

from matplotlib.pyplot import subplots, setp
from numpy import ndarray, zeros, isfinite, arange, linspace
from numpy.random import normal, uniform
from scipy.interpolate import RegularGridInterpolator

from .model import invert_cdf


class RelationMap:
    def __init__(self, xname: str, yname: str,
                 data: Optional[ndarray] = None, x: Optional[ndarray] = None, y: Optional[ndarray] = None,
                 xunit: str = '', yunit: str = '', pres: int = 100):

        self.xname = xname
        self.yname = yname
        self.xunit = xunit
        self.yunit = yunit
        self.zname = 'ICDF'
        self.relations = {'xy': 'xy', self.xname[0] + self.yname[0]: 'xy', 'yx': 'yx',
                          self.yname[0] + self.xname[0]: 'yx'}
        self.pres: int = pres
        self.probs: ndarray = linspace(0, 1, pres)

        self.xy_cdf: Optional[ndarray] = None
        self.xy_icdf: Optional[ndarray] = None
        self.yx_cdf: Optional[ndarray] = None
        self.yx_icdf: Optional[ndarray] = None

        self.data: Optional[ndarray] = None
        self.x: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.xres: int = 0
        self.yres: int = 0

        if data is not None:
            self.init_data(data, x, y, self.probs)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        raise NotImplementedError

    def init_data(self, data: ndarray, x: ndarray, y: ndarray, probs: ndarray,
                  xy_cdf: Optional[ndarray] = None, xy_icdf: Optional[ndarray] = None,
                  yx_cdf: Optional[ndarray] = None, yx_icdf: Optional[ndarray] = None):
        self.data = data.copy()
        self.x = x.copy()
        self.y = y.copy()
        self.probs = probs.copy()
        self.xres = x.size
        self.yres = y.size
        if xy_cdf is None:
            self._create_xy_maps()
            self._create_yx_maps()
        else:
            self.xy_cdf = xy_cdf.copy()
            self.xy_icdf = xy_icdf.copy()
            self.yx_cdf = yx_cdf.copy()
            self.yx_icdf = yx_icdf.copy()

    def _create_xy_maps(self):
        cdf = self.data.cumsum(axis=1)
        cdf /= cdf[:, -1:]
        icdf = zeros((self.xres, self.pres))
        for i in range(self.xres):
            probs, icdf[i] = invert_cdf(self.y, cdf[i], self.pres)
        self.xy_cdf = cdf
        self.xy_icdf = icdf
        self.probs = probs

    def _create_yx_maps(self):
        cdf = self.data.cumsum(axis=0).T
        cdf /= cdf[:, -1:]
        icdf = zeros((self.yres, self.pres))
        for i in range(self.yres):
            probs, icdf[i] = invert_cdf(self.x, cdf[i], self.pres)
        self.yx_cdf = cdf
        self.yx_icdf = icdf
        self.probs = probs

    def sample(self, v: tuple[float, float], relation: str, nsamples: int = 20_000) -> (ndarray, ndarray):
        if relation not in self.relations:
            raise ValueError(f"Relation needs to be on of {list(self.relations.keys())}")
        relation = self.relations[relation]
        if relation == 'xy':
            rgi = RegularGridInterpolator((self.x, self.probs), self.xy_icdf, bounds_error=False)
        else:
            rgi = RegularGridInterpolator((self.y, self.probs), self.yx_icdf, bounds_error=False)
        vs = normal(v[0], v[1], nsamples)
        samples = rgi((vs, uniform(size=nsamples)))
        m = isfinite(samples)
        return vs[m], samples[m]

    def plot_map(self, ax=None, cm=None):
        if ax is None:
            fig, ax = subplots()
        ax.imshow(self.data.T, origin='lower', aspect='auto', cmap=cm,
                  extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        setp(ax, xlabel=f"{self.xname} [{self.xunit}]", ylabel=f"{self.yname} [{self.yunit}]")

    def plot_cdf(self, relation: str = 'xy', ax=None):
        if relation not in self.relations:
            raise ValueError(f"Relation needs to be on of {list(self.relations.keys())}")
        relation = self.relations[relation]
        if ax is None:
            fig, ax = subplots()
        if relation == 'xy':
            ax.imshow(self.xy_cdf.T, origin='lower', aspect='auto',
                      extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
            setp(ax, xlabel=f"{self.xname} [{self.xunit}]", ylabel=f"{self.yname} [{self.yunit}]")
        else:
            ax.imshow(self.yx_cdf.T, origin='lower', aspect='auto',
                      extent=(self.y[0], self.y[-1], self.x[0], self.x[-1]))
            setp(ax, xlabel=f"{self.yname} [{self.yunit}]", ylabel=f"{self.xname} [{self.xunit}]")

    def plot_icdf(self, relation: str = 'xy', ax=None):
        if relation not in self.relations:
            raise ValueError(f"Relation needs to be one of {list(self.relations.keys())}")
        relation = self.relations[relation]

        if ax is None:
            fig, ax = subplots()
        else:
            fig = ax.get_figure()
        if relation == 'xy':
            l = ax.imshow(self.xy_icdf.T, origin='lower', aspect='auto',
                          extent=(self.x[0], self.x[-1], 0, 1), vmin=0)
            setp(ax, xlabel=f"{self.xname} [{self.xunit}]", ylabel='ICDF')
        else:
            l = ax.imshow(self.yx_icdf.T, origin='lower', aspect='auto',
                          extent=(self.y[0], self.y[-1], 0, 1), vmin=0)
            setp(ax, xlabel=f"{self.yname} [{self.yunit}]", ylabel='ICDF')
        fig.colorbar(l, label=f"{self.yname} [{self.yunit}]" if relation == 'xy' else f"{self.xname} [{self.xunit}]")


class RDRelationMap(RelationMap):
    def __init__(self, data: Optional[ndarray] = None, radii: Optional[ndarray] = None, densities: Optional[ndarray] = None, pres: int = 100):
        super().__init__('radius', 'density', data, radii, densities, xunit=r'R$_\oplus$', yunit=r'g/cm$^3$', pres=pres)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        with pf.open(filename) as f:
            rdr = f[0].data.byteswap().newbyteorder()
            rdc = f['rd_cdf'].data.byteswap().newbyteorder()
            rdi = f['rd_icdf'].data.byteswap().newbyteorder()
            drc = f['dr_cdf'].data.byteswap().newbyteorder()
            dri = f['dr_icdf'].data.byteswap().newbyteorder()
            h = f[0].header
            density = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
            radius = h['CRVAL2'] + h['CDELT2']*arange(h['NAXIS2'])
            h = f['rd_icdf'].header
            prob = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
        r = cls()
        r.init_data(rdr, radius, density, prob, rdc, rdi, drc, dri)
        return r

class RMRelationMap(RelationMap):
    def __init__(self, data: Optional[ndarray] = None, radii: Optional[ndarray] = None, masses: Optional[ndarray] = None, pres: int = 100):
        super().__init__('radius', 'mass', data, radii, masses, xunit=r'R$_\oplus$', yunit=r'M$_\oplus$', pres=pres)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        with pf.open(filename) as f:
            rmr = f['rmr'].data.byteswap().newbyteorder()
            rmc = f['rm_cdf'].data.byteswap().newbyteorder()
            rmi = f['rm_icdf'].data.byteswap().newbyteorder()
            mrc = f['mr_cdf'].data.byteswap().newbyteorder()
            mri = f['mr_icdf'].data.byteswap().newbyteorder()
            h = f['rmr'].header
            mass = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
            radius = h['CRVAL2'] + h['CDELT2']*arange(h['NAXIS2'])
            h = f['rm_icdf'].header
            prob = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
        r = cls()
        r.init_data(rmr, radius, mass, prob, rmc, rmi, mrc, mri)
        return r
