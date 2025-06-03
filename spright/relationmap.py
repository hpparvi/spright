import astropy.io.fits as pf

from pathlib import Path
from typing import Optional, Union

from matplotlib.pyplot import subplots, setp
from numpy import ndarray, zeros, isfinite, arange, linspace, inf, diff, clip
from numpy.random import uniform
from scipy.interpolate import RegularGridInterpolator

from .model import invert_cdf
from .util import sample_distribution


class RelationMap:
    """
    A class representing a relation between two variables with inverse cumulative distribution functions (ICDF).
    """

    def __init__(self, xname: str, yname: str,
                 data: Optional[ndarray] = None, x: Optional[ndarray] = None, y: Optional[ndarray] = None,
                 xunit: str = '', yunit: str = '', pres: int = 100):
        """
        Initializes a RelationMap instance with the specified parameters.

        Parameters
        ----------
        xname : str
            The name of the x-axis variable.
        yname : str
            The name of the y-axis variable.
        data : ndarray, optional
            The data array representing the relation map.
        x : ndarray, optional
            The x-axis values.
        y : ndarray, optional
            The y-axis values.
        xunit : str, optional
            The unit of the x-axis variable.
        yunit : str, optional
            The unit of the y-axis variable.
        pres : int, optional
            The number of points in the probability space.
        """

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
        self.xy_icdf_c: Optional[ndarray] = None

        self.yx_cdf: Optional[ndarray] = None
        self.yx_icdf: Optional[ndarray] = None
        self.yx_icdf_c: Optional[ndarray] = None

        self._pmapc: Optional[ndarray] = None
        self._pmapf: Optional[ndarray] = None
        self.x: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.xres: int = 0
        self.yres: int = 0

        if data is not None:
            self.init_data(data, x, y, self.probs)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        """
        Class method (not implemented) to load a RelationMap instance from a file.

        Parameters
        ----------
        filename : Union[Path, str]
            The path or filename to load the RelationMap from.
        """
        raise NotImplementedError

    def init_data(self, data: ndarray, x: ndarray, y: ndarray, probs: ndarray,
                  xy_cdf: Optional[ndarray] = None, xy_icdf: Optional[ndarray] = None,
                  yx_cdf: Optional[ndarray] = None, yx_icdf: Optional[ndarray] = None):
        """
        Initializes the data, x, y, and probability arrays, and optionally the ICDF arrays.

        Parameters
        ----------
        data : ndarray
            The data array representing the relation map.
        x : ndarray
            The x-axis values.
        y : ndarray
            The y-axis values.
        probs : ndarray
            The array of probabilities.
        xy_cdf : Optional[ndarray], optional
            The cumulative distribution function (CDF) for the xy relation.
        xy_icdf : Optional[ndarray], optional
            The inverse cumulative distribution function (ICDF) for the xy relation.
        yx_cdf : Optional[ndarray], optional
            The cumulative distribution function (CDF) for the yx relation.
        yx_icdf : Optional[ndarray], optional
            The inverse cumulative distribution function (ICDF) for the yx relation.
        """
        self._pmapc = data.copy()           # Component-wise data as a 3D ndarray
        self._pmapf = self._pmapc.sum(0)    # Flattened data as a 2D ndarray
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
        """
        Creates cumulative distribution function (CDF) and ICDF maps for the xy relation.
        """
        self.probs = linspace(0, 1, self.pres)
        cdf = self._pmapf.cumsum(axis=1)
        cdf /= cdf[:, -1:]
        icdf = zeros((self.xres, self.pres))
        for i in range(self.xres):
            _, icdf[i] = invert_cdf(self.y, cdf[i], self.pres)
        self.xy_cdf = cdf
        self.xy_icdf = icdf

    def _create_separated_xy_maps(self):
        """
        Creates cumulative distribution function (CDF) and ICDF maps for the xy relation.
        """
        self.probs = linspace(0, 1, self.pres)
        self.xy_icdf_c = zeros((3, self.xres, self.pres))
        for ic in range(3):
            cdf = self._pmapc[ic].cumsum(axis=1)
            cdf /= cdf[:, -1:]
            for i in range(self.xres):
                _, self.xy_icdf_c[i] = invert_cdf(self.y, cdf[i], self.pres)
            self.xy_icdf_c[ic, :, 0] = clip(self.xy_icdf_c[ic, :, 1] - diff(self.xy_icdf_c[ic, :, 1:3]).ravel(), 0.0, inf)
            self.xy_icdf_c[ic, :, -1] = self.xy_icdf_c[ic, :, -2] + diff(self.xy_icdf_c[ic, :, -3:-1]).ravel()

    def _create_yx_maps(self):
        """
        Creates cumulative distribution function (CDF) and ICDF maps for the yx relation.
        """
        self.probs = linspace(0, 1, self.pres)
        cdf = self._pmapf.cumsum(axis=0).T
        cdf /= cdf[:, -1:]
        icdf = zeros((self.yres, self.pres))
        for i in range(self.yres):
            _, icdf[i] = invert_cdf(self.x, cdf[i], self.pres)
        self.yx_cdf = cdf
        self.yx_icdf = icdf

    def _create_separated_yx_maps(self):
        """
        Creates cumulative distribution function (CDF) and ICDF maps for the separated yx relation.

        This method computes the cumulative distribution function (CDF) and inverse cumulative distribution
        function (ICDF) maps for the separated yx relation, where each component of the relation is treated
        separately.

        The resulting ICDF maps are stored in the 'yx_icdf_c' attribute.

        Notes
        -----
        The 'yx_icdf_c' attribute is a 3D array representing the ICDF maps for the separated yx relation.
        The first dimension corresponds to the component index, and the second and third dimensions correspond
        to the y-axis resolution and the probability resolution, respectively.

        The computed ICDF maps are adjusted at the boundaries to ensure validity.

        """
        self.probs = linspace(0, 1, self.pres)
        self.yx_icdf_c = zeros((3, self.yres, self.pres))
        for ic in range(3):
            cdf = self._pmapc[ic].cumsum(axis=0).T
            cdf /= cdf[:, -1:]
            for i in range(self.yres):
                _, self.yx_icdf_c[i] = invert_cdf(self.x, cdf[i], self.pres)
            self.yx_icdf_c[ic, :, 0] = clip(self.yx_icdf_c[ic, :, 1] - diff(self.yx_icdf_c[ic, :, 1:3]).ravel(), 0.0, inf)
            self.yx_icdf_c[ic, :, -1] = self.yx_icdf_c[ic, :, -2] + diff(self.yx_icdf_c[ic, :, -3:-1]).ravel()

    def sample(self, v: tuple[float, float], relation: str, nsamples: int = 20_000) -> (ndarray, ndarray):
        """
        Samples from the relation map for the given relation and number of samples.

        Parameters
        ----------
        v : tuple[float, float]
            A tuple representing the mean and standard deviation for sampling the x-axis variable.
        relation : str
            The type of relation ('xy' or 'yx') to sample from.
        nsamples : int, optional
            The number of samples to generate.

        Returns
        -------
        tuple
            A tuple containing two arrays representing the sampled x and y values.
        """
        if relation not in self.relations:
            raise ValueError(f"Relation needs to be on of {list(self.relations.keys())}")
        relation = self.relations[relation]
        if relation == 'xy':
            rgi = RegularGridInterpolator((self.x, self.probs), self.xy_icdf, bounds_error=False)
        else:
            rgi = RegularGridInterpolator((self.y, self.probs), self.yx_icdf, bounds_error=False)
        vs = sample_distribution(v, nsamples)
        samples = rgi((vs, uniform(size=nsamples)))
        m = isfinite(samples)
        return vs[m], samples[m]

    def sample_separated(self, v: tuple[float, float], pvs: ndarray, relation: str, nsamples: int = 20_000) -> (ndarray, ndarray):
        """
        Samples from the relation map for the given relation and number of samples.

        Parameters
        ----------
        v : tuple[float, float]
            A tuple representing the mean and standard deviation for sampling the x-axis variable.
        relation : str
            The type of relation ('xy' or 'yx') to sample from.
        nsamples : int, optional
            The number of samples to generate.

        Returns
        -------
        tuple
            A tuple containing two arrays representing the sampled x and y values.
        """
        if relation not in self.relations:
            raise ValueError(f"Relation needs to be on of {list(self.relations.keys())}")
        relation = self.relations[relation]
        if relation == 'xy':
            rgi = RegularGridInterpolator((self.x, self.probs), self.xy_icdf, bounds_error=False)
        else:
            rgi = RegularGridInterpolator((self.y, self.probs), self.yx_icdf, bounds_error=False)
        vs = sample_distribution(v, nsamples)
        samples = rgi((vs, uniform(size=nsamples)))
        m = isfinite(samples)
        return vs[m], samples[m]


    def plot_map(self, ax=None, cm=None, norm=None):
        """
        Plots the data map.

        Parameters
        ----------
        ax : Optional, optional
            The axes on which to plot. If None, a new figure and axes will be created.
        cm : Optional, optional
            The colormap to use for plotting.
        """
        if ax is None:
            fig, ax = subplots()
        ax.imshow(self._pmapf.T, origin='lower', aspect='auto', cmap=cm, norm=norm, interpolation='bicubic',
                  extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        setp(ax, xlabel=f"{self.xname} [{self.xunit}]", ylabel=f"{self.yname} [{self.yunit}]")

    def plot_cdf(self, relation: str = 'xy', ax=None):
        """
        Plots the cumulative distribution function (CDF) map for the specified relation.

        Parameters
        ----------
        relation : str, optional
            The type of relation ('xy' or 'yx') to plot.
        ax : Optional, optional
            The axes on which to plot. If None, a new figure and axes will be created.
        """
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
        """
        Plots the inverse cumulative distribution function (ICDF) map for the specified relation.

        Parameters
        ----------
        relation : str, optional
            The type of relation ('xy' or 'yx') to plot.
        ax : Optional, optional
            The axes on which to plot. If None, a new figure and axes will be created.
        """
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
    """
    A subclass of RelationMap representing a relation map between radius and density.

    Methods
    -------
    __init__(data=None, radii=None, densities=None, pres=100)
        Initializes an RDRelationMap instance with the specified parameters.

    load(filename)
        Class method to load an RDRelationMap instance from a file.
    """

    def __init__(self, data: Optional[ndarray] = None, radii: Optional[ndarray] = None, densities: Optional[ndarray] = None, pres: int = 100):
        super().__init__('radius', 'density', data, radii, densities, xunit=r'R$_\oplus$', yunit=r'g/cm$^3$', pres=pres)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        with pf.open(filename) as f:
            rdr = f[0].data
            rdc = f['rd_cdf'].data
            rdi = f['rd_icdf'].data
            drc = f['dr_cdf'].data
            dri = f['dr_icdf'].data
            h = f[0].header
            density = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
            radius = h['CRVAL2'] + h['CDELT2']*arange(h['NAXIS2'])
            h = f['rd_icdf'].header
            prob = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
        r = cls()
        r.init_data(rdr, radius, density, prob, rdc, rdi, drc, dri)
        return r

class RMRelationMap(RelationMap):
    """
    A subclass of RelationMap representing a relation map between radius and mass.

    Methods
    -------
    __init__(data=None, radii=None, masses=None, pres=100)
        Initializes an RMRelationMap instance with the specified parameters.

    load(filename)
        Class method to load an RMRelationMap instance from a file.
    """

    def __init__(self, data: Optional[ndarray] = None, radii: Optional[ndarray] = None, masses: Optional[ndarray] = None, pres: int = 100):
        super().__init__('radius', 'mass', data, radii, masses, xunit=r'R$_\oplus$', yunit=r'M$_\oplus$', pres=pres)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        with pf.open(filename) as f:
            rmr = f['rmr'].data
            rmc = f['rm_cdf'].data
            rmi = f['rm_icdf'].data
            mrc = f['mr_cdf'].data
            mri = f['mr_icdf'].data
            h = f['rmr'].header
            mass = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
            radius = h['CRVAL2'] + h['CDELT2']*arange(h['NAXIS2'])
            h = f['rm_icdf'].header
            prob = h['CRVAL1'] + h['CDELT1']*arange(h['NAXIS1'])
        r = cls()
        r.init_data(rmr, radius, mass, prob, rmc, rmi, mrc, mri)
        return r
