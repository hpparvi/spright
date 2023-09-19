from astropy import units as u
from numpy import ndarray, atleast_2d, pi
from numpy.random import uniform
from scipy.interpolate import RegularGridInterpolator

from .model import create_radius_density_icdf
from .rdmodel import RadiusDensityModel


def create_mock_sample(r: ndarray, pv: ndarray, quantity: str = 'mass') -> ndarray:
    rdm = RadiusDensityModel()
    radii, densities, probs, rdmap, icdf = create_radius_density_icdf(atleast_2d(pv),
                                                                      rdm._r0, rdm._dr, rdm.drocky, rdm.dwater,
                                                                      pres=300, rres=300, dres=300)
    rgi = RegularGridInterpolator((radii, probs), icdf, bounds_error=False)
    if quantity == 'density':
        return rgi((r, uniform(size=r.size)))
    elif quantity == 'mass':
        v = 4/3 * pi * (r*u.R_earth).to(u.cm)**3
        m_g = v * rgi((r, uniform(size=r.size))) * (u.g / u.cm**3)
        return m_g.to(u.M_earth).value
    else:
        raise ValueError("Quantity has to be either 'mass' or 'density'.")
