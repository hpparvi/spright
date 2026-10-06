#  Spright
#  Copyright (C) 2022-2026 Hannu Parviainen.
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

from pathlib import Path
from typing import Optional

import pandas as pd
from astropy.units.astrophys import M_jup, M_earth, R_jup, R_earth
from numpy import ones, transpose


def read_stpm(fname: Path, mask_bad: Optional[bool] = True, return_rho: Optional[bool] = False):
    """Read a small transiting planets around M dwarfs (STPM) catalogue.

    Parameters
    ----------
    fname : Path
        Path to the catalogue CSV file.
    mask_bad : bool, optional
        If ``True``, include only the planets with a relative mass uncertainty of 25% or
        smaller and a relative radius uncertainty of 8% or smaller.
    return_rho : bool, optional
        If ``True``, return the planet bulk densities instead of the planet masses.

    Returns
    -------
    names : ndarray
        Planet names.
    radii : list of ndarray
        Planet radii and their uncertainties [R_earth].
    masses or densities : list of ndarray
        Planet masses and their uncertainties [M_earth], or planet bulk densities and
        their uncertainties [g/cm^3] if ``return_rho`` is ``True``.

    Notes
    -----
    The uncertainties are the means of the lower and upper uncertainties given in the
    catalogue.
    """
    df = pd.read_csv(fname)
    df['eM_relative'] = 0.5*(df.euM_Mterra + df.edM_Mterra)/df.M_Mterra
    df['eR_relative'] = 0.5*(df.euR_Rterra + df.edR_Rterra)/df.R_Rterra

    if mask_bad:
        m = (df.eM_relative <= 0.25) & (df.eR_relative <= 0.08)
    else:
        m = ones(df.eM_relative.size, bool)

    planet_names = df[m]['Star'].values + ' ' + df[m]['Planet'].values
    radius_means = df[m].R_Rterra.values.copy()
    radius_uncertainties = df[['edR_Rterra', 'euR_Rterra']].mean(axis=1).values[m]

    if return_rho:
        density_means = df[m]['rho_gcm-3'].values.copy()
        density_uncertainties = df[['edrho_gcm-3', 'eurho_gcm-3']].mean(axis=1).values[m]
        return planet_names, [radius_means, radius_uncertainties], [density_means, density_uncertainties]
    else:
        mass_means = df[m].M_Mterra.values.copy()
        mass_uncertainties = df[['edM_Mterra', 'euM_Mterra']].mean(axis=1).values[m]
        return planet_names, [radius_means, radius_uncertainties], [mass_means, mass_uncertainties]


def read_tepcat(fname: Path, max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25):
    """Read a TEPCat catalogue.

    Reads the catalogue, converts the planet radii and masses from Jupiter to Earth
    units, and removes the brown dwarfs, the planets without a mass or radius estimate,
    and the planets with too uncertain a mass or radius.

    Parameters
    ----------
    fname : Path
        Path to the catalogue CSV file.
    max_rel_r_err : float, optional
        Maximum allowed relative radius uncertainty.
    max_rel_m_err : float, optional
        Maximum allowed relative mass uncertainty.

    Returns
    -------
    DataFrame
        Catalogue with the columns ``name``, ``r`` and ``rerr`` [R_earth], ``m`` and
        ``merr`` [M_earth], ``mstar`` [M_sun], ``teff`` [K], and ``teq`` [K].

    Notes
    -----
    The uncertainties are the means of the lower and upper uncertainties given in the
    catalogue.
    """
    df = pd.read_csv(fname)
    df = df[(df.M_b > 0.0) & (df.Type != 'BD')]
    ix = df.columns.get_loc('R_b')
    r = (df['R_b'].values*R_jup).to(R_earth).value
    rerr = (df.iloc[:, ix + 1: ix + 3].mean(1).values*R_jup).to(R_earth).value

    ix = df.columns.get_loc('M_b')
    m = (df['M_b'].values*M_jup).to(M_earth).value
    merr = (df.iloc[:, ix + 1: ix + 3].mean(1).values*M_jup).to(M_earth).value
    l = (merr > 0.0) & (rerr > 0.0)
    df = df[l]
    df = pd.DataFrame(transpose([df['System'].values, r[l], rerr[l], m[l], merr[l], df.M_A, df.Teff, df.Teq]),
                      columns='name r rerr m merr mstar teff teq'.split())
    numeric_columns = df.columns.drop('name')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df


def read_exoplanet_eu(fname, max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25):
    """Read an Exoplanet.eu catalogue.

    Reads the catalogue, converts the planet radii and masses from Jupiter to Earth
    units, and removes the unconfirmed planets, the planets without a radius, mass, or
    orbital period estimate, and the planets with too uncertain a mass or radius.

    Parameters
    ----------
    fname : Path
        Path to the catalogue CSV file.
    max_rel_r_err : float, optional
        Maximum allowed relative radius uncertainty.
    max_rel_m_err : float, optional
        Maximum allowed relative mass uncertainty.

    Returns
    -------
    DataFrame
        Catalogue with the columns ``name``, ``r`` and ``rerr`` [R_earth], ``m`` and
        ``merr`` [M_earth], ``period`` [d], ``mstar`` [M_sun], ``teff`` [K], and
        ``teq`` [K].

    Notes
    -----
    The uncertainties are the means of the lower and upper uncertainties given in the
    catalogue.
    """
    df = pd.read_csv(fname)
    df.dropna(subset=['radius', 'radius_error_min', 'mass', 'mass_error_min', 'orbital_period'], inplace=True)
    df = df[(df.planet_status == 'Confirmed')]
    r = (df.radius.values*R_jup).to(R_earth).value
    rerr = (df[['radius_error_min', 'radius_error_max']].mean(1).values * R_jup).to(R_earth).value
    m = (df.mass.values*M_jup).to(M_earth).value
    merr = (df[['mass_error_min', 'mass_error_max']].mean(1).values * M_jup).to(M_earth).value
    df = pd.DataFrame(transpose([df.name.values, r, rerr, m, merr, df.orbital_period, df.star_mass, df.star_teff, df.temp_calculated]),
                      columns='name r rerr m merr period mstar teff teq'.split())
    numeric_columns = df.columns.drop('name')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df
