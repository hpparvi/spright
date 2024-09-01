from pathlib import Path
from typing import Optional

import pandas as pd
from astropy.units.astrophys import M_jup, M_earth, R_jup, R_earth
from numpy import ones, transpose


def read_stpm(fname: Path, mask_bad: Optional[bool] = True, return_rho: Optional[bool] = False):
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
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df


def read_exoplanet_eu(fname, max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25):
    df = pd.read_csv(fname)
    df.dropna(subset=['radius', 'radius_error_min', 'mass', 'mass_error_min', 'orbital_period'], inplace=True)
    df = df[(df.planet_status == 'Confirmed')]
    r = (df.radius.values*R_jup).to(R_earth).value
    rerr = (df[['radius_error_min', 'radius_error_max']].mean(1).values * R_jup).to(R_earth).value
    m = (df.mass.values*M_jup).to(M_earth).value
    merr = (df[['mass_error_min', 'mass_error_max']].mean(1).values * M_jup).to(M_earth).value
    df = pd.DataFrame(transpose([df.name.values, r, rerr, m, merr, df.orbital_period, df.star_mass, df.star_teff, df.temp_calculated]),
                      columns='name r rerr m merr period mstar teff teq'.split())
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df
