from pathlib import Path
from typing import Optional

import pandas as pd
from astropy import units as u
from numpy import ones


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


def read_tepcat(fname: Path):
    df = pd.read_csv(fname, delim_whitespace=True)
    ix = df.columns.get_loc('R_b')
    r = (df['R_b'].values*u.R_jup).to(u.R_earth).value
    rerr = (df.iloc[:, ix + 1: ix + 3].mean(1).values*u.R_jup).to(u.R_earth).value

    ix = df.columns.get_loc('M_b')
    m = (df['M_b'].values*u.M_jup).to(u.M_earth).value
    merr = (df.iloc[:, ix + 1: ix + 3].mean(1).values*u.M_jup).to(u.M_earth).value

    return df['System'].values, (r, rerr), (m, merr)
