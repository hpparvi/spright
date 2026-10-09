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

import re

from pathlib import Path
from typing import Optional, Iterable

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.units.astrophys import M_jup, M_earth, R_jup, R_earth
from numpy import ones, transpose, arange, argsort, asarray, isfinite

from .core import root

_host_aliases = {'Gliese': 'GJ', 'Gl': 'GJ'}
_catalog_files = {'stpm': root / 'data/stpm_230202.csv',
                  'tepcat': root / 'data/TEPCat.csv',
                  'exoplanet_eu': root / 'data/exoplanet_eu.csv'}


def _teff_mask(teff, min_teff: Optional[float], max_teff: Optional[float]):
    """Select the rows whose host star effective temperature is inside the given limits.

    Parameters
    ----------
    teff : array_like
        Host star effective temperatures [K].
    min_teff : float or None
        Minimum effective temperature [K], inclusive. No lower limit if ``None``.
    max_teff : float or None
        Maximum effective temperature [K], inclusive. No upper limit if ``None``.

    Returns
    -------
    ndarray
        Boolean mask selecting the rows inside the limits.

    Notes
    -----
    A row without an effective temperature estimate fails a limit that is given, and passes
    if neither limit is given.
    """
    teff = asarray(teff, dtype='d')
    mask = ones(teff.size, bool)
    if min_teff is not None:
        mask &= teff >= min_teff
    if max_teff is not None:
        mask &= teff <= max_teff
    return mask


def read_stpm(fname: Path, mask_bad: Optional[bool] = True, return_rho: Optional[bool] = False,
              min_teff: Optional[float] = None, max_teff: Optional[float] = None):
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
    min_teff : float, optional
        Minimum host star effective temperature [K], inclusive. No lower limit by default.
    max_teff : float, optional
        Maximum host star effective temperature [K], inclusive. No upper limit by default.

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

    A planet whose host star has no effective temperature estimate is removed if either
    temperature limit is given.
    """
    df = pd.read_csv(fname)
    df['eM_relative'] = 0.5*(df.euM_Mterra + df.edM_Mterra)/df.M_Mterra
    df['eR_relative'] = 0.5*(df.euR_Rterra + df.edR_Rterra)/df.R_Rterra

    if mask_bad:
        m = (df.eM_relative <= 0.25) & (df.eR_relative <= 0.08)
    else:
        m = ones(df.eM_relative.size, bool)
    m = asarray(m) & _teff_mask(df.Teff_K.values, min_teff, max_teff)

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


def read_tepcat(fname: Path, max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25,
                min_teff: Optional[float] = None, max_teff: Optional[float] = None):
    """Read a TEPCat catalogue.

    Reads the catalogue, converts the planet radii and masses from Jupiter to Earth
    units, and removes the brown dwarfs, the planets without a mass or radius estimate,
    the planets with too uncertain a mass or radius, and the planets whose host star
    effective temperature is outside the given limits.

    Parameters
    ----------
    fname : Path
        Path to the catalogue CSV file.
    max_rel_r_err : float, optional
        Maximum allowed relative radius uncertainty.
    max_rel_m_err : float, optional
        Maximum allowed relative mass uncertainty.
    min_teff : float, optional
        Minimum host star effective temperature [K], inclusive. No lower limit by default.
    max_teff : float, optional
        Maximum host star effective temperature [K], inclusive. No upper limit by default.

    Returns
    -------
    DataFrame
        Catalogue with the columns ``name``, ``r`` and ``rerr`` [R_earth], ``m`` and
        ``merr`` [M_earth], ``period`` [d], ``mstar`` [M_sun], ``teff`` [K], ``teq`` [K],
        and ``ra`` and ``dec`` [deg].

    Notes
    -----
    The uncertainties are the means of the lower and upper uncertainties given in the
    catalogue.

    A planet whose host star has no effective temperature estimate is removed if either
    temperature limit is given.
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
    df = pd.DataFrame(transpose([df['System'].values, r[l], rerr[l], m[l], merr[l], df['Period(day)'], df.M_A,
                                 df.Teff, df.Teq, df['RA(deg)'], df['Dec(deg)']]),
                      columns='name r rerr m merr period mstar teff teq ra dec'.split())
    numeric_columns = df.columns.drop('name')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df[_teff_mask(df.teff.values, min_teff, max_teff)]


def read_exoplanet_eu(fname, max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25,
                      min_teff: Optional[float] = None, max_teff: Optional[float] = None):
    """Read an Exoplanet.eu catalogue.

    Reads the catalogue, converts the planet radii and masses from Jupiter to Earth
    units, and removes the unconfirmed planets, the planets without a radius, mass, or
    orbital period estimate, the planets with too uncertain a mass or radius, and the
    planets whose host star effective temperature is outside the given limits.

    Parameters
    ----------
    fname : Path
        Path to the catalogue CSV file.
    max_rel_r_err : float, optional
        Maximum allowed relative radius uncertainty.
    max_rel_m_err : float, optional
        Maximum allowed relative mass uncertainty.
    min_teff : float, optional
        Minimum host star effective temperature [K], inclusive. No lower limit by default.
    max_teff : float, optional
        Maximum host star effective temperature [K], inclusive. No upper limit by default.

    Returns
    -------
    DataFrame
        Catalogue with the columns ``name``, ``r`` and ``rerr`` [R_earth], ``m`` and
        ``merr`` [M_earth], ``period`` [d], ``mstar`` [M_sun], ``teff`` [K], ``teq`` [K],
        and ``ra`` and ``dec`` [deg].

    Notes
    -----
    The uncertainties are the means of the lower and upper uncertainties given in the
    catalogue.

    A planet whose host star has no effective temperature estimate is removed if either
    temperature limit is given. Exoplanet.eu does not give an effective temperature for
    all its host stars.
    """
    df = pd.read_csv(fname)
    df.dropna(subset=['radius', 'radius_error_min', 'mass', 'mass_error_min', 'orbital_period'], inplace=True)
    df = df[(df.planet_status == 'Confirmed')]
    r = (df.radius.values*R_jup).to(R_earth).value
    rerr = (df[['radius_error_min', 'radius_error_max']].mean(1).values * R_jup).to(R_earth).value
    m = (df.mass.values*M_jup).to(M_earth).value
    merr = (df[['mass_error_min', 'mass_error_max']].mean(1).values * M_jup).to(M_earth).value
    df = pd.DataFrame(transpose([df.name.values, r, rerr, m, merr, df.orbital_period, df.star_mass, df.star_teff,
                                 df.temp_calculated, df.ra, df.dec]),
                      columns='name r rerr m merr period mstar teff teq ra dec'.split())
    numeric_columns = df.columns.drop('name')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df[(df.rerr/df.r < max_rel_r_err) & (df.merr/df.m < max_rel_m_err)]
    return df[_teff_mask(df.teff.values, min_teff, max_teff)]


def normalize_planet_name(name: str) -> str:
    """Normalise a planet name to a common form shared by the supported catalogues.

    The catalogues write the same planet in different ways, such as ``K2-018b`` (TEPCat),
    ``K2-18 b`` (STPM), and ``KELT-3 Ab`` (Exoplanet.eu). The normalised name has the form
    ``<host> <planet letter>``.

    Parameters
    ----------
    name : str
        Planet name.

    Returns
    -------
    str
        Normalised planet name.

    Notes
    -----
    The normalisation

    - replaces underscores with spaces and removes extra whitespace,
    - separates the planet letter from the host name, and adds the letter ``b`` to a name
      that ends with a number and has no planet letter (the TEPCat convention),
    - removes the stellar component (``A``, ``B``, ``C``, or ``(AB)``) preceding the planet
      letter,
    - removes the zero padding from numbers, except from coordinate-based names, and
    - replaces the ``Gliese`` and ``Gl`` catalogue prefixes with ``GJ``.

    A name ending with a separate capital letter and no planet letter (such as
    ``HD 130948 B``) is assumed to be a companion named after its host and is left as it is.
    Different designations of the same host (such as ``HD 3167`` and ``K2-96``) are not
    recognised as the same planet.
    """
    s = re.sub(r'\s+', ' ', name.replace('_', ' ')).strip()

    if (m := re.match(r'^(.*\d|.*\(AB\)|.*[ \d][ABC])([b-z])$', s)) or (m := re.match(r'^(.*) ([b-z])$', s)):
        host, letter = m.group(1).strip(), m.group(2)
        host = re.sub(r'\s*\(AB\)$', '', host)
        host = re.sub(r'(?<=\d)[ABC]$| [ABC]$', '', host)
    elif m := re.match(r'^(.*\d)[ABC]?$', s):
        host, letter = m.group(1), 'b'
    else:
        return s

    if not re.search(r'\d{4}[+-]\d{2,}', host):
        host = re.sub(r'(?<=[A-Za-z0-9][- ])0+(?=\d)', '', host)
    prefix, _, rest = host.partition(' ')
    if rest and prefix in _host_aliases:
        host = f'{_host_aliases[prefix]} {rest}'
    return f'{host} {letter}'


def _match_planets(df: pd.DataFrame, max_separation: Optional[float], max_rel_period_diff: float):
    """Group the rows of a combined catalogue that refer to the same planet.

    Two rows from different catalogues are taken to be the same planet if their normalised
    names match ignoring the letter case, spaces, and hyphens, or if their host stars are
    close to each other in the sky and their orbital periods agree.

    Parameters
    ----------
    df : DataFrame
        Combined catalogue with the columns ``name``, ``period``, ``ra``, ``dec``, and
        ``catalog``.
    max_separation : float or None
        Maximum angular separation between the host stars in arcseconds. The planets are
        matched using only their names if ``None``.
    max_rel_period_diff : float
        Maximum relative difference between the orbital periods.

    Returns
    -------
    ndarray
        Group label for each row. The label is the index of the first row of the group.

    Notes
    -----
    A group never contains two rows from the same catalogue: a match that would bring two
    such rows together is skipped. The position matches are applied first, from the smallest
    separation to the largest, and the name matches after them. A position match so overrides
    a conflicting name match, which happens when the catalogues give the planets of a system
    different letters.
    """
    labels = arange(df.shape[0])
    members = {i: {c} for i, c in enumerate(df.catalog.values)}

    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i

    def union(i, j):
        i, j = sorted((find(i), find(j)))
        if i != j and not (members[i] & members[j]):
            labels[j] = i
            members[i] |= members.pop(j)

    if max_separation is not None:
        ok = isfinite(df.ra.values) & isfinite(df.dec.values) & (df.period.values > 0)
        rows = arange(df.shape[0])[ok]
        sc = SkyCoord(df.ra.values[ok] * u.deg, df.dec.values[ok] * u.deg)
        i1, i2, sep, _ = search_around_sky(sc, sc, max_separation * u.arcsec)
        p = df.period.values[ok]
        m = (i1 < i2) & (abs(p[i1] - p[i2]) <= max_rel_period_diff * p[i1])
        for k in argsort(sep[m]):
            union(rows[i1[m][k]], rows[i2[m][k]])

    key = df.name.str.lower().str.replace(r'[ -]', '', regex=True)
    for ix in key.groupby(key).indices.values():
        for j in ix[1:]:
            union(ix[0], j)

    return pd.Series(labels).map(find).values


def read_combined(catalogs: Iterable[str] = ('stpm', 'tepcat', 'exoplanet_eu'),
                  max_rel_r_err: float = 0.08, max_rel_m_err: float = 0.25,
                  files: Optional[dict[str, Path]] = None,
                  max_separation: Optional[float] = 60.0, max_rel_period_diff: float = 0.01,
                  min_teff: Optional[float] = None, max_teff: Optional[float] = None):
    """Read and combine any of the STPM, TEPCat, and Exoplanet.eu catalogues.

    Reads the chosen catalogues, identifies the planets found in several catalogues, gives
    each planet the same name in all the catalogues, and concatenates the catalogues into a
    single table. A planet found in several catalogues has one row for each catalogue.
    `RMEstimator` treats the rows sharing a name as alternative measurements of the same
    planet.

    Two rows from different catalogues are taken to be the same planet if their normalised
    names match, or if their host stars are within ``max_separation`` from each other in the
    sky and their orbital periods agree within ``max_rel_period_diff``. The latter identifies
    the planets whose host stars have a different designation in different catalogues (such
    as ``HD 15337`` and ``TOI-402``).

    Parameters
    ----------
    catalogs : iterable of {'stpm', 'tepcat', 'exoplanet_eu'}, optional
        Catalogues to combine, or a single catalogue name. All three are combined by default.
    max_rel_r_err : float, optional
        Maximum allowed relative radius uncertainty.
    max_rel_m_err : float, optional
        Maximum allowed relative mass uncertainty.
    files : dict, optional
        Paths to the catalogue CSV files keyed by the catalogue name. The catalogue files
        shipped with the package are used for the catalogues without an entry.
    max_separation : float, optional
        Maximum angular separation between the host stars in arcseconds for two planets to
        be matched by their positions and orbital periods. The planets are matched using
        only their names if ``None``.
    max_rel_period_diff : float, optional
        Maximum relative difference between the orbital periods for two planets to be matched
        by their positions and orbital periods.
    min_teff : float, optional
        Minimum host star effective temperature [K], inclusive. No lower limit by default.
    max_teff : float, optional
        Maximum host star effective temperature [K], inclusive. No upper limit by default.

    Returns
    -------
    DataFrame
        Combined catalogue with the columns ``name``, ``r`` and ``rerr`` [R_earth], ``m`` and
        ``merr`` [M_earth], ``period`` [d], ``mstar`` [M_sun], ``teff`` [K], ``ra`` and
        ``dec`` [deg], ``catalog``, and ``catalog_name``, the normalised name of the planet
        in its catalogue.

    Raises
    ------
    ValueError
        If no catalogues are given, or if a catalogue name is not recognised or is repeated.

    Notes
    -----
    The names are matched ignoring the letter case, spaces, and hyphens in the normalised
    names (see `normalize_planet_name`). A planet is given the name it has in the first
    catalogue it is found in, where the catalogues are searched in the order they are given.
    Two rows from the same catalogue are never taken to be the same planet.

    The host star coordinates are not consistent between the catalogues, and can differ by
    tens of arcseconds for a star with a high proper motion. The default maximum separation
    is loose because of this, and it is the agreement of the orbital periods that makes the
    match reliable. A match by position and period takes precedence over a match by name,
    because the catalogues do not always give the planets of a system the same letters.

    The catalogues cover different host stars: STPM contains only M dwarfs, while TEPCat and
    Exoplanet.eu contain all the spectral types. Use ``min_teff`` and ``max_teff``, or the
    ``mstar`` column, to select a consistent sample.

    The temperature limits are applied to each row before the planets are matched, so a planet
    is kept only in the catalogues that place its host star inside the limits. A planet whose
    host star has no effective temperature estimate is removed if either limit is given.

    Examples
    --------
    >>> df = read_combined(['stpm', 'tepcat'], max_teff=4000)
    >>> rme = RMEstimator(names=df.name.values, radii=(df.r.values, df.rerr.values),
    ...                   masses=(df.m.values, df.merr.values))
    """
    catalogs = [catalogs] if isinstance(catalogs, str) else list(catalogs)
    files = {**_catalog_files, **(files or {})}
    if not catalogs:
        raise ValueError('At least one catalogue is needed.')
    if unknown := set(catalogs) - set(_catalog_files):
        raise ValueError(f'Unknown catalogues {sorted(unknown)}, the catalogues must be chosen from '
                         f'{list(_catalog_files)}.')
    if len(set(catalogs)) < len(catalogs):
        raise ValueError('Each catalogue can be given only once.')

    columns = 'name r rerr m merr period mstar teff ra dec'.split()

    def read(catalog):
        fname = files[catalog]
        if catalog == 'stpm':
            # read_stpm is called unfiltered on purpose: the rows of the arrays it returns are
            # matched one by one with the rows of the catalogue file read below. The quality and
            # temperature cuts are applied to the combined table instead.
            names, (r, rerr), (m, merr) = read_stpm(fname, mask_bad=False)
            host = pd.read_csv(fname)
            # Some of the declinations in the catalogue file begin with a stray '='.
            sc = SkyCoord(host.RA_J2000.values, host.DE_J2000.str.lstrip('=').values, unit=(u.hourangle, u.deg))
            return pd.DataFrame({'name': names, 'r': r, 'rerr': rerr, 'm': m, 'merr': merr,
                                 'period': host.Porb_d.values, 'mstar': host.M_Msol.values,
                                 'teff': host.Teff_K.values, 'ra': sc.ra.deg, 'dec': sc.dec.deg})
        elif catalog == 'tepcat':
            return read_tepcat(fname, max_rel_r_err, max_rel_m_err)[columns]
        else:
            return read_exoplanet_eu(fname, max_rel_r_err, max_rel_m_err)[columns]

    df = pd.concat([read(c).assign(catalog=c) for c in catalogs], ignore_index=True)
    mask = ((df.rerr / df.r < max_rel_r_err) & (df.merr / df.m < max_rel_m_err)).values
    df = df[mask & _teff_mask(df.teff.values, min_teff, max_teff)].reset_index(drop=True)
    df['name'] = df['catalog_name'] = df.name.map(normalize_planet_name)
    df['name'] = df.name.groupby(_match_planets(df, max_separation, max_rel_period_diff)).transform('first')
    return df
