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

import pytest

from spright.io import read_stpm, read_tepcat, read_exoplanet_eu, read_combined, normalize_planet_name

root = Path(__file__).parent

def test_read_stmp():
    read_stpm(root / '../spright/data/stpm_230202.csv')

def test_read_tepcat():
    read_tepcat(root / '../spright/data/TEPCat.csv')

def test_read_exoeu():
    read_exoplanet_eu(root / '../spright/data/exoplanet_eu.csv')


@pytest.mark.parametrize('name, normalized', [
    ('K2-018b', 'K2-18 b'),              # TEPCat: zero padding and an attached planet letter
    ('WASP-019', 'WASP-19 b'),           # TEPCat: no planet letter
    ('55_Cnc_e', '55 Cnc e'),            # TEPCat: underscores
    ('HD_003167c', 'HD 3167 c'),
    ('TOI-858B', 'TOI-858 b'),           # TEPCat: stellar component without a planet letter
    ('KELT-3 Ab', 'KELT-3 b'),           # Exoplanet.eu: stellar component
    ('Kepler-16 (AB)b', 'Kepler-16 b'),  # Exoplanet.eu: circumbinary planet
    ('Gliese 12 b', 'GJ 12 b'),
    ('LTT 1445 A b', 'LTT 1445 b'),      # STPM: stellar component
    ('LTT_1445Ab', 'LTT 1445 b'),        # TEPCat: attached stellar component
    ('TRAPPIST-1 h', 'TRAPPIST-1 h'),
    ('2S 0918-549 b', '2S 0918-549 b'),  # Coordinate-based names keep their zeros
    ('HD 130948 B', 'HD 130948 B'),      # Companion named after its host
])
def test_normalize_planet_name(name, normalized):
    assert normalize_planet_name(name) == normalized


def test_read_combined():
    df = read_combined()
    assert list(df.columns) == 'name r rerr m merr period mstar teff ra dec catalog catalog_name'.split()
    assert set(df.catalog) == {'stpm', 'tepcat', 'exoplanet_eu'}
    assert not df.duplicated(['name', 'catalog']).any()
    assert set(df[df.name == 'L 98-59 c'].catalog) == {'stpm', 'tepcat', 'exoplanet_eu'}
    assert (df.rerr / df.r < 0.08).all() and (df.merr / df.m < 0.25).all()


@pytest.mark.parametrize('catalogs', [
    ['stpm'], ['tepcat'], ['exoplanet_eu'],
    ['stpm', 'tepcat'], ['stpm', 'exoplanet_eu'], ['tepcat', 'exoplanet_eu'],
    ['exoplanet_eu', 'stpm', 'tepcat'],
])
def test_read_combined_combinations(catalogs):
    df = read_combined(catalogs)
    assert list(df.catalog.unique()) == catalogs
    assert not df.duplicated(['name', 'catalog']).any()


def test_read_combined_single_catalogue():
    df = read_combined('stpm')
    names = read_stpm(root / '../spright/data/stpm_230202.csv')[0]
    assert set(df.catalog) == {'stpm'}
    assert df.shape[0] == names.size


def test_read_combined_files():
    df = read_combined(['stpm'], files={'stpm': root / '../spright/data/stpm_230202.csv'})
    assert df.equals(read_combined(['stpm']))


def test_read_combined_invalid_catalogues():
    for catalogs in ([], ['nasa'], ['stpm', 'stpm']):
        with pytest.raises(ValueError):
            read_combined(catalogs)


def test_read_combined_position_matching():
    df = read_combined()
    by_name = read_combined(max_separation=None)
    assert df.name.nunique() < by_name.name.nunique()
    assert not df.duplicated(['name', 'catalog']).any()

    # A host with a different designation in TEPCat (HD 15337) and Exoplanet.eu (TOI-402)
    planet = df[df.name == 'HD 15337 b']
    assert set(planet.catalog) == {'tepcat', 'exoplanet_eu'}
    assert set(planet.catalog_name) == {'HD 15337 b', 'TOI-402 b'}
    assert set(by_name[by_name.name == 'HD 15337 b'].catalog) == {'tepcat'}

    # Kepler-289 c and d have their letters swapped between TEPCat and Exoplanet.eu
    for name in ('Kepler-289 c', 'Kepler-289 d'):
        periods = df[df.name == name].period
        assert periods.size == 2
        assert periods.max() / periods.min() - 1 < 0.01


# Temperature limits that straddle the M dwarf / FGK boundary.
TEFF_LO, TEFF_HI = 3200.0, 4000.0

STPM = root / '../spright/data/stpm_230202.csv'
TEPCAT = root / '../spright/data/TEPCat.csv'
EXOEU = root / '../spright/data/exoplanet_eu.csv'


@pytest.mark.parametrize('reader, fname', [(read_tepcat, TEPCAT), (read_exoplanet_eu, EXOEU)])
def test_read_teff_limits(reader, fname):
    """The limits are inclusive, and keep exactly the rows inside them."""
    teff = reader(fname).teff

    assert reader(fname, min_teff=TEFF_LO).shape[0] == (teff >= TEFF_LO).sum() > 0
    assert reader(fname, max_teff=TEFF_HI).shape[0] == (teff <= TEFF_HI).sum() > 0

    df = reader(fname, min_teff=TEFF_LO, max_teff=TEFF_HI)
    assert df.shape[0] == ((teff >= TEFF_LO) & (teff <= TEFF_HI)).sum() > 0
    assert df.teff.min() >= TEFF_LO and df.teff.max() <= TEFF_HI

    # An exactly-inclusive bound keeps the row sitting on it.
    assert (reader(fname, min_teff=teff.max()).teff == teff.max()).all()

    # Omitting the limits changes nothing.
    assert reader(fname, min_teff=None, max_teff=None).equals(reader(fname))


def test_read_stpm_teff_limits():
    """read_stpm returns arrays rather than a table, so check the planet count and membership."""
    names = read_stpm(STPM)[0]
    limited = read_stpm(STPM, min_teff=TEFF_LO, max_teff=TEFF_HI)[0]
    assert 0 < limited.size < names.size
    assert set(limited) <= set(names)

    # STPM holds only M dwarfs (2566-4089 K), so these bracket the whole catalogue.
    assert read_stpm(STPM, min_teff=5000.0)[0].size == 0
    assert read_stpm(STPM, min_teff=1000.0, max_teff=1e4)[0].size == names.size

    # The limits compose with mask_bad rather than replacing it.
    assert read_stpm(STPM, mask_bad=False, max_teff=TEFF_HI)[0].size >= limited.size

    # The radii and masses stay aligned with the names.
    n, (r, rerr), (m, merr) = read_stpm(STPM, max_teff=TEFF_HI)
    assert n.size == r.size == rerr.size == m.size == merr.size


def test_read_teff_limits_drop_missing():
    """Exoplanet.eu has host stars without a temperature; a limit must remove them."""
    assert read_exoplanet_eu(EXOEU).teff.isna().any()
    assert not read_exoplanet_eu(EXOEU, min_teff=0.0).teff.isna().any()
    assert not read_exoplanet_eu(EXOEU, max_teff=1e6).teff.isna().any()


def test_read_combined_teff_limits():
    df = read_combined(min_teff=TEFF_LO, max_teff=TEFF_HI)
    assert df.teff.min() >= TEFF_LO and df.teff.max() <= TEFF_HI
    assert 0 < df.shape[0] < read_combined().shape[0]
    assert not df.duplicated(['name', 'catalog']).any()
    assert read_combined(min_teff=None, max_teff=None).equals(read_combined())

    # STPM holds only M dwarfs, so an FGK cut must leave it out entirely.
    assert 'stpm' not in set(read_combined(min_teff=5000.0).catalog)
