import pandas as pd
import astropy.constants as c
import astropy.units as u

from pathlib import Path
from scipy.interpolate import interp1d
from numpy import pi, newaxis

from scipy.stats import multivariate_normal
from numpy import diag, dstack, meshgrid, linspace, zeros
from scipy.stats import gaussian_kde as gkde

root = Path(__file__).parent.resolve()


def rho(r, m):
    return m/(4/3*pi*r**3)


def read_mr_earthlike():
    df = pd.read_csv(root/'data/massradiusEarthlikeRocky.txt', delim_whitespace=True, header=None, index_col=0)
    df.columns = ['radius']
    df.index.name = 'mass'
    df['density'] = rho(df.radius.values*c.R_earth, df.index.values*c.M_earth).to(u.g/u.cm**3).value
    return df


def read_mr(normalize_to_earth: bool = False):
    mr = pd.read_csv(root/'data/mrtable3.txt', delim_whitespace=True, header=0, skiprows=[1], index_col=0)
    mr.index.name = 'mass'
    md = pd.DataFrame(rho(mr.values*c.R_earth, mr.index.values[:, newaxis]*c.M_earth).to(u.g/u.cm**3).value)
    md.columns = mr.columns
    md.set_index(mr.index, inplace=True)
    if normalize_to_earth:
        mr_earth = read_mr_earthlike()
        ip = interp1d(mr_earth.index.values, mr_earth.density)
        md = md/ip(md.index.values)[:, newaxis]
    return mr, md


def read_stpm():
    mr_earth = read_mr_earthlike()
    ip = interp1d(mr_earth.index.values, mr_earth.density)
    df = pd.read_csv(root/'data/stpm_220816.csv')
    for c in 'rho_gcm-3', 'eurho_gcm-3', 'edrho_gcm-3':
        df[c.replace('_gcm-3', '_rhoterra')] = df[c]/ip(df.M_Mterra)

    df['eM_relative'] = 0.5*(df.euM_Mterra + df.edM_Mterra)/df.M_Mterra
    df['eR_relative'] = 0.5*(df.euR_Rterra + df.edR_Rterra)/df.R_Rterra
    df['eD_relative'] = 0.5*(df.eurho_rhoterra + df.edrho_rhoterra)/df.rho_rhoterra
    return df


def create_rd_density_map(ns: int = 100, nxy: int = 100, bw: float = 0.2):
    df = read_stpm()
    df = df[df['rho_gcm-3'] > 0].copy()
    samples = []
    for _, p in df.iterrows():
        m, em = p.M_Mterra, 0.5*(p.euM_Mterra + p.edM_Mterra)
        r, er = p.R_Rterra, 0.5*(p.euR_Rterra + p.edR_Rterra)
        rho, erho = p.rho_rhoterra, 0.5*(p.eurho_rhoterra + p.edrho_rhoterra)
        d = multivariate_normal([r, rho], diag([er**2, erho**2]))
        samples.append(d.rvs(size=ns))
    samples = dstack(samples)

    xradius, xdensity = meshgrid(linspace(0.5, 3.0, nxy), linspace(0.0, 2.0, nxy))
    gkvs = zeros((ns, nxy, nxy))

    for i in range(ns):
        gk = gkde(samples[i])
        gk.set_bandwidth(bw)
        gkvs[i] = gk((xradius.ravel(), xdensity.ravel())).reshape([nxy, nxy])
    return gkvs
