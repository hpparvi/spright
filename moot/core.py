from typing import Optional

import astropy.units as u
import pandas as pd

from pathlib import Path
from numpy import pi, newaxis

root = Path(__file__).parent.resolve()

mearth = (5.9742e24 * u.kg).to(u.g).value
rearth = (6.371e6 * u.m).to(u.cm).value

def rho(r, m):
    return m/(4/3*pi*r**3)


def read_stpm(fname: Optional[Path] = None):
    fname = fname or root / 'data/stpm_220816.csv'
    df = pd.read_csv(fname)
    df['eM_relative'] = 0.5*(df.euM_Mterra + df.edM_Mterra)/df.M_Mterra
    df['eR_relative'] = 0.5*(df.euR_Rterra + df.edR_Rterra)/df.R_Rterra
    return df


def read_mr():
    mr = pd.read_csv(root / 'data/mrtable3.txt', delim_whitespace=True, header=0, skiprows=[1], index_col=0)
    mr.index.name = 'mass'
    mr.drop(['cold_h2/he', 'max_coll_strip'], axis=1, inplace=True)
    md = pd.DataFrame(rho(mr.values*rearth, mr.index.values[:,newaxis] * mearth))
    md.columns = mr.columns
    md.set_index(mr.index, inplace=True)
    return mr, md