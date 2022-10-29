import pandas as pd

from pathlib import Path
from numpy import pi

root = Path(__file__).parent.resolve()


def rho(r, m):
    return m/(4/3*pi*r**3)


def read_stpm():
    df = pd.read_csv(root/'data/stpm_220816.csv')
    df['eM_relative'] = 0.5*(df.euM_Mterra + df.edM_Mterra)/df.M_Mterra
    df['eR_relative'] = 0.5*(df.euR_Rterra + df.edR_Rterra)/df.R_Rterra
    return df
