from pathlib import Path
from spright.io import read_stpm, read_tepcat, read_exoplanet_eu

root = Path(__file__).parent

def test_read_stmp():
    read_stpm(root / '../spright/data/stpm_230202.csv')

def test_read_tepcat():
    read_tepcat(root / '../spright/data/TEPCat.csv')

def test_read_exoeu():
    read_exoplanet_eu(root / '../spright/data/exoplanet_eu.csv')
