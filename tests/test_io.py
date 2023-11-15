from pathlib import Path
from spright.io import read_stpm, read_tepcat

root = Path(__file__).parent

def test_read_stmp():
    read_stpm(root / '../spright/data/stpm_230202.csv')

def test_read_tepcat():
    read_tepcat(root / '../spright/data/TEPCat_FGK_20230522.csv')