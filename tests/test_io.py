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
from spright.io import read_stpm, read_tepcat, read_exoplanet_eu

root = Path(__file__).parent

def test_read_stmp():
    read_stpm(root / '../spright/data/stpm_230202.csv')

def test_read_tepcat():
    read_tepcat(root / '../spright/data/TEPCat.csv')

def test_read_exoeu():
    read_exoplanet_eu(root / '../spright/data/exoplanet_eu.csv')
