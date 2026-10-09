"""Shared helpers for the documentation figures."""
from matplotlib.colors import LinearSegmentedColormap
from numpy import abs, clip, linspace, median

from spright import RMRelation
from spright.rdmodel import RadiusDensityModel

# Component colours and names in the (rocky, water, puffy) order used throughout Spright.
COLORS = ('#eb6834', '#2a78d6', '#1baf7a')
NAMES = ('Rocky planets', 'Water worlds', 'Sub-Neptunes')
INK = '#0b0b0b'
MUTED = '#898781'

# Single-hue sequential colour map for the probability maps.
CMAP = LinearSegmentedColormap.from_list('spright_blue', ['#fcfcfb', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])

R_LABEL = r'Radius [R$_\oplus$]'
D_LABEL = r'Density [g cm$^{-3}$]'
M_LABEL = r'Mass [M$_\oplus$]'


def load(model: str = 'stpm'):
    """Returns a shipped relation, its theoretical density model, and its median parameter vector."""
    rmr = RMRelation(model)
    rdm = RadiusDensityModel('z19', 'a21' if model.endswith('a21') else 'z19')
    return rmr, rdm, median(rmr.posterior_samples.values, 0)


def transition_radii(r1, r4, ww, ws):
    """Calculates the inner transition radii r2 and r3 from the sampling parameters."""
    d = r4 - r1
    a = 0.5 - abs(ww - 0.5)
    return r1 + d * (1.0 - ww + ws * a), r1 + d * (ww + ws * a)


def weights(r, r1, r2, r3, r4):
    """Calculates the rocky, water-world, and sub-Neptune mixture weights."""
    x = clip((r - r3) / max(r4 - r3, 1e-4), 0.0, 1.0)
    y = clip(clip((r - r1) / max(r2 - r1, 1e-4), 0.0, 1.0) - x, 0.0, 1.0)
    return 1.0 - x - y, y, x


def tables(rdm):
    """Unpacks a RadiusDensityModel into the arguments taken by the numba model functions."""
    return (rdm._rr0, rdm._rdr, rdm._rx0, rdm._rdx, rdm.drocky,
            rdm._wr0, rdm._wdr, rdm._wx0, rdm._wdx, rdm.dwater)


def radius_grid(n: int = 300, rmin: float = 0.6, rmax: float = 3.4):
    return linspace(rmin, rmax, n)
