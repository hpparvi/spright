from matplotlib.pyplot import subplots, setp
from numpy import argmin, linspace
from numpy.random import default_rng
from scipy.interpolate import RegularGridInterpolator
from docfigs import COLORS, INK, MUTED, D_LABEL, R_LABEL, load

rmr, rdm, pv = load()
m = rmr.rdmap
rng = default_rng(1)
radius, sigma, n = 1.8, 0.1, 5000

rs = rng.normal(radius, sigma, n)
us = rng.uniform(size=n)
rho = RegularGridInterpolator((m.x, m.probs), m.xy_icdf)((rs, us))

fig, axs = subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
axs[0].hist(rs, bins=40, color=COLORS[1], alpha=0.8)
setp(axs[0], xlabel=R_LABEL, yticks=[], title='1. Draw radii from the measurement')

for r, ls in zip((radius - sigma, radius, radius + sigma), (':', '-', '--')):
    axs[1].plot(m.probs, m.xy_icdf[argmin(abs(m.x - r))], c=INK, ls=ls, lw=1.2, label=f'r = {r:.1f}')
axs[1].plot(us[:150], rho[:150], '.', c=COLORS[1], ms=4, alpha=0.7)
setp(axs[1], xlabel='Uniform random number $u$', ylabel=D_LABEL, title='2. Map $(r, u)$ through the inverse CDF',
     ylim=(0, 10))
axs[1].legend(frameon=False, fontsize=7, title='Inverse CDF at', title_fontsize=7)

axs[2].hist(rho, bins=linspace(0, 10, 60), color=COLORS[1], alpha=0.8)
setp(axs[2], xlabel=D_LABEL, yticks=[], title='3. Predicted density samples')
for ax in axs:
    ax.title.set_fontsize(9)
