from matplotlib.pyplot import subplots, setp
from numpy import linspace, meshgrid, ones
from spright.analytical_model import model
from docfigs import CMAP, NAMES, R_LABEL, D_LABEL, load, tables

rmr, rdm, pv = load()
radii, densities = linspace(0.6, 3.4, 250), linspace(0.05, 12, 250)
rg, dg = meshgrid(radii, densities)
pdf = model(dg.ravel(), rg.ravel(), pv, ones(3), *tables(rdm)).reshape((3, 250, 250))
extent = (radii[0], radii[-1], densities[0], densities[-1])

fig, axs = subplots(1, 4, figsize=(11, 3.2), sharey='all', constrained_layout=True)
vmax = pdf.sum(0).max()
for ax, p, name in zip(axs, list(pdf) + [pdf.sum(0)], NAMES + ('Full model',)):
    l = ax.imshow(p, extent=extent, origin='lower', aspect='auto', cmap=CMAP, vmin=0, vmax=vmax)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel(R_LABEL)
axs[0].set_ylabel(D_LABEL)
fig.colorbar(l, ax=axs, label=r'$p(\rho \mid r, \theta)$', pad=0.01)
