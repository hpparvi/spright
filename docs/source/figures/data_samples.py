from matplotlib.pyplot import subplots, setp
from numpy import linspace, meshgrid, ones
from spright.analytical_model import model
from docfigs import CMAP, INK, R_LABEL, D_LABEL, load, tables

rmr, rdm, pv = load()
nplanets = rmr.catalog.shape[0]
rs = rmr.rdsamples.radius.values.reshape((-1, nplanets))
ds = rmr.rdsamples.density.values.reshape((-1, nplanets))

radii, densities = linspace(0.6, 3.4, 250), linspace(0.05, 12, 250)
rg, dg = meshgrid(radii, densities)
pdf = model(dg.ravel(), rg.ravel(), pv, ones(3), *tables(rdm)).sum(0).reshape((250, 250))
extent = (radii[0], radii[-1], densities[0], densities[-1])

fig, axs = subplots(1, 2, figsize=(10, 3.8), sharex='all', sharey='all', constrained_layout=True)
for ax in axs:
    ax.imshow(pdf, extent=extent, origin='lower', aspect='auto', cmap=CMAP, alpha=0.8)
cat = rmr.catalog
axs[0].errorbar(cat.radius, cat.density, xerr=cat.radius_e, yerr=cat.density_e, fmt='o', ms=3, c=INK, lw=0.6)
axs[0].set_title('Catalogue: means and uncertainties', fontsize=9)
axs[1].plot(rs, ds, '.', ms=1.5, c=INK, alpha=0.35)
axs[1].set_title(f'What the likelihood sees: {rs.shape[0]} samples per planet', fontsize=9)
setp(axs, xlabel=R_LABEL, xlim=extent[:2], ylim=extent[2:])
axs[0].set_ylabel(D_LABEL)
