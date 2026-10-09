from matplotlib.pyplot import subplots, setp
from docfigs import CMAP, INK, R_LABEL, D_LABEL, load

rmr, rdm, pv = load()
m = rmr.rdmap
fig, axs = subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

axs[0].imshow(m._pmapc.T, origin='lower', aspect='auto', cmap=CMAP, extent=(m.x[0], m.x[-1], m.y[0], m.y[-1]))
axs[0].set_title(r'Probability map $p(\rho \mid r)$', fontsize=9)
axs[1].imshow(m.xy_cdf.T, origin='lower', aspect='auto', cmap=CMAP, extent=(m.x[0], m.x[-1], m.y[0], m.y[-1]))
axs[1].set_title(r'CDF $P(\rho^\prime < \rho \mid r)$', fontsize=9)
l = axs[2].imshow(m.xy_icdf.T, origin='lower', aspect='auto', cmap=CMAP, extent=(m.x[0], m.x[-1], 0, 1), vmin=0)
cs = axs[2].contour(m.x, m.probs, m.xy_icdf.T, levels=[1, 2, 3, 4, 6, 8], colors=INK, linewidths=0.6)
axs[2].clabel(cs, fontsize=7, fmt='%g')
axs[2].set_title(r'Inverse CDF $\rho(r, u)$', fontsize=9)
fig.colorbar(l, ax=axs[2], label=D_LABEL)

setp(axs, xlabel=R_LABEL, xlim=(0.5, 4.0))
setp(axs[:2], ylabel=D_LABEL)
axs[2].set_ylabel('Probability $u$')
