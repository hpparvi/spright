from matplotlib.pyplot import subplots, setp
from numpy import array, percentile
from docfigs import COLORS, NAMES, R_LABEL, load, radius_grid, transition_radii, weights

rmr, rdm, pv = load()
ps = rmr.posterior_samples.values[:2000]
r = radius_grid(200, 0.8, 3.2)

w = array([weights(r, p[0], *transition_radii(*p[:4]), p[1]) for p in ps])   # [sample, component, radius]

fig, ax = subplots(figsize=(7, 3.4), constrained_layout=True)
for i in range(3):
    lo, hi = percentile(w[:, i], [16, 84], axis=0)
    ax.fill_between(r, lo, hi, color=COLORS[i], alpha=0.12, lw=0)
    ax.plot(r, w[:, i].mean(0), c=COLORS[i], lw=2, label=NAMES[i])
setp(ax, xlabel=R_LABEL, ylabel='Class probability', xlim=(r[0], r[-1]), ylim=(0, 1))
ax.legend(frameon=False, fontsize=8, loc='center right')
