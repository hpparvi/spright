from matplotlib.pyplot import subplots, setp
from numpy import array, linspace
from docfigs import COLORS, NAMES, MUTED, R_LABEL, D_LABEL, load, radius_grid

rmr, rdm, pv = load()
r = radius_grid()
fig, axs = subplots(1, 3, figsize=(10, 3.4), sharey='all', constrained_layout=True)

for c in linspace(0, 1, 6):
    axs[0].plot(r, rdm.evaluate_rocky(c, r), c=COLORS[0], lw=1, alpha=0.35)
    axs[0].annotate(f'{c:.1f}', (r[105], rdm.evaluate_rocky(c, r[105:106])[0]), fontsize=7, color=MUTED,
                    ha='center', va='center', backgroundcolor='w')
axs[0].plot(r, rdm.evaluate_rocky(pv[4], r), c=COLORS[0], lw=2, label=f'$c_r$ = {pv[4]:.2f}')

for c in linspace(0.1, 1, 5):
    axs[1].plot(r, rdm.evaluate_water(c, r), c=COLORS[1], lw=1, alpha=0.35)
    axs[1].annotate(f'{c:.1f}', (r[150], rdm.evaluate_water(c, r[150:151])[0]), fontsize=7, color=MUTED,
                    ha='center', va='center', backgroundcolor='w')
axs[1].plot(r, rdm.evaluate_water(pv[5], r), c=COLORS[1], lw=2, label=f'$c_w$ = {pv[5]:.2f}')

for sp in (-2.0, -1.5, -0.5, 0.0):
    axs[2].plot(r, pv[6] * (r / 2) ** sp, c=COLORS[2], lw=1, alpha=0.35)
    axs[2].annotate(f'{sp:.1f}', (r[20], pv[6] * (r[20] / 2) ** sp), fontsize=7, color=MUTED,
                    ha='center', va='center', backgroundcolor='w')
axs[2].plot(r, pv[6] * (r / 2) ** pv[7], c=COLORS[2], lw=2, label=f'$s_p$ = {pv[7]:.2f}')
axs[2].plot(2.0, pv[6], 'o', c=COLORS[2], ms=6, mec='w')
axs[2].annotate(r'$i_p$', (2.0, pv[6]), xytext=(6, 6), textcoords='offset points')

for ax, name, par in zip(axs, NAMES, ('iron mass fraction', 'water mass fraction', 'density exponent')):
    ax.set_title(f'{name}\n(thin lines: {par})', fontsize=9)
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlabel(R_LABEL)
setp(axs, ylim=(0, 14), xlim=(r[0], r[-1]))
axs[0].set_ylabel(D_LABEL)
