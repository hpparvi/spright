from matplotlib.pyplot import subplots, setp
from docfigs import COLORS, NAMES, MUTED, R_LABEL, radius_grid, transition_radii, weights

r = radius_grid(400, 0.8, 3.2)
r1, r4 = 1.2, 2.8
cases = [(0.2, 0.0), (0.5, 0.0), (0.8, 0.0), (0.5, -0.8), (0.5, 0.8), (0.8, 0.8)]

fig, axs = subplots(2, 3, figsize=(10, 4.6), sharex='all', sharey='all', constrained_layout=True)
for ax, (ww, ws) in zip(axs.flat, cases):
    r2, r3 = transition_radii(r1, r4, ww, ws)
    for w, c, name in zip(weights(r, r1, r2, r3, r4), COLORS, NAMES):
        ax.plot(r, w, c=c, lw=2, label=name)
    for x, label in zip((r1, r2, r3, r4), ('$r_1$', '$r_2$', '$r_3$', '$r_4$')):
        ax.axvline(x, c=MUTED, lw=0.7, ls=':')
        ax.text(x, 1.04, label, ha='center', fontsize=8, color=MUTED)
    ax.text(0.03, 0.5, f'$w_w$ = {ww}\n$s_w$ = {ws}', transform=ax.transAxes, fontsize=8, va='center')
setp(axs, ylim=(0, 1.15), xlim=(r[0], r[-1]))
setp(axs[1], xlabel=R_LABEL)
setp(axs[:, 0], ylabel='Mixture weight')
axs[0, 2].legend(frameon=False, fontsize=7, loc='center right')
