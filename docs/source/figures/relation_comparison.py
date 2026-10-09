from matplotlib.pyplot import subplots, setp
from numpy import argmin
from docfigs import INK, R_LABEL, M_LABEL
from spright import RMRelation

relations = {'stpm': 'STPM (M dwarfs)', 'tepcat_m_z19': 'TEPCat, M dwarfs', 'tepcat_fgk_z19': 'TEPCat, FGK stars'}

fig, axs = subplots(1, 3, figsize=(11, 3.4), sharex='all', sharey='all', constrained_layout=True)
c = INK
for ax, (key, label) in zip(axs, relations.items()):
    m = RMRelation(key).rmmap
    q = [m.xy_icdf[:, argmin(abs(m.probs - p))] for p in (0.025, 0.16, 0.5, 0.84, 0.975)]
    ax.fill_between(m.x, q[0], q[4], color=c, alpha=0.1, lw=0, label='95%')
    ax.fill_between(m.x, q[1], q[3], color=c, alpha=0.25, lw=0, label='68%')
    ax.plot(m.x, q[2], c=c, lw=2, label='Median')
    ax.set_title(f"{label}\nRMRelation('{key}')", fontsize=9)
    ax.set_xlabel(R_LABEL)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
setp(axs, xlim=(0.6, 3.5), ylim=(0, 20))
axs[0].set_ylabel(M_LABEL)
