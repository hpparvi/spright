from matplotlib.pyplot import subplots, setp
from numpy import linspace, full_like
from scipy.stats import norm, uniform
from docfigs import COLORS, MUTED, load

rmr, rdm, pv = load()
ps = rmr.posterior_samples
priors = {'r1': uniform(0.5, 2.0), 'r4': uniform(1.0, 3.0), 'ww': uniform(0.0, 1.0), 'ws': uniform(-1.0, 2.0),
          'cr': uniform(0.0, 1.0), 'cw': norm(0.5, 0.1), 'ip': uniform(0.1, 6.9), 'sp': norm(-0.5, 1.5),
          'log10_sr': norm(0.0, 0.35), 'log10_sw': norm(0.0, 0.35), 'log10_sp': norm(0.0, 0.35)}
labels = {'r1': r'$r_1$ [R$_\oplus$]', 'r4': r'$r_4$ [R$_\oplus$]', 'ww': '$w_w$', 'ws': '$s_w$', 'cr': '$c_r$',
          'cw': '$c_w$', 'ip': r'$i_p$ [g cm$^{-3}$]', 'sp': '$s_p$', 'log10_sr': r'$\log_{10} \sigma_r$',
          'log10_sw': r'$\log_{10} \sigma_w$', 'log10_sp': r'$\log_{10} \sigma_p$'}

fig, axs = subplots(3, 4, figsize=(10, 6), constrained_layout=True)
for ax, name in zip(axs.flat, ps.columns):
    v = ps[name].values
    lo, hi = priors[name].ppf([0.001, 0.999]) if isinstance(priors[name].dist, type(uniform)) else (v.min(), v.max())
    lo, hi = min(lo, v.min()), max(hi, v.max())
    x = linspace(lo, hi, 300)
    ax.hist(v, bins=40, range=(lo, hi), density=True, color=COLORS[1], alpha=0.8, label='Posterior')
    ax.plot(x, priors[name].pdf(x), c=MUTED, lw=1.5, ls='--', label='Prior')
    setp(ax, xlabel=labels[name], yticks=[])
axs[0, 0].legend(frameon=False, fontsize=7)
axs.flat[-1].remove()
