import seaborn as sb
import plotext as tpl
from arviz.stats import kde

from argparse import ArgumentParser
from math import pi

from numpy import median, percentile
from uncertainties import ufloat

from matplotlib.pyplot import subplots, setp, show
from matplotlib import cm

from spright import RMRelation
from spright.model import plot_model_means

PAGEWIDTH = 7.09  # A typical journal full page width
COLWIDTH = 3.46   # A typical journal column width

def plot_map(rmr: RMRelation, r: float, re: float, ax):
    rmr.rdmap.plot_map(ax, cm=cm.Blues)
    plot_model_means(rmr.posterior_samples, rmr.rdm, ax)
    #ax.errorbar(rmr.catalog.radius, rmr.catalog.density, xerr=rmr.catalog.radius_e, yerr=rmr.catalog.density_e,
    #            fmt='.', c='0.5')
    ax.axvspan(r-re, r+re, fill=False, lw=1, ec='k', ls='--', alpha=0.8)
    ax.axvline(r, c='k')
    setp(ax, xlim=(0.5, 4), ylim=(0, 10))

def plot_terminal(distribution):
    tpl.plot_size(min(80, tpl.terminal_width()), min(20, tpl.terminal_height()))
    x, y = kde(distribution.samples)
    tpl.plot(x, y)
    #tpl.hist(distribution.samples, bins=80)
    [tpl.vline(v) for v in percentile(distribution.samples, [50, 2.5, 97.5])]
    tpl.xlabel(f"{distribution.quantity.title()}")
    tpl.yticks([])
    tpl.xlim(*percentile(distribution.samples, [1, 99]))
    tpl.show()

def arg_to_ufloat(arg):
    if arg is None:
        return None
    elif isinstance(arg, float):
        return ufloat(arg, 1e-4)
    elif len(arg) > 2:
        raise ValueError
    elif len(arg) == 2:
        return ufloat(float(arg[0]), float(arg[1]))
    else:
        return ufloat(float(arg[0]), 1e-4)

def __main__():
    ap  = ArgumentParser(description="Spright: a fast mass-density-radius relation for small exoplanets.",
                         epilog="Note: all the numeric values can be given as a single float or as two float where the second one stands for the measurement uncertainty.")
    ap.add_argument('--predict', type=str,  choices=['radius', 'mass', 'density', 'rv'], default='mass', dest='quantity', help='Quantity to predict (default: %(default)s).')
    ap.add_argument('--radius', '-r', nargs='+', dest='radius', help='Radius of the planet [R_Earth].')
    ap.add_argument('--mass', '-m', nargs='+', dest='mass', help='Mass of the planet [M_Earth] either as a single float or two floats where the second one stands for the measurement uncertainty.')
    ap.add_argument('--mstar', '-s', nargs='+', default=1.0, dest='mstar', help='Host star mass [M_Sun] either as a single float or two floats where the second one stands for the measurement uncertainty (default: %(default)s)..')
    ap.add_argument('--period', '-p', nargs='+', default=10.0, dest='period', help="Planet's orbital period [d] (default: %(default)s).")
    ap.add_argument('--eccentricity', '-e', nargs='+', default=0.0, dest='ecc', help="Planet's orbital eccentricity (default: %(default)s).")
    ap.add_argument('--n-samples', '-n', type=int, default=20_000, dest='ns', help='Number of samples to draw (default: %(default)s).')
    ap.add_argument('--model', default='stpm', help='Spright model to use. Can be either one of the included models or a path to a custom one (default: %(default)s).')
    ap.add_argument('--plot-distribution', action='store_true', default=False, help='Plot the predicted distribution.')
    ap.add_argument('--plot-map', action='store_true', default=False, help='Plot the 2D probability map used to create the distribution.')
    ap.add_argument('--dont-plot-terminal', action='store_false', default=True, dest='plot_terminal', help="Don't plot the predicted distribution in the terminal.")

    args = ap.parse_args()
    rmr = RMRelation(args.model)

    if args.quantity in ('mass', 'density'):
        if args.radius is None:
            print("Error: spright requires planet radius when predicting planet mass or density")
            exit()

    radius = arg_to_ufloat(args.radius)
    mass = arg_to_ufloat(args.mass)

    mstar = arg_to_ufloat(args.mstar)
    period = arg_to_ufloat(args.period)
    ecc = arg_to_ufloat(args.ecc)

    distribution = rmr.sample(args.quantity, radius, mass, mstar, period, ecc, nsamples=args.ns)

    print(distribution)
    if args.plot_terminal:
        plot_terminal(distribution)

    if args.plot_map:
        with sb.plotting_context('paper', font_scale=0.9):
            fig, ax = subplots(figsize=(PAGEWIDTH, 0.5*PAGEWIDTH))
            plot_map(rmr, radius.n, radius.s, ax=ax)
            fig.tight_layout()

    if args.plot_distribution:
        with sb.plotting_context('paper', font_scale=0.9):
            fig, ax = subplots(figsize=(COLWIDTH, 0.75*COLWIDTH))
            distribution.plot(ax=ax)
            ax.text(0.98, 0.98, f'R$_\mathrm{{p}}$ = {radius.n} ± {radius.s} R$_\oplus$', ha='right', va='top', transform=ax.transAxes)
            fig.tight_layout()

    if args.plot_distribution or args.plot_map:
        show()
