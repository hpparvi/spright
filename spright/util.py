from numpy import ndarray, iterable, isscalar, full
from numpy.random import normal
from uncertainties.core import Variable as UVar


def sample_distribution(d, nsamples: int) -> ndarray:
    if hasattr(d, 'rvs'):
        return d.rvs(size=nsamples)
    elif isinstance(d, UVar):
        return normal(d.n, d.s, size=nsamples)
    elif iterable(d) and len(d) == 2:
        return normal(d[0], d[1], size=nsamples)
    elif isscalar(d):
        return full(nsamples,  d)
    else:
        raise ValueError("Could not interpret 'd' as a distribution")