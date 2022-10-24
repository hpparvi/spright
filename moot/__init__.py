from importlib.metadata import version

from .rmrelation import RMRelation
from .rmestimator import RMEstimator

version = version('moot')

__all__ = ['version', 'RMRelation', 'RMEstimator']
