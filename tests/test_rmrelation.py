import pytest
import pandas
from numpy import ndarray
from spright.distribution import Distribution
from spright.rmrelation import RMRelation


def test_rmrelation_initialization_with_default_fname():
    """
    Test initializing RMRelation with default provided fname.
    """
    rmrelation = RMRelation()
    assert rmrelation.rdmap is not None
    assert rmrelation.rmmap is not None
    assert isinstance(rmrelation.posterior_samples, pandas.DataFrame)
    assert isinstance(rmrelation.catalog, pandas.DataFrame)
    assert isinstance(rmrelation.rdsamples, pandas.DataFrame)


def test_rmrelation_sample_invalid_quantity():
    """
    Test that RMRelation.sample method raises ValueError for invalid quantity.
    """
    rmrelation = RMRelation()
    with pytest.raises(ValueError):
        rmrelation.sample(quantity="invalid_quantity")


def test_rmrelation_predict_density():
    """
    Test the predict_density method of RMRelation.
    """
    rmrelation = RMRelation()
    distribution = rmrelation.predict_density(radius=(1.0, 0.1), nsamples=1000)
    assert isinstance(distribution, Distribution)
    assert distribution.quantity == "density"
    assert isinstance(distribution.samples, ndarray)


def test_rmrelation_predict_mass():
    """
    Test the predict_mass method of RMRelation.
    """
    rmrelation = RMRelation()
    distribution = rmrelation.predict_mass(radius=(1.0, 0.1), nsamples=1000)
    assert isinstance(distribution, Distribution)
    assert distribution.quantity == "mass"
    assert isinstance(distribution.samples, ndarray)


def test_rmrelation_predict_radius():
    """
    Test the predict_radius method of RMRelation.
    """
    rmrelation = RMRelation()
    distribution = rmrelation.predict_radius(mass=(1.0, 0.1), nsamples=1000)
    assert isinstance(distribution, Distribution)
    assert distribution.quantity == "radius"
    assert isinstance(distribution.samples, ndarray)


def test_rmrelation_sample_valid_quantity():
    """
    Test that RMRelation.sample method works correctly for a valid quantity.
    """
    rmrelation = RMRelation()
    distribution = rmrelation.sample(quantity="density", radius=(1.0, 0.1), nsamples=1000)
    assert isinstance(distribution, Distribution)
    assert distribution.quantity == "density"
