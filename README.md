# Spright

Spright (/spraɪt/) is a fast Bayesian radius-density-mass relation for small planets.

## Installation

    pip install spright

## Usage

### Planet mass prediction

    from spright import RMRelation 

    rmr = RMRelation()
    mds = rmr.predict_mass(radius=(1.8, 0.05))
    mds.plot()

![Predicted mass](notebooks/f01_mass.svg)


### RV semi-amplitude prediction

The radial velocity semi-amplitude can be predicted given the planet's radius, orbital period, orbital eccentricity (optional),
and the host star mass.

    from spright import RMRelation 

    rmr = RMRelation()
    mds = rmr.predict_rv_semi_amplitude(radius=(1.8, 0.1), period=2.2, mstar=(0.5, 0.05))
    mds.plot()

![Predicted RV semi-amplitude](notebooks/f02_rv_semi_amplitude.svg)

Here the `RMRelation.predict_rv_semi_amplitude` method can also be given the planet's orbital eccentricity (`ecc`), 
and all the parameters (`radius`, `period`, `mstar`, and `ecc`) can either be floats, ufloats, or two-value tuples where the second value gives
the parameter uncertainty.

### Calculation of a new radius-density-mass relation

    from spright import RMEstimator
    
    rme = RMEstimator(names=names, radii=radii, masses=masses)
    rme.optimize()
    rme.sample()
    rme.compute_maps()
    rme.save('map_name.fits')

---
&copy; 2023 Hannu Parviainen