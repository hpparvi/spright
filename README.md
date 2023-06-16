# Spright

Spright (/spraɪt/) is a fast Bayesian radius-density-mass relation for small planets.

## Installation

    pip install spright

## Usage

### Planet mass estimation

    from spright import RMRelation 

    rmr = RMRelation()
    mds = rmr.sample('mass', (1.8, 0.05))
    mds.plot()

### Calculation of a new radius-density-mass relation

    from spright import RMEstimator
    
    rme = RMEstimator(names=names, radii=radii, masses=masses)
    rme.optimize()
    rme.sample()
    rme.compute_maps()
    rme.save('map_name.fits')