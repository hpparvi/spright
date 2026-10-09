.. _theory:

How Spright works
=================

Spright predicts the mass, bulk density, or RV semi-amplitude of a small planet from its radius
(or the radius from its mass) as a full probability distribution. This page explains what happens
between a catalogue of measured planets and such a prediction: how the radius-density-mass relation
is **modelled**, how the model is **estimated** from a planet sample, and how the result is **used**.

The method is described in detail in
`Parviainen, Luque & Pallé (2023) <https://doi.org/10.1093/mnras/stad3504>`_. Here we follow the
code, and all the figures are created from the relations shipped with the package.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

A classical mass-radius relation is a curve with a scatter around it. This does not describe small
planets well: between roughly 1.5 and 2.5 Earth radii, planets of the same radius can be rocky,
water-rich, or have a puffy hydrogen-helium envelope, and their densities differ by a factor of
three. The distribution of the mass given the radius is then wide, skewed, and often bimodal.

Spright deals with this in three steps, and the package is divided in the same way:

1. **Model.** The probability of a planet having a density :math:`\rho` given its radius
   :math:`r` is written as a mixture of three planet populations. The model has 11 parameters.
2. **Estimate.** :class:`~spright.RMEstimator` infers the posterior distribution of the model
   parameters from a catalogue of planets with measured radii and masses, averages the model over
   the posterior, and stores the result as numerical probability maps in a FITS file.
3. **Use.** :class:`~spright.RMRelation` reads such a file and draws samples of the predicted
   quantity from the maps. No model evaluation is needed at this stage, which makes the
   predictions fast.

The model lives in the radius-density space rather than in the radius-mass space, because the
theoretical composition models are close to linear there and the three populations separate
cleanly. A density is converted to a mass with the volume of the planet whenever needed.

The model
---------

Three populations
~~~~~~~~~~~~~~~~~

Each population has a mean density that depends on the radius:

- **Rocky planets** follow the theoretical radius-density models by Zeng et al. (2019) for a
  mixture of rock and iron. The free parameter is the iron mass fraction :math:`c_r`: zero stands
  for a pure-rock planet and one for a pure-iron planet. An Earth-like composition has
  :math:`c_r \approx 0.3`.
- **Water worlds** follow the theoretical models for a planet made of rock and water, with the
  water mass fraction :math:`c_w` as the free parameter. Two sets of models are available: Zeng
  et al. (2019, ``'z19'``) and Aguichine et al. (2021, ``'a21'``).
- **Sub-Neptunes**, the planets with a puffy envelope, have no simple theoretical radius-density
  model, because their radius depends strongly on the envelope mass, age, and irradiation. Their
  mean density is described by a power law

  .. math::

      \mu_p(r) = i_p \left( \frac{r}{2\,R_\oplus} \right)^{s_p},

  where :math:`i_p` is the density at two Earth radii and :math:`s_p` is the exponent.

The theoretical models are stored as tables over the radius and composition and interpolated
bilinearly, so evaluating a mean density is cheap.

.. plot:: figures/mean_densities.py

   The mean densities of the three populations. The thick lines show the posterior median model
   of the default ``'stpm'`` relation, and the thin lines show how the mean density changes with
   the free parameter of each population.

Planets do not follow the mean densities exactly. The density of a planet in population :math:`i`
is distributed around the mean following a Student's t-distribution,

.. math::

    p_i(\rho \mid r) = T\left(\rho;\ \mu_i(r),\ \sigma_i,\ \nu=5\right),

where :math:`\sigma_i` is the scale of the distribution and a free parameter for each population.
The t-distribution with five degrees of freedom has heavier tails than a normal distribution,
which keeps a few planets with unusual densities from dominating the fit.

Mixture weights
~~~~~~~~~~~~~~~

The populations are combined into a single probability density using radius-dependent weights,

.. math::

    p(\rho \mid r, \theta) = \sum_{i \in \{r, w, p\}} w_i(r)\ T\left(\rho;\ \mu_i(r),\ \sigma_i,\ 5\right),
    \qquad \sum_i w_i(r) = 1.

Small planets are all rocky and large planets are all sub-Neptunes, and the weights describe what
happens in between. They are defined by four transition radii: the rocky planets start to give way
to the water worlds at :math:`r_1` and are gone by :math:`r_2`, and the water worlds start to give
way to the sub-Neptunes at :math:`r_3` and are gone by :math:`r_4`. The weights change linearly
within the transitions.

Sampling four ordered radii directly is awkward, so the model uses the outer radii :math:`r_1`
and :math:`r_4` together with the relative width :math:`w_w` and shape :math:`s_w` of the
water-world population. The inner radii are

.. math::

    r_2 = r_1 + d\,(1 - w_w + s_w a), \qquad
    r_3 = r_1 + d\,(w_w + s_w a),

where :math:`d = r_4 - r_1` and :math:`a = 0.5 - |w_w - 0.5|`. The distance between the two is
:math:`r_3 - r_2 = d\,(2 w_w - 1)`, which gives the width parameter its meaning:

- :math:`w_w > 0.5`: there is a range of radii where all the planets are water worlds.
- :math:`w_w = 0.5`: the water worlds reach a weight of one at a single radius.
- :math:`w_w < 0.5`: the two transitions overlap, and the water worlds are always mixed with
  the other two populations. At :math:`w_w = 0` the water-world population vanishes and the rocky
  planets change directly into sub-Neptunes.

The shape parameter :math:`s_w \in [-1, 1]` moves the water-world population towards the small
or the large radii without changing its width. A model without water worlds is therefore not a
separate model but a corner of the parameter space, and the data decide how strong the
water-world population is.

.. plot:: figures/mixture_weights.py

   The mixture weights as a function of the radius for six combinations of the water-world
   population width :math:`w_w` and shape :math:`s_w`, with :math:`r_1 = 1.2` and
   :math:`r_4 = 2.8`.

Putting the pieces together gives the full model. The figure below shows the three weighted
components and their sum for the posterior median parameters of the ``'stpm'`` relation.

.. plot:: figures/model_components.py

   The weighted probability densities of the three populations and the full model in the
   radius-density space.

Parameters and priors
~~~~~~~~~~~~~~~~~~~~~

The model has 11 free parameters. :math:`U(a, b)` stands for a uniform prior and
:math:`N(\mu, \sigma)` for a normal prior.

.. list-table::
   :header-rows: 1
   :widths: 14 14 46 26

   * - Name
     - Symbol
     - Meaning
     - Prior
   * - ``r1``
     - :math:`r_1`
     - Start of the rocky-to-water transition [R\ :sub:`⊕`]
     - :math:`U(0.5, 2.5)`
   * - ``r4``
     - :math:`r_4`
     - End of the water-to-sub-Neptune transition [R\ :sub:`⊕`]
     - :math:`U(1.0, 4.0)`
   * - ``ww``
     - :math:`w_w`
     - Relative width of the water-world population
     - :math:`U(0, 1)`
   * - ``ws``
     - :math:`s_w`
     - Shape of the water-world population
     - :math:`U(-1, 1)`
   * - ``cr``
     - :math:`c_r`
     - Iron mass fraction of the rocky planets
     - :math:`U(0, 1)`
   * - ``cw``
     - :math:`c_w`
     - Water mass fraction of the water worlds
     - :math:`N(0.5, 0.1)`
   * - ``ip``
     - :math:`i_p`
     - Sub-Neptune density at 2 R\ :sub:`⊕` [g cm\ :sup:`-3`]
     - :math:`U(0.1, 7.0)`
   * - ``sp``
     - :math:`s_p`
     - Sub-Neptune density exponent
     - :math:`N(-0.5, 1.5)`
   * - ``log10_sr``
     - :math:`\log_{10} \sigma_r`
     - Scale of the rocky-planet density distribution
     - :math:`N(0, 0.35)`
   * - ``log10_sw``
     - :math:`\log_{10} \sigma_w`
     - Scale of the water-world density distribution
     - :math:`N(0, 0.35)`
   * - ``log10_sp``
     - :math:`\log_{10} \sigma_p`
     - Scale of the sub-Neptune density distribution
     - :math:`N(0, 0.35)`

Two additional constraints apply. A solution with :math:`r_1 > r_4` is rejected, and so is a
solution where the sub-Neptunes would be denser than a rocky planet without iron where the
populations meet. The priors can be changed with :meth:`~spright.RMEstimator.add_lnprior` or by
modifying ``RMEstimator.lpf.ps`` before the optimisation.

Estimating the relation
-----------------------

Likelihood
~~~~~~~~~~

The data are a catalogue of :math:`N` planets with measured radii and masses (or densities) and
their uncertainties. The uncertainties are too large to be ignored: a planet with a 20% mass
uncertainty covers a good part of the density range of the model.

Spright takes the uncertainties into account by drawing :math:`K` samples of the radius and mass
for each planet from normal distributions defined by the measurements, and converting each sample
to a density. The likelihood of a planet is the model averaged over its samples, and the planets
are independent, so that

.. math::

    \ln \mathcal{L}(\theta) = \sum_{i=1}^{N} \ln \left[ \frac{1}{K} \sum_{j=1}^{K}
    p(\rho_{ij} \mid r_{ij}, \theta) \right].

The average over the samples is a Monte Carlo estimate of the integral of the model over the
measurement uncertainty of the planet. Because the density samples are calculated from the radius
and mass samples, the strong correlation between the radius and density uncertainties is carried
along automatically.

.. plot:: figures/data_samples.py

   The catalogue behind the ``'stpm'`` relation as measurements with uncertainties (left) and as
   the samples used to calculate the likelihood (right), drawn over the posterior median model.
   The curved sample clouds show the correlation between the radius and the density.

Combining catalogues
~~~~~~~~~~~~~~~~~~~~

Several catalogues can be combined by concatenating them. Measurements that share a planet name
are treated as alternative measurements of the same planet, and the :math:`K` samples of the
planet are divided evenly between them. This averages the likelihood of the planet over the
catalogues, which marginalises over the choice of the catalogue planet by planet without counting
any planet twice.

:func:`spright.io.read_combined` builds such a combined catalogue from the STPM, TEPCat, and
Exoplanet.eu catalogues. It identifies the planets found in several catalogues by their names and
by their host star positions and orbital periods.

.. code-block:: python

    from spright import RMEstimator
    from spright.io import read_combined

    df = read_combined(['stpm', 'tepcat', 'exoplanet_eu'], max_teff=4000)
    rme = RMEstimator(nsamples=100, names=df.name.values,
                      radii=(df.r.values, df.rerr.values),
                      masses=(df.m.values, df.merr.values))

Optimisation and sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

The posterior is the product of the priors and the likelihood. It is multimodal and has sharp
edges, so the estimation starts with a global optimisation using Differential Evolution and
continues with MCMC sampling using ``emcee``, starting the chains from the optimised parameter
vector population.

.. code-block:: python

    from spright import RMEstimator
    from spright.io import read_stpm

    names, radii, masses = read_stpm('stpm_230202.csv')
    rme = RMEstimator(nsamples=100, names=names, radii=radii, masses=masses, seed=0)

    rme.optimize(niter=500)                  # Global optimisation
    rme.sample(10000, thin=50, repeats=6)    # MCMC sampling
    df = rme.posterior_samples()             # Posterior samples as a DataFrame

The likelihood is compiled with ``numba`` and evaluated in parallel over the parameter vector
population. The example above takes some tens of minutes on a desktop computer.

.. plot:: figures/posterior.py

   The marginal posterior distributions of the model parameters for the ``'stpm'`` relation
   together with their priors. The data constrain the transition radii, the compositions, and the
   sub-Neptune density well, while the width and shape of the water-world population stay
   uncertain.

From the posterior to the relation maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A prediction should include both the intrinsic scatter of the planet densities and the
uncertainty in the model parameters. This is done by averaging the model over the posterior
samples :math:`\theta_k`,

.. math::

    p(\rho \mid r) = \frac{1}{M} \sum_{k=1}^{M} p(\rho \mid r, \theta_k),

which gives the posterior predictive distribution of the density given the radius.
:meth:`~spright.RMEstimator.compute_maps` evaluates this average on a regular grid in radius and
density, and stores it as a *probability map*. The same is done in radius and mass, where the
density of each grid point is calculated from its mass and radius, and the probability density is
multiplied by :math:`\mathrm{d}\rho/\mathrm{d}m = 1/V(r)`.

Two more maps are computed from each probability map. The **cumulative distribution function**
(CDF) is the cumulative sum of the probability map along the density axis, normalised to one for
each radius. The **inverse CDF** gives the density as a function of the radius and a probability
:math:`u \in [0, 1]`, and is what the predictions are drawn from. The maps are also computed in
the other direction, giving the distribution of the radius for a given density or mass.

.. plot:: figures/relation_maps.py

   The radius-density probability map of the ``'stpm'`` relation, its CDF, and its inverse CDF.
   The contours in the inverse CDF panel show the densities in g cm\ :sup:`-3`.

.. code-block:: python

    rme.compute_maps(nsamples=5000, rres=400, dres=200, pres=200)
    rme.save('my_relation.fits')

The maps cover the radii from 0.5 to 6 R\ :sub:`⊕`, the densities from 0 to 12 g cm\ :sup:`-3`,
and the masses from 0 to 25 M\ :sub:`⊕` by default. The FITS file contains the maps, the posterior
samples they were averaged over, the planet catalogue, and the radius, mass, and density samples
used in the inference, so a saved relation documents how it was made.

Using the relation
------------------

Sampling from the maps
~~~~~~~~~~~~~~~~~~~~~~

:class:`~spright.RMRelation` reads a relation file and predicts by *inverse transform sampling*:

1. Draw radius samples from the distribution of the measured radius.
2. Draw a uniform random number :math:`u` between zero and one for each radius sample.
3. Interpolate the inverse CDF at each :math:`(r, u)`. The result is a density sample.

A mass sample is calculated from each density sample and its radius sample. The uncertainty of
the radius measurement is so included in the prediction, as is its correlation with the density.

.. plot:: figures/icdf_sampling.py

   Predicting the density of a planet with a radius of 1.8 ± 0.1 R\ :sub:`⊕`. The inverse CDF has
   a step where the planet population changes, and the step moves with the radius. This is what
   makes the predicted distribution bimodal.

Predicting a mass
~~~~~~~~~~~~~~~~~

The prediction methods return a :class:`~spright.distribution.Distribution` object, which holds
the samples and can summarise and plot them.

.. plot::
   :include-source:

    from matplotlib.pyplot import subplots
    from spright import RMRelation

    rmr = RMRelation()

    fig, axs = subplots(1, 3, figsize=(10, 3))
    for ax, radius in zip(axs, (1.2, 1.8, 2.6)):
        mass = rmr.predict_mass(radius=(radius, 0.05))
        mass.plot(ax=ax)
        ax.set_title(f'r = {radius} ± 0.05')

A 1.2 R\ :sub:`⊕` planet is almost certainly rocky and its mass is predicted well. At
1.8 R\ :sub:`⊕` the planet can belong to any of the three populations and the predicted mass
distribution is bimodal, and at 2.6 R\ :sub:`⊕` the planet is a sub-Neptune.

The radius can be given as a float, a ``(mean, sigma)`` tuple, an ``uncertainties.ufloat``, or a
frozen ``scipy.stats`` distribution. The same holds for all the other input quantities. Printing
a distribution gives its median and central intervals, and the parameters of a one- or
two-component Student's t-distribution model fitted to the samples:

.. code-block:: python

    >>> rmr.predict_mass(radius=(1.8, 0.05))
    Mass distribution
    size: 5000
    is bimodal: True

    Median: 4.10,
    68% limits: [3.1 7.9],
    95% limits: [2.2 9.6]

    Distribution model:
      0.62 × T(m=3.51, σ=0.62, λ=5.05)
    + 0.38 × T(m=7.73, σ=0.99, λ=5.13)

The samples themselves are in ``Distribution.samples``, and should be preferred over the summary
whenever the distribution is used in a further calculation.

Other predictions
~~~~~~~~~~~~~~~~~

The **bulk density** is predicted like the mass, without the conversion from density to mass. The
**radius** is predicted from a mass using the radius-mass map in the other direction. The **RV
semi-amplitude** is calculated from the predicted mass samples together with samples of the
orbital period :math:`P`, stellar mass :math:`M_\star`, and eccentricity :math:`e`,

.. math::

    K = \left( \frac{2 \pi G}{P} \right)^{1/3} \frac{m}{M_\star^{2/3}} \frac{1}{\sqrt{1 - e^2}},

assuming an edge-on orbit, which is a good approximation for a transiting planet.

.. plot::
   :include-source:

    from matplotlib.pyplot import subplots
    from spright import RMRelation

    rmr = RMRelation()
    density = rmr.predict_density(radius=(1.8, 0.05))
    radius = rmr.predict_radius(mass=(5.0, 0.5))
    k = rmr.predict_rv_semi_amplitude(radius=(1.8, 0.05), period=5.2, mstar=(0.45, 0.02))

    fig, axs = subplots(1, 3, figsize=(10, 3))
    density.plot(ax=axs[0])
    radius.plot(ax=axs[1])
    k.plot(ax=axs[2])

Planet class
~~~~~~~~~~~~

The mixture weights give the probability of a planet belonging to each population.
:meth:`~spright.RMRelation.predict_class` evaluates the weights for the posterior samples stored
in the relation file and returns them as a table, and :meth:`~spright.RMRelation.plot_class`
shows them in a ternary diagram.

.. plot:: figures/class_probabilities.py

   The probability of a planet being a rocky planet, a water world, or a sub-Neptune as a
   function of its radius for the ``'stpm'`` relation. The lines show the posterior means and the
   shaded areas the central 68% posterior intervals.

Choosing a relation
~~~~~~~~~~~~~~~~~~~

A relation describes the planet sample it was estimated from. Spright ships relations estimated
from three catalogues, for M dwarf and FGK star hosts, and for the two water-world models:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Planet sample
   * - ``stpm`` (default)
     - Small transiting planets around M dwarfs (STPM catalogue)
   * - ``tepcat_m_z19``, ``tepcat_m_a21``
     - TEPCat, M dwarf hosts
   * - ``tepcat_fgk_z19``, ``tepcat_fgk_a21``
     - TEPCat, FGK star hosts
   * - ``exoeu_m_z19``, ``exoeu_m_a21``
     - Exoplanet.eu, M dwarf hosts
   * - ``exoeu_fgk_z19``, ``exoeu_fgk_a21``
     - Exoplanet.eu, FGK star hosts

The suffix tells the water-world model: ``z19`` for Zeng et al. (2019) and ``a21`` for Aguichine
et al. (2021). A relation is chosen by its name, and a relation estimated with
:class:`~spright.RMEstimator` by its file name:

.. code-block:: python

    rmr = RMRelation('tepcat_fgk_z19')
    rmr = RMRelation('my_relation.fits')

.. plot:: figures/relation_comparison.py

   The predicted mass as a function of the radius for three of the shipped relations.

The same predictions are available from the command line:

.. code-block:: console

    $ spright --predict mass --radius 1.8 0.05 --model tepcat_fgk_z19 --plot-distribution

Things to keep in mind
----------------------

- **The relation is only as good as its sample.** The predictions describe planets like those in
  the catalogue: small, transiting, mostly short-period planets with precise radius and mass
  measurements. Use a relation estimated for a similar host star type.
- **The maps have a limited range.** Samples that fall outside the map, such as radii below 0.5
  or above 6 R\ :sub:`⊕`, are dropped, so a prediction close to the edges contains fewer samples
  than requested.
- **A prediction is a distribution.** The distributions are often bimodal, and a mean and a
  standard deviation describe them poorly. Use the samples.
- **The populations are model components.** The class probabilities tell how the model divides
  the planets given the data and the priors. Whether water worlds exist as a distinct population
  is the question the model is built to study, not something it assumes.

References
----------

- Parviainen, H., Luque, R., & Pallé, E. 2023, MNRAS,
  `doi:10.1093/mnras/stad3504 <https://doi.org/10.1093/mnras/stad3504>`_
- Luque, R. & Pallé, E. 2022, Science, 377, 1211
- Zeng, L. et al. 2019, PNAS, 116, 9723
- Aguichine, A. et al. 2021, ApJ, 914, 84
- Foreman-Mackey, D. et al. 2013, PASP, 125, 306
