.. Spright documentation master file, created by
   sphinx-quickstart on Sun Oct 29 14:54:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spright
=======

**Spright** (/spraɪt/) is a fast Bayesian radius-density-mass relation for small planets. The package offers an easy-to-use
command line script for quick prediction of an exoplanet mass or radius given the other and a set of Python classes that
can be incorporated into your study.

.. _installation:

Installation
============

Spright can be installed easily using pip:

.. code-block:: console

    $ pip install spright

Quickstart
==========

The radius-density-mass relations calculated by ``spright`` can be accessed either using a command line interface or classes
provided by the package.

Spright supports the prediction of four quantities:

- **mass** [M_Earth] given planet's radius and its uncertainty
- **bulk density** [g/cm^3] given planet's radius and its uncertainty
- **radius**  [R_Earth] given planet's mass (+ uncertainty)
- **RV semi-amplitude (K)** [m/s] given planetary mass, orbital period, eccentricity, and host star mass, all with
  optional uncertainties.

Command line
---------------------

The command line interface can be used to predict one of the quantities supported by ``spright`` from others, and
also to create publication-quality plots directly without the need to write a single line of code:

.. code-block:: console

    $ spright --predict mass --radius 1.8 0.1 --plot-distribution

The predicted quantity is chosen using the ``--predict`` argument, and can be one of ``mass``, ``radius``, ``density``,
or ``k``. 

Python code
-----------

Spright also offers a set of Python classes for the people who prefer to code and the classes offer direct access to the
predicted numerical distributions.


.. code-block:: python

    from spright import RMRelation

    rmr = RMRelation()
    mds = rmr.predict_mass(radius=(1.8, 0.1))
    mds.plot()

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
