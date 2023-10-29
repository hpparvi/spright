.. Spright documentation master file, created by
   sphinx-quickstart on Sun Oct 29 14:54:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spright
=======

**Spright** (/spraɪt/) is a fast Bayesian radius-density-mass relation for small planets that allows one to predict planetary masses, densities, and RV semi-amplitudes from an estimate of the
planet's radius, or planetary radii given an estimate of the planet's mass.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

Spright can be installed easily using pip:

.. code-block:: console

    $ pip install spright

Usage
=====

From the command line
---------------------

Spright offers an easy-to-use command line script for people who are not overly interested in coding, and
nearly-as-easy-to-use set of Python classes for the people who prefer to code. The command line script can create
directly publication-quality plots, and the classes offer direct access to the predicted numerical distributions.

.. code-block:: console

    $ spright --predict mass --radius 1.8 0.1 --plot-distribution

Python code
-----------

.. code-block:: python

    from spright import RMRelation

    rmr = RMRelation()
    mds = rmr.predict_mass(radius=(1.8, 0.1))
    mds.plot()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
