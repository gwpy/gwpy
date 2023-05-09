.. currentmodule:: pydischarge.astro

.. _pydischarge-astro:

#######################
Astrophysical modelling
#######################

Currently the only methods available from `pydischarge.astro` are concerned with calculating sensitive distance.

.. _pydischarge-astro-range:

==================
Sensitive distance
==================

The sensitive distance (sometimes called 'range', or 'horizon') is a measure of how far out into the universe a gravitational-wave source can be and still be detectable by a gravitational-wave detector 

`pydischarge.astro` provides methods to calculate the distance to simple models of inspiral and burst signals:

.. autosummary::
   :toctree: ../api

   burst_range
   burst_range_spectrum
   inspiral_range
   inspiral_range_psd
