.. currentmodule:: gwpy.astro

.. _gwpy-astro:

#######################
Astrophysical modelling
#######################

Currently the only methods available from `gwpy.astro` are concerned with calculating sensitive distance.

.. _gwpy-astro-range:

==================
Sensitive distance
==================

The sensitive distance (sometimes called 'range', or 'horizon') is a measure of how far out into the universe a gravitational-wave source can be and still be detectable by a gravitational-wave detector 

`gwpy.astro` provides methods to calculate the distance to simple models of inspiral and burst signals:

.. autosummary::

    burst_range
    burst_range_spectrum
    inspiral_range
    inspiral_range_psd
    range_timeseries
    range_spectrogram
