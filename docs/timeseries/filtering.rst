.. currentmodule:: gwpy.timeseries.core

########################
Filtering a `TimeSeries`
########################

Much of the real-time control system of the LIGO interferometers is handled by monitoring and filtering time-series data.
Additionally, in gravitational-wave analysis, the interesting frequency bands for a given search can be a sub-set of the full bandwidth of the instruments.
In both cases it is required to apply time-domain filters to some data in order to extract the most useful information.

===========
Bandpassing
===========

The following methods of the `TimeSeries` provide functionality for band-passing data:

.. autosummary::

   TimeSeries.lowpass
   TimeSeries.highpass
   TimeSeries.bandpass
