.. _gwpy-timeseries:

.. currentmodule:: gwpy.timeseries

################
Time-domain data
################

================
The `TimeSeries`
================

Gravitational-wave detectors are time-domain instruments, attempting to record gravitational-wave amplitude as a differential change in the lengths of each of the interferometer arms.
The primary output of these detectors is a single time-stream of gravitational-wave strain.

Alongside these data, thousands of auxiliary instrumental control and error signals and environmental monitors are recorded in real-time and archived for off-line study.

GWpy provides the `TimeSeries` object as a way of representing these and similar time-domain data.
The `TimeSeries` is built from the :class:`numpy.ndarray`, and so many of the methods and applications of this object should be familiar to :mod:`numpy` users.

For example, to create a simple `TimeSeries` filled with :mod:`~numpy.random` data::

   >>> from numpy.random import random
   >>> from gwpy.timeseries import TimeSeries
   >>> t = TimeSeries(random(1000))
   >>> print(t)
   TimeSeries([ 0.59442285, 0.61979421, 0.62968915,...,  0.98309223,
                0.94513298, 0.1826175 ]
              unit: Unit(dimensionless),
              t0: 0.0 s,
              dt: 1.0 s,
              name: None,
              channel: None)

Here we see the random data we have created, as well as the associated metadata common to any time-domain data:

.. autosummary::

   ~TimeSeries.unit
   ~TimeSeries.t0
   ~TimeSeries.dt
   ~TimeSeries.name
   ~TimeSeries.channel

==================
Associated classes
==================

Alongside the `TimeSeries` class, `gwpy.timeseries` module provides a small set of related classes for handling bit-vector data, and collections of data:

.. autosummary::
   :nosignatures:
   :toctree: ../api/

   TimeSeries
   StateVector
   TimeSeriesDict
   StateVectorDict

================================
Reading/writing time-domain data
================================

.. toctree::
   :maxdepth: 2

   remote-access
   ldg
   io

=========================
Plotting time-domain data
=========================

.. toctree::
   :maxdepth: 2

   plot

=============
Reference/API
=============

The above documentation references the following objects:

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   TimeSeries
   TimeSeriesDict
   TimeSeriesList
   StateVector
   StateVectorDict
   StateTimeSeries
