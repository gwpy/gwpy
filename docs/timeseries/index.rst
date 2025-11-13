.. _gwpy-timeseries:

.. currentmodule:: gwpy.timeseries

################
Time Series data
################

Gravitational-wave detectors are time-domain instruments, recording
gravitational-wave amplitude as a differential change in the lengths
of each of the interferometer arms.
The primary output of these detectors is a single time-stream of
gravitational-wave strain (also referred to as *h(t)*).

Alongside the strain data, thousands of instrumental control and
error signals and environmental monitors are recorded in real-time and
archived for off-line study.
These data are colloquially called the *auxiliary channels*.

=======================
The :class:`TimeSeries`
=======================

.. code-block:: python
    :caption: Importing the `TimeSeries`
    :name: gwpy-timeseries-import

    from gwpy.timeseries import TimeSeries

GWpy provides the `TimeSeries` object as a way of representing time series
data.
The `TimeSeries` is built on top of the :class:`numpy.ndarray`, and so many
methods and applications of this object should be familiar to
:mod:`numpy` users.

For example, to create a simple `TimeSeries` filled with
:mod:`~numpy.random` data:

.. code-block:: python
    :caption: Creating a `TimeSeries` with random data
    :name: gwpy-timeseries-random

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

Here we see the random data we have created, as well as the associated
metadata common to any time series data:

.. autosummary::

    ~TimeSeries.unit
    ~TimeSeries.t0
    ~TimeSeries.dt
    ~TimeSeries.name
    ~TimeSeries.channel

==================
Associated classes
==================

Alongside the `TimeSeries` class, `gwpy.timeseries` module provides a
small set of related classes for handling collections of data:

.. autosummary::
    :nosignatures:

    TimeSeriesDict
    TimeSeriesList

================================
Reading/writing time series data
================================

.. toctree::
    :maxdepth: 2

    io
    get

=========================
Plotting time series data
=========================

.. toctree::
    :maxdepth: 2

    plot

=============
Reference/API
=============

The above documentation references the following objects:

.. autosummary::
    :nosignatures:

    TimeSeries
    TimeSeriesDict
    TimeSeriesList
