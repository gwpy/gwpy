.. currentmodule:: gwpy.timeseries

#######################
The :class:`TimeSeries`
#######################

.. code-block:: python

   >>> from gwpy.timeseries import TimeSeries


Gravitational-wave detectors are time-domain instruments, attempting to record gravitational-wave amplitude as a differential change in the lengths of each of the interferometer arms.
Alongside these data, thousands of auxiliary instrumental control and error signals and environmental monitors are recorded in real-time and recorded to disk and archived for off-line study.

GWpy represents these data through the :class:`TimeSeries` object, a sub-class of the :class:`numpy.ndarray` containing the data themselves and a full set of metadata.

Any `TimeSeries` can be generated from a standard `~numpy.ndarray` or `list` by providing the data and the minimal :attr:`~TimeSeries.epoch` and :attr:`~TimeSeries.sample_rate` metadata::

   >>> series = TimeSeries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], epoch=1000000000, sample_rate=1)
   >>> print(series)
   TimeSeries([ 1  2  3  4  5  6  7  8  9 10],
              name: None,
              unit: None,
              epoch: 2011-09-14 01:46:59.000,
              channel: None,
              sample_rate: 1 Hz)

The full set of metadata that can be provided is as follows:

.. autosummary::

   ~TimeSeries.name
   ~TimeSeries.unit
   ~TimeSeries.epoch
   ~TimeSeries.sample_rate
   ~TimeSeries.channel

=============================
Accessing interferometer data
=============================

As described above, the data from each instrument are archived for off-line study in gravitational-wave frame (``.gwf``) files.
To learn more about this particular data format, take a look at the specification document `LIGO-T970130 <https://dcc.ligo.org/LIGO-T970130/public>`_.
These files are stored on disk by the LIGO Data Grid and can be either accessed either directly or remotely.

The easiest way to access interferometer data is to use `TimeSeries.get`:

.. automethod:: TimeSeries.get
   :noindex:

This method will try direct file access and remote NDS2 access in order to get you the data you want, with the minimum of inputs (direct file access is preferred)::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.get('L1:PSL-ODC_CHANNEL_OUT_DQ', 1067042880, 1067042912)

If you want to be more specific, there are three other methods that you can use that allow more control over the data access:

.. autosummary::

   TimeSeries.fetch
   TimeSeries.find
   TimeSeries.read

All of these should return identical `TimeSeries` objects.

For more details on accessing data via either of these sources, or from publicly-released data files, please read the following tutorials:

.. toctree::
   :maxdepth: 1

   gwf
   nds
   public-data

========================
`TimeSeries` collections
========================

A single `TimeSeries` is meant to represent one stream of contiguous data from a single source.
There are two collections provided that allow you to bundle multiple `TimeSeries` from a single source, and multiple `TimeSeries` of the same epoch from a number of sources:

--------------------
The `TimeSeriesList`
--------------------

The `TimeSeriesList` is an extension of the builtin `list` type to allow easy collection and manipulation of a number of `TimeSeries` from a single source, say different epochs from a single :class:`~gwpy.detector.channel.Channel`.
This object comes with a few handy methods that make combining multiple epochs very simple:

.. autosummary::

   ~TimeSeriesList.coalesce
   ~TimeSeriesList.join

--------------------
The `TimeSeriesDict`
--------------------

The `TimeSeriesDict` does for multiple channels what the `TimeSeriesList` does for multiple epochs, allowing easy collection of `TimeSeries` from a single epochs but multiple sources.
This class offers the same data access as for the `TimeSeries`, with the `TimeSeriesDict.get` the easiest way to load some data.
See the full reference for what other functionality is available.

.. _plotting-a-timeseries:

=======================
Plotting a `TimeSeries`
=======================

The `TimeSeries` object comes with its own :meth:`~TimeSeries.plot` method, which will quickly construct a :class:`~gwpy.plotter.timeseries.TimeSeriesPlot`.
In the following example, we download ten seconds of gravitational-wave strain data from the LIGO Hanford Observatory, and display it:

.. plot:: timeseries/timeseries_plot.py
   :include-source:

|

As described in the `NDS access documentation <nds>`_, downloading these data requires LIGO.ORG credentials (issued to members of the LIGO-Virgo Collaboration and friends).
However, these data, and more from other LIGO milestones, are available publicly `here <http://www.ligo.org/science/data-releases.php>`_.
For instructions on how to download and read those data, please `read this <public-data>`_.

=========================
`TimeSeries` applications
=========================

.. toctree::
   :titlesonly:

   filtering
   statevector

=========================
`Class <class>` reference
=========================

This reference contains the following `Class` entries:

.. autosummary::
   :nosignatures:

   TimeSeries
   TimeSeriesList
   TimeSeriesDict

.. autoclass:: TimeSeries

.. autoclass:: TimeSeriesList
   :no-inherited-members:

.. autoclass:: TimeSeriesDict
   :no-inherited-members:

