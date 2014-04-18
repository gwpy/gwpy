#################################
Plotting in GWpy (`gwpy.plotter`)
#################################

.. currentmodule:: gwpy.plotter

Visualisation of the data produced by gravitational-wave detectors is obviously a key part of understanding their sensitivity, and studying the potential gravitational-wave signals they record.
The :mod:`~gwpy.plotter` module provides a number of plot classes, each representing display of a corresponding data type:

.. autosummary::
   :nosignatures:

   ~core.Plot
   ~timeseries.TimeSeriesPlot
   ~spectrum.SpectrumPlot
   ~spectrogram.SpectrogramPlot
   ~segments.SegmentPlot
   ~table.EventTablePlot

Other than the :class:`~core.Plot` (which takes no data arguments), each of the series plots takes an instance of the associated series as the argument when generating a plot, for example with a :class:`~gwpy.timeseries.core.TimeSeries` (called ``timeseries``)::

    >>> from gwpy.plotter import TimeSeriesPlot
    >>> plot = TimeSeriesPlot(timeseries)

==========
Plot types
==========

The following diagram displays the available Plot objects and their inheritance from :class:`~core.Plot`.

.. inheritance-diagram:: core timeseries spectrum spectrogram table filter

=================
Plot applications
=================

.. toctree::
   :maxdepth: 1

   filter

===============
Class reference
===============

.. toctree::
   :hidden:

   api

A full reference of the above plotting `class` objects can be found :doc:`here <api>`.
