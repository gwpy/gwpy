#################################
Plotting in GWpy (`gwpy.plotter`)
#################################

.. currentmodule:: gwpy.plotter

Visualisation of the data produced by gravitational-wave detectors is obviously a key part of understanding their sensitivity, and studying the potential gravitational-wave signals they record.
The :mod:`~gwpy.plotter` module provides a number of plot classes, each representing display of a corresponding data type.

These classes include

- :class:`~core.Plot` (for quick display of any data)
- :class:`~timeseries.TimeSeriesPlot` and :class:`~gwf.GWFramePlot`
- :class:`~spectrum.SpectrumPlot`
- :class:`~spectrogram.SpectrogramPlot`

Other than the :class:`~core.Plot` (which takes no data arguments), each of the series plots takes an instance of the associated series as the argument when generating a plot, for example with a :class:`~gwpy.timeseries.core.TimeSeries` (called ``timeseries``)::

    >>> from gwpy.plotter import TimeSeriesPlot
    >>> plot = TimeSeriesPlot(timeseries)

==========
Plot types
==========

The following diagram displays the available Plot objects and their inheritance from `~core.Plot`.

.. inheritance-diagram:: core timeseries spectrum spectrogram gwf filter

.. toctree::
   :maxdepth: 1

   filter

========
Plot API
========

.. currentmodule:: gwpy.plotter.core

.. autoclass:: Plot
   :members:
