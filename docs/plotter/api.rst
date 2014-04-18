############
Plotting API
############

This document provides a reference for the following :class:`~matplotlib.figure.Figure` `class` objects:

.. autosummary::
   :nosignatures:

   ~gwpy.plotter.core.Plot
   ~gwpy.plotter.timeseries.TimeSeriesPlot
   ~gwpy.plotter.spectrum.SpectrumPlot
   ~gwpy.plotter.spectrogram.SpectrogramPlot
   ~gwpy.plotter.segments.SegmentPlot
   ~gwpy.plotter.table.EventTablePlot

and the following :class:`~matplotlib.axes.Axes` `class` objects:

.. autosummary::
   :nosignatures:

   ~gwpy.plotter.axes.Axes
   ~gwpy.plotter.timeseries.TimeSeriesAxes
   ~gwpy.plotter.spectrum.SpectrumAxes
   ~gwpy.plotter.segments.SegmentAxes
   ~gwpy.plotter.table.EventTableAxes

================
`Figure` objects
================

Each of the below classes represents a figure object; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. currentmodule:: gwpy.plotter


.. autoclass:: gwpy.plotter.core.Plot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.timeseries.TimeSeriesPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.spectrum.SpectrumPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.spectrogram.SpectrogramPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.segments.SegmentPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.table.EventTablePlot
   :no-inherited-members:


==============
`Axes` objects
==============

Each of the below classes represents a set of axes on which data are displayed; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. autoclass:: gwpy.plotter.axes.Axes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.timeseries.TimeSeriesAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.spectrum.SpectrumAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.segments.SegmentAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.table.EventTableAxes
   :no-inherited-members:
