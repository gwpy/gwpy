############
Plotting API
############

This document provides a reference for the following :class:`~matplotlib.figure.Figure` `class` objects:

.. autosummary::
   :nosignatures:

   ~gwpy.plotter.Plot
   ~gwpy.plotter.TimeSeriesPlot
   ~gwpy.plotter.SpectrumPlot
   ~gwpy.plotter.SpectrogramPlot
   ~gwpy.plotter.SegmentPlot
   ~gwpy.plotter.EventTablePlot

and the following :class:`~matplotlib.axes.Axes` `class` objects:

.. autosummary::
   :nosignatures:

   ~gwpy.plotter.Axes
   ~gwpy.plotter.TimeSeriesAxes
   ~gwpy.plotter.SpectrumAxes
   ~gwpy.plotter.SegmentAxes
   ~gwpy.plotter.EventTableAxes

================
`Figure` objects
================

Each of the below classes represents a figure object; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. currentmodule:: gwpy.plotter


.. autoclass:: gwpy.plotter.Plot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.TimeSeriesPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.SpectrumPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.SpectrogramPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.SegmentPlot
   :no-inherited-members:


.. autoclass:: gwpy.plotter.EventTablePlot
   :no-inherited-members:

.. autoclass:: gwpy.plotter.BodePlot
   :no-inherited-members:


==============
`Axes` objects
==============

Each of the below classes represents a set of axes on which data are displayed; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. autoclass:: gwpy.plotter.Axes
   :no-inherited-members:
   :exclude-members: legend


.. autoclass:: gwpy.plotter.TimeSeriesAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.SpectrumAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.SegmentAxes
   :no-inherited-members:


.. autoclass:: gwpy.plotter.EventTableAxes
   :no-inherited-members:
