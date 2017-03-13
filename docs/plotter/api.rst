.. currentmodule:: gwpy.plotter

.. _gwpy-plotter-api:

############
Plotting API
############

This document provides a reference for the following :class:`~matplotlib.figure.Figure` `class` objects:

.. autosummary::
   :nosignatures:

   Plot
   TimeSeriesPlot
   FrequencySeriesPlot
   SpectrogramPlot
   SegmentPlot
   EventTablePlot
   BodePlot

and the following :class:`~matplotlib.axes.Axes` `class` objects:

.. autosummary::
   :nosignatures:

   Axes
   TimeSeriesAxes
   FrequencySeriesAxes
   SegmentAxes
   EventTableAxes

================
`Figure` objects
================

Each of the below classes represents a figure object; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. autoclass:: Plot
   :no-inherited-members:


.. autoclass:: TimeSeriesPlot
   :no-inherited-members:


.. autoclass:: FrequencySeriesPlot
   :no-inherited-members:


.. autoclass:: SpectrogramPlot
   :no-inherited-members:


.. autoclass:: SegmentPlot
   :no-inherited-members:


.. autoclass:: EventTablePlot
   :no-inherited-members:

.. autoclass:: BodePlot
   :no-inherited-members:


==============
`Axes` objects
==============

Each of the below classes represents a set of axes on which data are displayed; for brevity inherited methods and attributes are not documented here, please follow links to the parent classes for documentation of available methods and attributes.

.. autoclass:: Axes
   :no-inherited-members:
   :exclude-members: legend


.. autoclass:: TimeSeriesAxes
   :no-inherited-members:


.. autoclass:: FrequencySeriesAxes
   :no-inherited-members:


.. autoclass:: SegmentAxes
   :no-inherited-members:


.. autoclass:: EventTableAxes
   :no-inherited-members:
