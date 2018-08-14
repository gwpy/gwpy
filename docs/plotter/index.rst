.. currentmodule:: gwpy.plotter

.. _gwpy-plotter:

#################################
Plotting in GWpy (`gwpy.plotter`)
#################################

Visualisation of the data produced by gravitational-wave detectors is
obviously a key part of understanding their sensitivity, and studying the
potential gravitational-wave signals they record.
The :mod:`gwpy.plotter` module provides a number of plot classes, each
representing display of a corresponding data type.

==============
Basic plotting
==============

The majority of core data objects in GWpy come with a built-in `plot()`
method, allowing quick display of a single data set, for example:

.. plot::
   :include-source:
   :context:

    >>> from gwpy.timeseries import TimeSeries
    >>> h1 = TimeSeries.fetch_open_data('H1', 1126259457, 1126259467)
    >>> plot = h1.plot()
    >>> plot.show()

|

Users can also import the relevant plotting `class` objects and generate
more complicated plots manually:

.. plot::
   :include-source:
   :context:

    >>> l1 = TimeSeries.fetch_open_data('L1', 1126259457, 1126259467)
    >>> from gwpy.plotter import TimeSeriesPlot
    >>> plot = TimeSeriesPlot()
    >>> ax = plot.gca()
    >>> ax.plot(h1, color='gwpy:ligo-hanford')
    >>> ax.plot(l1, color='gwpy:ligo-livingston')
    >>> ax.set_ylabel('Strain noise')
    >>> plot.show()


==========
Plot types
==========

The following diagram displays the available Plot objects and their
inheritance from :class:`Plot`.

.. inheritance-diagram:: core timeseries frequencyseries spectrogram table filter

===============
Class reference
===============

.. toctree::
   :hidden:

   api

A full reference of the above plotting `class` objects can be found :doc:`here <api>`.
