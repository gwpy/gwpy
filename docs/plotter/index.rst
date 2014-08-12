#################################
Plotting in GWpy (`gwpy.plotter`)
#################################

.. currentmodule:: gwpy.plotter

Visualisation of the data produced by gravitational-wave detectors is
obviously a key part of understanding their sensitivity, and studying the
potential gravitational-wave signals they record.
The :mod:`gwpy.plotter` module provides a number of plot classes, each
representing display of a corresponding data type.

=============
Plotting data
=============

The majority of core data objects in GWpy come with a built-in :meth:`plot`
method, allowing quick display of a single data set, for example:

.. plot::
   :include-source:
   :context:

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.fetch('H1:LDAS-STRAIN', 968654552, 968654562)
    >>> plot = data.plot()
    >>> plot.show()

|

Users can also import the relevant plotting `class` objects and generate
more complicated plots manually:

.. plot::
   :include-source:
   :context:

    >>> data2 = TimeSeries.fetch('L1:LDAS-STRAIN', 968654552, 968654562)
    >>> from gwpy.plotter import TimeSeriesPlot
    >>> plot2 = TimeSeriesPlot()
    >>> ax2 = plot2.gca()
    >>> ax2.plot(data, color='k', linestyle='--')
    >>> ax2.plot(data2, color='r', linestyle=':')
    >>> plot2.show()

==========
Plot types
==========

The following diagram displays the available Plot objects and their
inheritance from :class:`Plot`.

.. inheritance-diagram:: core timeseries spectrum spectrogram table filter

The following pages outline specific applications of some of the specialist plot types

.. toctree::
   :titlesonly:

   filter

===============
Class reference
===============

.. toctree::
   :hidden:

   api

A full reference of the above plotting `class` objects can be found :doc:`here <api>`.
