.. currentmodule:: pydischarge.plot

.. _pydischarge-plot:

###################################
Plotting in pyDischarge (:mod:`pydischarge.plot`)
###################################

==============
Basic plotting
==============

The :mod:`pydischarge.plot` module provides integrated extensions to the fantastic
data visualisation tools provided by |matplotlib|_.

---------------------------------------------
Basic plotting with :mod:`~matplotlib.pyplot`
---------------------------------------------

Each of the data representations provided by `pydischarge` can be directly passed
to the standard methods available in `~matplotlib.pyplot`:

.. plot::
   :include-source:

   >>> from pydischarge.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> from matplotlib import pyplot as plt
   >>> plt.plot(data)
   >>> plt.show()

----------------------------
``.plot()`` instance methods
----------------------------

Each of the data representations provided by `pydischarge` also come with a
``.plot()`` method that provides a figure with improved defaults tailored
to those data:

.. plot::
   :include-source:

   >>> from pydischarge.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> plot = data.plot()
   >>> plot.show()

The ``.plot()`` methods accept any keywords that can be used to create the
:class:`~matplotlib.figure.Figure` and the :class:`~matplotlib.axes.Axes`,
and to draw the element itself, e.g:

.. plot::
   :include-source:

   >>> from pydischarge.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> plot = data.plot(figsize=(8, 4.8), ylabel='Strain',
   ...                  color='pydischarge:ligo-livingston')
   >>> plot.show()

----------------
Multi-data plots
----------------

pyDischarge enables trivial generation of plots with multiple datasets.
The :class:`~pydischarge.plot.Plot` constructor will accept an arbitrary
collection of data objects and will build a figure with the required geometry
automatically.
By default, a flat set of objects are shown on the same axes:

.. plot::
   :include-source:
   :context: reset

   >>> from pydischarge.timeseries import TimeSeries
   >>> hdata = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
   >>> ldata = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> from pydischarge.plot import Plot
   >>> plot = Plot(hdata, ldata, figsize=(12, 4.8))
   >>> plot.show()

However, `separate=True` can be given to show each dataset on a separate
`~pydischarge.plot.Axes`:

.. plot::
   :include-source:
   :context: close-figs

   >>> plot = Plot(hdata, ldata, figsize=(12, 6), separate=True, sharex=True)
   >>> plot.show()

.. warning::

   The `Plot` constructor can only handle one plotting method at any time
   (e.g. ``ax.plot()``, ``ax.imshow()``), so you can't create plots with
   a line and an image using this call,
   for example.

==================
Plot customisation
==================

.. toctree::

   gps
   colorbars
   legend
   log

=================
Plot applications
=================

.. toctree::

   colors
   filter
