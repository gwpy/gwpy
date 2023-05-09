.. currentmodule:: pydischarge.timeseries

.. plot::
   :include-source: False
   :context: reset
   :nofigs:

   >>> from pydischarge.timeseries import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

.. _pydischarge-timeseries-plot:

#########################
Plotting time-domain data
#########################

=========================
Plotting one `TimeSeries`
=========================

The `TimeSeries` class includes a :meth:`~TimeSeries.plot` method to
trivialise visualisation of the contained data.
Reproducing the example from :ref:`pydischarge-timeseries-remote`:

.. plot::
   :include-source:
   :context:

   >>> l1hoft = TimeSeries.fetch_open_data('L1', 'Sep 14 2015 09:50:29', 'Sep 14 2015 09:51:01')
   >>> plot = l1hoft.plot()
   >>> plot.show()

The returned object `plot` is a :class:`~pydischarge.plot.Plot`, a sub-class of
:class:`matplotlib.figure.Figure` adapted for GPS time-stamped data.
Customisations of the figure or the underlying :class:`~pydischarge.plot.Axes` can
be done using standard :mod:`matplotlib` methods.
For example:

.. plot::
   :include-source:
   :context:

   >>> ax = plot.gca()
   >>> ax.set_ylabel('Gravitational-wave amplitude [strain]')
   >>> ax.set_epoch(1126259462)
   >>> ax.set_title('LIGO-Livingston strain data around GW150914')
   >>> ax.axvline(1126259462, color='orange', linestyle='--')
   >>> plot.refresh()

Here the :meth:`~pydischarge.plot.Axes.set_epoch` method is used to reset the
reference time for the x-axis.

=======================================
Plotting multiple `TimeSeries` together
=======================================

Multiple `TimeSeries` classes can be combined on a figure in a number of
different ways, the most obvious is to :meth:`~TimeSeries.plot` the first,
then add the second on the same axes.
Reusing the same ``plot`` from above:

.. plot::
   :include-source:
   :context:

   >>> h1hoft = TimeSeries.fetch_open_data('H1', 'Sep 14 2015 09:50:29', 'Sep 14 2015 09:51:01')
   >>> ax = plot.gca()
   >>> ax.plot(h1hoft)
   >>> plot.refresh()

Alternatively, the two `TimeSeries` could be combined into a
`TimeSeriesDict` to use the :meth:`~TimeSeriesDict.plot` method from that class:

.. plot::
   :include-source:
   :context: close-figs

   >>> combined = TimeSeriesDict()
   >>> combined['L1'] = l1hoft
   >>> combined['H1'] = h1hoft
   >>> plot = combined.plot()
   >>> plot.gca().legend()
   >>> plot.show()

The third method of achieving the same result is by importing and accessing the `~pydischarge.plot.Plot` object directly:

.. plot::
   :include-source:
   :context: close-figs

   >>> from pydischarge.plot import Plot
   >>> plot = Plot(l1hoft, h1hoft)
   >>> plot.show()

Using the `~pydischarge.plot.Plot` directly allows for greater customisation.
The ``separate=True`` keyword argument can be used to plot each `TimeSeries`
on its own axes, with ``sharex=True`` given to link the time scales for each
:class:`~matplotlib.axes.Axes`:

.. plot::
   :include-source:
   :context: close-figs

   >>> plot = Plot(l1hoft, h1hoft, separate=True, sharex=True)
   >>> plot.show()
