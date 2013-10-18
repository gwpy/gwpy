.. currentmodule:: gwpy.timeseries.core
================================
GWpy.Ex: plotting a `TimeSeries`
================================

Problem
-------

I would like to study the gravitational wave strain time-series around
the time of an interesting simulated signal during the last science run
(S6).

Solution
--------

First up, we need to decide what times we want. The interesting signal
happened between 06:42 an 06:43 on September 16 2010, and so we can set
the times from there:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 28,37,38

Next, we can read the relevant data using the Network Data Server, via the :meth:`TimeSeries.fetch` method:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 29,41

Now we can make a plot:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 45

.. plot:: ../examples/gw_ex_plot_timeseries.py
