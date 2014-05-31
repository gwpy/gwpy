=============================================================
GWpy Example: plotting a :class:`~gwpy.timeseries.TimeSeries`
=============================================================

.. currentmodule:: gwpy.timeseries

Problem
-------

I would like to study the gravitational wave strain time-series around
the time of an interesting simulated signal during the last science run
(S6).
For this we can use a set of data public available
`here <http://www.ligo.org/science/GW100916/>`_.

Solution
--------

Since the data we are interested in are public, we can download the data
as follows:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 34,41

and can format them into a :class:`TimeSeries` using `numpy`, and by
supplying our own metadata (which is all documented on the website):

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 36,38,44

Now we can make a plot:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :append: plot.show()
   :lines: 47-49

.. plot:: ../examples/gw_ex_plot_timeseries.py

|
