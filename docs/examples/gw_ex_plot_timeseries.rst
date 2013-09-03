===============================
GWpy.Ex: plotting a time-series
===============================

.. currentmodule:: gwpy.timeseries.core


Problem
-------

I would like to study the gravitational wave strain time-series around
the time of an interesting simulated signal during the last science run
(S6). I have access to the frame files on the LIGO Data Grid machine
``ldas-pcdev2.ligo-wa.caltech.edu`` and so can read them directly.

Solution
--------

First up, we need to decide what times we want. The interesting signal
happened between 06:42 an 06:43 on September 16 2010, and so we can set
the times from there:

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 13,22,23

The relevant frame files for those times exist on disk, we just have to
find them. The relevant tool for this is the GLUE datafind client. First
we open a connection to the default server (set in the environment for
the system):

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 11,26

Then we can ask it to find the frames we care about. These are for the
``H`` observatory (LIGO Hanford), specifically the ``H1_LDAS_C02_L2``
data type (representing the LIGO Data Analysis System (LDAS) calibration
version 02, at level 2):

.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 27,28

Now we know where the data are, we can read the strain channel ``H1:LDAS-STRAIN`` into a `TimeSeries` and plot:


.. literalinclude:: ../../examples/gw_ex_plot_timeseries.py
   :lines: 14,31,34-35

.. plot:: ../examples/gw_ex_plot_timeseries.py


