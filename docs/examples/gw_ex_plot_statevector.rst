==============================================================
GWpy Example: plotting a :class:`~gwpy.timeseries.StateVector`
==============================================================

.. currentmodule:: gwpy.timeseries

Problem
-------

I would like to examine the state of the internal seismic isolation system
supporting the Fabry-Perot mirror at the end of the Y-arm at LHO,
in order to investigate a noise source.

Access to these data is restricted to the LIGO Scientific Collaboration and
the Virgo Collaboration, but collaboration members can use the NDS2 service
to download data.

Solution
--------

First, we define the bitmask for the state-vector in question, simply a list
of what each binary bit in the vector represents:

.. literalinclude:: ../../examples/gw_ex_plot_statevector.py
   :lines: 37-46

and can download the data for the relevant GPS interval as follows:

.. literalinclude:: ../../examples/gw_ex_plot_statevector.py
   :lines: 34,49

At this point it is useful to downsample the `StateVector` from the natural
sampling frequency of 4096 Hz to 16 Hz, simply for ease of plotting:

.. literalinclude:: ../../examples/gw_ex_plot_statevector.py
   :lines: 50

Now we can make a plot:

.. literalinclude:: ../../examples/gw_ex_plot_statevector.py
   :append: plot.show()
   :lines: 53-55

.. plot:: ../examples/gw_ex_plot_statevector.py

|

Here we see that the seismic isolation system moved in and out of its
operating state, as a result of a large seismic transient shaking the vacuum
chamber.
