###########################################
GWpy Example: measuring a transfer function
###########################################

.. currentmodule:: gwpy.timeseries

Problem
-------

I would like to study how ground motion is mitigated by the HEPI
active seismic isolation system.

Solution
--------

First up, we import everything we're going to need:

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :lines: 31-33

Next we can set the times and the channels we need:

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :lines: 40-41,43-44

Now we can dowload the data using the
:meth:`TimeSeriesDict.fetch` method:

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :lines: 47-51

At this point it is useful to take a quick look at the amplitude spectral
densities of each of the two signals:

.. literalinclude:: gw_ex_plot_transfer_function_asds.py
   :lines: 2-8

.. plot:: examples/gw_ex_plot_transfer_function_asds.py

Now, in order to calculate the transfer function, we need to calculate an
averaged Fourier transform of each :class:`TimeSeries`.
Desiring 0.01 Hz resolution, we use a 100-second FFT, with 50%
overlap (specified in seconds):

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :lines: 57,58

Now we can simply calculate the transfer function by dividing the output
signal (the HEPI data) by the input signal (the ground motion data), taking
care because the two data channels may be recorded at different rates (meaning
their FFTs will have different sizes):

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :lines: 61,62

And now we can plot the result as a :class:`~gwpy.plotter.BodePlot`:

.. literalinclude:: ../../examples/gw_ex_plot_transfer_function.py
   :append: plot.show()
   :lines: 64-66

.. plot:: ../examples/gw_ex_plot_transfer_function.py

|
