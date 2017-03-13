.. currentmodule:: gwpy.plotter

.. _gwpy-plotter-bode:

#######################################
Visualising filters (:class:`BodePlot`)
#######################################

Any time-domain or Fourier-domain filters can be visualised using the Bode plot, showing the magnitude (in decibels) and phase (in degress) response of a linear time-invariant filter. The `BodePlot` allows for simple display of these responses for any filter, for example a 10 Hertz high-pass filter (from the :doc:`Bode plot example <../examples/miscellaneous/bode_plot>`:

.. plot::
   :include-source:

   from gwpy.signal import bandpass
   from gwpy.plotter import BodePlot
   zpk = bandpass(40, 1000, 4096, analog=True)
   plot = BodePlot(f, title='40-1000\,Hz bandpass filter')
   plot.show()

|
