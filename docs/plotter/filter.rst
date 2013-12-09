############################################################
Visualising filters (:class:`~gwpy.plotter.filter.BodePlot`)
############################################################

.. currentmodule:: gwpy.plotter.filter

Any time-domain or Fourier-domain filters can be visualised using the Bode plot, showing the magnitude (in decibels) and phase (in degress) response of a linear time-invariant filter. The `BodePlot` allows for simple display of these responses for any filter, for example a 10 Hertz high-pass applied to a signal recorded at 256 Hertz:

.. plot::
   :include-source:

   >>> from scipy import signal
   >>> from gwpy.plotter import BodePlot
   >>> highpass = signal.butter(4, 10 * (2.0 / 256), btype='highpass')
   >>> plot = BodePlot(highpass, sample_rate=256)
