===============================================================
GWpy Example: plotting a :class:`~gwpy.spectrogram.Spectrogram`
===============================================================

.. currentmodule:: gwpy.timeseries

Problem
-------

I would like to plot a spectrogram (time-frequency-amplitude heat map) of the gravitational wave strain signal.

Solution
--------

We can quickly fetch the data we want using the :meth:`TimeSeries.fetch` method:

.. literalinclude:: ../../examples/gw_ex_plot_spectrogram.py
   :lines: 33,36-37

We can then generate a :class:`~gwpy.spectrogram.Spectrogram` by calling the :meth:`~TimeSeries.spectrogram` method of the existing `TimeSeries`, taking a square root to return an amplitude spectral density spectrogram, rather than a power spectral density spectrogram

.. literalinclude:: ../../examples/gw_ex_plot_spectrogram.py
   :lines: 40

Not we can plot it:

.. literalinclude:: ../../examples/gw_ex_plot_spectrogram.py
   :append: plot.show()
   :lines: 43-45

.. plot:: ../examples/gw_ex_plot_spectrogram.py

|
