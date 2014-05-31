.. currentmodule:: gwpy.spectrogram

========================================================================
GWpy Example: plotting a whitened :class:`~gwpy.spectrogram.Spectrogram`
========================================================================

Problem
-------

I would like to calculate a whitened spectrogram of the gravitational-wave strain signal to try and find some signals...

Solution
--------

We can quickly fetch the data we want using the :meth:`TimeSeries.fetch` method:

.. literalinclude:: ../../examples/gw_ex_plot_median_spectrogram.py
   :lines: 33,36-37

We can then generate a :class:`~gwpy.spectrogram.Spectrogram` by calling the :meth:`~TimeSeries.spectrogram` method of the existing `TimeSeries`, taking a square root to return an amplitude spectral density spectrogram, rather than a power spectral density spectrogram

.. literalinclude:: ../../examples/gw_ex_plot_median_spectrogram.py
   :lines: 40

and can whiten it by normalising with the median in each frequency bin using the :meth:`~gwpy.spectrogram.Spectrogram.ratio` method:

.. literalinclude:: ../../examples/gw_ex_plot_median_spectrogram.py
   :lines: 41

Not we can plot it:

.. literalinclude:: ../../examples/gw_ex_plot_median_spectrogram.py
   :append: plot.show()
   :lines: 44-46

.. plot:: ../examples/gw_ex_plot_median_spectrogram.py

|
