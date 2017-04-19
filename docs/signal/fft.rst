.. include:: ../references.txt

.. _gwpy-signal-fft:

#####################
FFT routines for GWpy
#####################

The Fourier transform (and the associated digital FFT algorithm) is the most
common way of investigating the frequency-domain content of a time-domain signal.
GWpy provides wrappers of average spectrum methods from :mod:`scipy.signal`, and |lal|_ to simplify calculating a `~gwpy.frequencyseries.FrequencySeries` from a `~gwpy.timeseries.TimeSeries`.

The `gwpy.signal.fft` sub-package provides the following PSD averaging methods:

.. autosummary::
   :toctree: ../api
   :nosignatures:

   gwpy.signal.fft.scipy.welch
   gwpy.signal.fft.scipy.bartlett
   gwpy.signal.fft.lal.welch
   gwpy.signal.fft.lal.bartlett
   gwpy.signal.fft.lal.median
   gwpy.signal.fft.lal.median_mean

Each of these can be specified by passing the function name as the ``method`` keyword argument to any of the relevant `~gwpy.timeseries.TimeSeries` instance methods, e.g::

   >>> ts = TimeSeries(...)
   >>> psd = ts.psd(..., method='welch', ...)

.. note::

   Since `welch` and `bartlett` are defined in both the `scipy` and `lal`
   sub-modules, the LAL versions are registered as ``'lal-welch'`` and
   ``'lal-bartlett'``, so to use them, pass ``method='lal-welch'`` to the
   relevant `~gwpy.timeseries.TimeSeries` method.
