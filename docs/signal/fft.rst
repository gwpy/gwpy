.. _gwpy-signal-fft:

#####################
FFT routines for GWpy
#####################

The Fourier transform (and the associated digital FFT algorithm) is the most
common way of investigating the frequency-domain content of a time-domain signal.
GWpy provides wrappers of average spectrum methods from :mod:`scipy.signal`, |lal|_, and :mod:`pycbc.psd` to simplify calculating a `~gwpy.frequencyseries.FrequencySeries` from a `~gwpy.timeseries.TimeSeries`.

The `gwpy.signal.fft` sub-package provides the following FFT averaging methods:

=======================  ===================================
      Method name                     Function
=======================  ===================================
      ``'pycbc_welch'``  `gwpy.signal.fft.pycbc.welch`
   ``'pycbc_bartlett'``  `gwpy.signal.fft.pycbc.bartlett`
     ``'pycbc_median'``  `gwpy.signal.fft.pycbc.median`
``'pycbc_median_mean'``  `gwpy.signal.fft.pycbc.median_mean`
        ``'lal_welch'``  `gwpy.signal.fft.lal.welch`
     ``'lal_bartlett'``  `gwpy.signal.fft.lal.bartlett`
       ``'lal_median'``  `gwpy.signal.fft.lal.median`
  ``'lal_median_mean'``  `gwpy.signal.fft.lal.median_mean`
      ``'scipy_welch'``  `gwpy.signal.fft.scipy.welch`
   ``'scipy_bartlett'``  `gwpy.signal.fft.scipy.bartlett`
=======================  ===================================

Each of these can be specified by passing the function name as the ``method`` keyword argument to any of the relevant `~gwpy.timeseries.TimeSeries` instance methods, e.g::

   >>> ts = TimeSeries(...)
   >>> psd = ts.psd(..., method='pycbc_welch', ...)

Additionally, short versions the following registrations map to the first registered instance of that name in the registry:

=======================  ===================================
      Method name                     Function
=======================  ===================================
            ``'welch'``  `gwpy.signal.fft.pycbc.welch`
         ``'bartlett'``  `gwpy.signal.fft.pycbc.bartlett`
           ``'median'``  `gwpy.signal.fft.pycbc.median`
      ``'median_mean'``  `gwpy.signal.fft.pycbc.median_mean`
=======================  ===================================

.. note::

   The simple names (e.g. ``'welch'``) are mapped at runtime, meaning if
   you don't have `pycbc` on your system, the registrations will map to
   |lal|_ or `scipy` instead. These are provided only for convenience, so
   if you want to be able to reproduce your results on any system,
   use the full API-specific name when using :meth:`TimeSeries.psd`
   and friends.


.. note::

   You can check which FFT library will be used via the
   :func:`gwpy.signal.fft.get_default_fft_api` method, e.g::

      >>> from gwpy.signal.fft import get_default_fft_api
      >>> get_default_fft_api()
      'pycbc'
