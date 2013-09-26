########################################
Spectral data (`gwpy.spectrum.Spectrum`)
########################################

.. currentmodule:: gwpy.spectrum.core

While gravitational-wave detectors are time-domain instruments, their sensitivity is often measured as a power-spectral-density over the range of interesting GW frequencies (~10-10,000 Hz).

Generating a `Spectrum`
=======================

A new `Spectrum` can be generated from any data array, as long as you provide two key attributes:

  - `f0`: the starting frequency for this `Spectrum`
  - `df`: the frequency resolution of the `Spectrum`

From this information, any new `Spectrum` can be generated as follows::

    >>> from gwpy.spectrum import Spectrum
    >>> mydata = Spectrum([1,2,3,4,5,6,7,8,9,10], f0=0, df=1)
    >>> mydata
    <Spectrum object: name='None' f0=0 df=1>

===========================================================================
Calculating the `Spectrum` from a :class:`~gwpy.timeseries.core.TimeSeries`
===========================================================================

The canonical method for generating a `Spectrum` is to calculated either the power-spectral density or amplitude-spectral density of a :class:`~gwpy.timeseries.core.TimeSeries`, using either the :meth:`~gwpy.timeseries.core.TimeSeries.psd` or :meth:`~gwpy.timeseries.core.TimeSeries.asd` methods of that class::

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.read('G-G1_RDS_C01_L3-1049587200-60.gwf', 'G1:DER_DATA_H')
    >>> hoff = hoft.asd(2, fftstride=1, method='welch')

where the result is an average spectrum, in this examples using the `Welch method <https://en.wikipedia.org/wiki/Welch_method>`_.
The ``fftlength`` and ``fftstride`` arguments are the length (in seconds) of each Fourier transform, and the stride between successive transforms in the average, respectively.

=============
Reference/API
=============

.. autoclass:: Spectrum
   :show-inheritance:
   :members: filter, plot
