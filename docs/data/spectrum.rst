####################################
Spectral data (:mod:`gwpy.spectrum`)
####################################

.. currentmodule:: gwpy.spectrum.core

While gravitational-wave detectors are time-domain instruments, their sensitivity is often measured as a power-spectral-density over the range of interesting GW frequencies (~10-10,000 Hz).

==============================
Generating a :class:`Spectrum`
==============================

A new :class:`Spectrum` can be generated from any data array, as long as you provide two key attributes:

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

.. currentmodule:: gwpy.timeseries.core
The frequency-spectrum of a :class:`TimeSeries` can be calculated using either of the following methods:

.. autosummary::
   :nosignatures:

   TimeSeries.psd
   TimeSeries.asd

For example:

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch('H1:LDAS-STRAIN', 966211215, 966211815)
    >>> hoff = hoft.asd(2, fftstride=1, method='welch')

where the result is an average spectrum calculated using the `Welch method <https://en.wikipedia.org/wiki/Welch_method>`_.
The ``fftlength`` and ``fftstride`` arguments are the length (in seconds) of each Fourier transform, and the stride between successive transforms in the average, respectively.
