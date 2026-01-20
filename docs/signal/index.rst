.. currentmodule:: gwpy.timeseries

.. _gwpy-signal-processing:

#################
Signal processing
#################

In a wide-array of applications, the original data recorded from a digital
system must be manipulated in order to extract the greatest amount of information.
GWpy provides a suite of functions to simplify and extend the excellent digital
signal processing suite in :mod:`scipy.signal`.

.. seealso::

    - **Reference**: :mod:`gwpy.signal`

===========================
Spectral density estimation
===========================

`Spectral density estimation <https://en.wikipedia.org/wiki/Spectral_density>`__
is a common way of investigating the frequency-domain content of a time-domain signal.
GWpy provides wrappers of power spectral density (PSD) estimation methods from
:mod:`scipy.signal` to simplify calculating a
:class:`~gwpy.frequencyseries.FrequencySeries` from a :class:`TimeSeries`.

The :mod:`gwpy.signal.spectral` sub-package provides the following
PSD estimation averaging methods:

- ``'bartlett'`` - mean average of non-overlapping periodograms
- ``'median'`` - median average of overlapping periodograms
- ``'welch'`` - mean average of overlapping periodograms

Each of these can be specified by passing the function name as the
``method`` keyword argument to any of the relevant :class:`TimeSeries`
instance methods:

.. autosummary::

    TimeSeries.psd
    TimeSeries.asd
    TimeSeries.spectrogram
    TimeSeries.spectrogram2

e.g, :meth:`TimeSeries.psd`::

    >>> ts = TimeSeries(...)
    >>> psd = ts.psd(..., method='median', ...)

See :func:`scipy.signal.welch` for more detailed documentation on the PSD
estimation method used.

=====================
Time-domain filtering
=====================

The `TimeSeries` object comes with a number of instance methods that should
make filtering data easy for a number of common use cases.
Available methods include:

.. autosummary::
    :nosignatures:

    TimeSeries.highpass
    TimeSeries.lowpass
    TimeSeries.bandpass
    TimeSeries.zpk
    TimeSeries.whiten
    TimeSeries.filter

Each of the above methods eventually calls out to :meth:`TimeSeries.filter` to
apply a digital linear filter, normally via cascaded second-order-sections.

For a worked example of how to filter LIGO data to discover a gravitational-wave signal,
see :ref:`sphx_glr_examples_signal_gw150914.py`.

==========================
Frequency-domain filtering
==========================

Additionally, the `TimeSeries` object includes a number of instance methods to
generate frequency-domain information for some data.
Available methods include:

.. autosummary::
    :nosignatures:

    TimeSeries.psd
    TimeSeries.asd
    TimeSeries.spectrogram
    TimeSeries.q_transform
    TimeSeries.rayleigh_spectrum
    TimeSeries.rayleigh_spectrogram

For a worked example of how to load data and calculate the Amplitude Spectral
Density `~gwpy.frequencyseries.FrequencySeries`,
see :ref:`sphx_glr_examples_frequencyseries_hoff.py`.

.. _gwpy-filter-design:

=============
Filter design
=============

:mod:`gwpy.signal.filter_design` provides a number of filter design methods which,
when combined with the `~gwpy.plot.BodePlot` visualisation,
can be used to create a number of common filters:

.. autosummary::
    :nosignatures:

    ~gwpy.signal.filter_design.lowpass
    ~gwpy.signal.filter_design.highpass
    ~gwpy.signal.filter_design.bandpass
    ~gwpy.signal.filter_design.notch
    ~gwpy.signal.filter_design.concatenate_zpks

Each of these will return filter coefficients that can be passed directly into
`~TimeSeries.filter`.

For a worked example of designing a digital filter, and then visualising it,
see :ref:`gwpy-plot-bode`.

==========================
Cross-channel correlations
==========================

The `TimeSeries` object also includes instance methods to calculate
cross-channel correlations in the frequency domain.
Available methods include:

.. autosummary::
    :nosignatures:

    TimeSeries.coherence
    TimeSeries.coherence_spectrogram

For a worked example of how to compare channels like this,
see :ref:`sphx_glr_examples_frequencyseries_coherence.py`.

.. currentmodule:: gwpy.signal
