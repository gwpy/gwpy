.. currentmodule:: pydischarge.timeseries

.. _pydischarge-signal-processing:

#################
Signal processing
#################

In a wide-array of applications, the original data recorded from a digital system must be manipulated in order to extract the greatest amount of information.
pyDischarge provides a suite of functions to simplify and extend the excellent digital signal processing suite in :mod:`scipy.signal`.

===========================
Spectral density estimation
===========================

`Spectral density estimation <https://en.wikipedia.org/wiki/Spectral_density>`_
is a common way of investigating the frequency-domain content of a time-domain
signal.
pyDischarge provides wrappers of power spectral density (PSD) estimation methods
from :mod:`scipy.signal` to simplify calculating a
:class:`~pydischarge.frequencyseries.FrequencySeries` from a :class:`TimeSeries`.

The :mod:`pydischarge.signal.spectral` sub-package provides the following
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

The `TimeSeries` object comes with a number of instance methods that should make filtering data trivial for a number of common use cases.
Available methods include:

.. autosummary::
   :nosignatures:

   TimeSeries.highpass
   TimeSeries.lowpass
   TimeSeries.bandpass
   TimeSeries.zpk
   TimeSeries.whiten
   TimeSeries.filter

Each of the above methods eventually calls out to :meth:`TimeSeries.filter` to apply a digital linear filter, normally via cascaded second-order-sections (requires `scipy >= 0.16`).

For a worked example of how to filter LIGO data to discover a gravitational-wave signal, see the example :ref:`pydischarge-example-signal-gw150914`.

==========================
Frequency-domain filtering
==========================

Additionally, the `TimeSeries` object includes a number of instance methods to generate frequency-domain information for some data.
Available methods include:

.. autosummary::
   :nosignatures:

   TimeSeries.psd
   TimeSeries.asd
   TimeSeries.spectrogram
   TimeSeries.q_transform
   TimeSeries.rayleigh_spectrum
   TimeSeries.rayleigh_spectrogram

For a worked example of how to load data and calculate the Amplitude Spectral Density `~pydischarge.frequencyseries.FrequencySeries`, see the example :ref:`pydischarge-example-frequencyseries-hoff`.

.. _pydischarge-filter-design:

=============
Filter design
=============

The :mod:`pydischarge.signal` provides a number of filter design methods which, when combined with the `~pydischarge.plot.BodePlot` visualisation, can be used to create a number of common filters:

.. autosummary::
   :nosignatures:

   ~pydischarge.signal.filter_design.lowpass
   ~pydischarge.signal.filter_design.highpass
   ~pydischarge.signal.filter_design.bandpass
   ~pydischarge.signal.filter_design.notch
   ~pydischarge.signal.filter_design.concatenate_zpks

Each of these will return filter coefficients that can be passed directly into `~TimeSeries.zpk` (default for analogue filters) or `~TimeSeries.filter` (default for digital filters).

For a worked example of how to filter LIGO data to discover a gravitational-wave signal, see the example :ref:`pydischarge-example-signal-gw150914`.

**Cross-channel correlations:**

.. autosummary::
   :nosignatures:

   TimeSeries.coherence
   TimeSeries.coherence_spectrogram

For a worked example of how to compare channels like this, see the example :ref:`pydischarge-example-frequencyseries-coherence`.

.. currentmodule:: pydischarge.signal

=============
Reference/API
=============

.. automethod:: pydischarge.signal.filter_design.bandpass

.. automethod:: pydischarge.signal.filter_design.lowpass

.. automethod:: pydischarge.signal.filter_design.highpass

.. automethod:: pydischarge.signal.filter_design.notch

.. automethod:: pydischarge.signal.filter_design.concatenate_zpks
