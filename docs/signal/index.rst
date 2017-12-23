.. currentmodule:: gwpy.timeseries

.. _gwpy-signal-processing:

#################
Signal processing
#################

In a wide-array of applications, the original data recorded from a digital system must be manipulated in order to extract the greatest amount of information.
GWpy provides a suite of functions to simplify and extend the excellent digital signal processing suite in :mod:`scipy.signal`.

===============
Fourier methods
===============

The `TimeSeries` object comes with a number of Fourier methods to calculate a `~gwpy.frequencyseries.FrequencySeries` or `~gwpy.spectrogram.Spectrogram` by calculating and averaging FFTs.

See :ref:`gwpy-signal-fft` for more details.

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

For a worked example of how to filter LIGO data to discover a gravitational-wave signal, see the example :ref:`gwpy-example-signal-gw150914`.

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

For a worked example of how to load data and calculate the Amplitude Spectral Density `~gwpy.frequencyseries.FrequencySeries`, see the example :ref:`gwpy-example-frequencyseries-hoff`.

.. _gwpy-filter-design:

=============
Filter design
=============

The :mod:`gwpy.signal` provides a number of filter design methods which, when combined with the `~gwpy.plotter.BodePlot` visualisation, can be used to create a number of common filters:

.. autosummary::
   :nosignatures:

   ~gwpy.signal.filter_design.lowpass
   ~gwpy.signal.filter_design.highpass
   ~gwpy.signal.filter_design.bandpass
   ~gwpy.signal.filter_design.notch
   ~gwpy.signal.filter_design.concatenate_zpks

Each of these will return filter coefficients that can be passed directly into `~TimeSeries.zpk` (default for analogue filters) or `~TimeSeries.filter` (default for digital filters).

For a worked example of how to filter LIGO data to discover a gravitational-wave signal, see the example :ref:`gwpy-example-signal-gw150914`.

**Cross-channel correlations:**

.. autosummary::
   :nosignatures:

   TimeSeries.coherence
   TimeSeries.coherence_spectrogram

For a worked example of how to compare channels like this, see the example :ref:`gwpy-example-frequencyseries-coherence`.

.. currentmodule:: gwpy.signal

=============
Reference/API
=============

--------------------------------
:mod:`gwpy.signal.filter_design`
--------------------------------

.. automethod:: gwpy.signal.filter_design.bandpass

.. automethod:: gwpy.signal.filter_design.lowpass

.. automethod:: gwpy.signal.filter_design.highpass

.. automethod:: gwpy.signal.filter_design.notch

.. automethod:: gwpy.signal.filter_design.concatenate_zpks
