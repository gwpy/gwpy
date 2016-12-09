.. _signal-processing:

.. currentmodule:: gwpy.timeseries

#################
Signal processing
#################

In a wide-array of applications, the original `TimeSeries` recorded from a digital system must be manipulated in order to extract the greatest amount of information.
GWpy provides a suite of functions to simplify and extend the excellent digital signal processing suite in :mod:`scipy.signal`.

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
   TimeSeries.filter
   TimeSeries.whiten

For a worked example of how to filter LIGO data to discover a gravitational-wave signal, see the :doc:`GW150914 example <../examples/signal/gw150914>`.

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

=============
Filter design
=============

The :mod:`gwpy.signal` provides a number of filter design methods which, when combined with the `~gwpy.plotter.BodePlot` visualisation, can be used to create a number of common filters:

.. autosummary::
   :nosignatures:

   ~gwpy.signal.lowpass
   ~gwpy.signal.highpass
   ~gwpy.signal.bandpass
   ~gwpy.signal.notch

Each of these will return filter coefficients that can be pass directly into `~TimeSeries.zpk` (default for analogue filters) or `~TimeSeries.filter` (default for digital filters).

For example, you can easily create, view, and apply a band-pass filter to some gravitational-wave data by combining the above methods.

We start by loading some data

.. plot::
   :context: reset
   :nofigs:
   :include-source:

   from gwpy.timeseries import TimeSeries
   data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

Now we can design and examine a bandpass filter to zone in on data between 40Hz nd 500Hz -- the main band of interest for a higher-mass binary black hole system (for example):

.. plot::
   :context:
   :include-source:

   from gwpy.signal import bandpass
   from gwpy.plotter import BodePlot
   f = bandpass(40, 1000, data.sample_rate)
   plot = BodePlot(f, sample_rate=data.sample_rate,
                   title='40-1000\,Hz bandpass filter')
   plot.show()

Next we can apply that filter to the `data` and look at the before-and-after `TimeSeries`:

.. plot::
   :context:
   :include-source:

   from gwpy.plotter import TimeSeriesPlot
   bp = data.filter(f).crop(1126259446+2, 1126259446-2)
   plot = TimeSeriesPlot(data.crop(1126259446+2, 1126259446-2), bp, sep=True)
   plot.show()

and can verify that the new `bp` series has the right spectral content:

.. plot::
   :context:
   :include-source:

   asd = data.asd(8, 4)
   bpasd = bp.asd(8, 4)
   plot = asd.plot(label='raw')
   plot.add_frequencyseries(bpasd, label='bandpass')
   plot.show()

**Cross-channel correlations:**

.. autosummary::
   :nosignatures:

   TimeSeries.coherence
   TimeSeries.coherence_spectrogram

For example:

For more examples like this, see :ref:`examples`.

.. currentmodule:: gwpy.signal

=============
Reference/API
=============

The :mod:`gwpy.signal` module provides the following methods:

.. autosummary::
   :nosignatures:

   bandpass
   lowpass
   highpass
   notch
   concatenate_zpks

.. automethod:: gwpy.signal.bandpass

.. automethod:: gwpy.signal.lowpass

.. automethod:: gwpy.signal.highpass

.. automethod:: gwpy.signal.notch

.. automethod:: gwpy.signal.concatenate_zpks
