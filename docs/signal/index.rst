.. _signal-processing:

.. currentmodule:: gwpy.timeseries

#################
Signal processing
#################

Oftentimes a `TimeSeries` is not the most informative way to look at data from a gravitational-wave interferometer.
GWpy provides convenient wrappers around some of the most common signal-processing methods.

**Time-domain filtering:**

.. autosummary::

   ~TimeSeries.highpass
   ~TimeSeries.lowpass
   ~TimeSeries.bandpass
   ~TimeSeries.zpk

**Frequency-domain transforms:**

.. autosummary::

   ~TimeSeries.psd
   ~TimeSeries.asd
   ~TimeSeries.spectrogram
   ~TimeSeries.q_transform
   ~TimeSeries.rayleigh_spectrum
   ~TimeSeries.rayleigh_spectrogram

**Cross-channel correlations:**

.. autosummary::

   ~TimeSeries.coherence
   ~TimeSeries.coherence_spectrogram

For example:

.. plot::

   from gwpy.timeseries import TimeSeries  # import the class
   data = TimeSeries.fetch_open_data('L1', 968654500, 968654600)  # fetch data from LOSC
   asd = data.asd(4, 2)  # calculated amplitude spectral density with 4-second FFT and 50% overlap
   plot = asd.plot()  # make plot
   ax = plot.gca()  # extract Axes
   ax.set_xlabel('Frequency [Hz]')  # set X-axis label
   ax.set_ylabel(r'ASD [strain/\rtHz]')  # set Y-axis label (requires latex)
   ax.set_xlim(40, 2000)  # set X-axis limits
   ax.set_ylim(8e-24, 5e-20)  # set Y-axis limits
   ax.set_title('Strain sensitivity of LLO during S6')  # set Axes title
   plot.show()  # show me the plot

|

For more examples like this, see :ref:`examples`.
