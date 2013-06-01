#########################################
Time series data (`gwpy.data.TimeSeries`)
#########################################

Getting started
===============

Each GW detector records the time-streams of thousands of data channels in gravitational wave frame files. Data from one of these files can be read into a `~gwpy.data.TimeSeries` object as follows::

    >>> from gwpy.data import TimeSeries
    >>> hoft = TimeSeries.read('G-G1_RDS_C01_L3-1049587200-60.gwf', 'G1:DER_DATA_H')

The first argument, `'G-G1_RDS_C01_L3-1049587200-60.gwf'` is the path to the GWF file, and the second, `'G1:DER_DATA_H'` is the name of the data channel of interest.

Each `~gwpy.data.TimeSeries` records the data for the channel along with the following metadata:

  - `epoch`: the `~gwpy.time.Time` (in GPS format) marking the start of the data,
  - `dt`: the time between successive samples,
  - `units`: the units of the data,
  - `name`: the name for this `~gwpy.data.TimeSeries`, normally the data channel name.

Each of these can be accessed, including `data`, as attributes::

    >>> hoft.data
    array([ -4.54581312e-12,  -4.54459216e-12,  -4.54321471e-12, ...,
            -5.42630392e-12,  -5.42534292e-12,  -5.42453795e-12])
    >>> hoft.epoch
    <Time object: scale='utc' format='gps' vals=1049587200.0>
    >>> hoft.dt
    6.103515625e-05

Calculating spectra
===================

The (average) spectrum for any `~gwpy.data.TimeSeries` can be calculated using one of the `~gwpy.data.TimeSeries.psd` (power spectral density) or `~gwpy.data.TimeSeries.asd` (amplitude spectral density) methods, for example::

    >>> h_spectrum = hoft.asd(method='welch', fft_length=2, overlap=1)

where the result is an average spectrum, using one of the `Bartlett`, `Welch` or `median-mean` average spectrum methods.
Here `fft_length` and `overlap` are the lengths (in seconds) of each Fourier transform and the overlap between successive transforms respectively.

Information on the resulting `~gwpy.data.Spectrum` object can be found in (REF).

Reference/API
=============

.. currentmodule:: gwpy.data.series

.. autoclass:: TimeSeries
   :show-inheritance:
   :members: get_times, psd, asd, spectrogram, plot, read, fetch
