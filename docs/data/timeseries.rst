#########################################
Time series data (`gwpy.data.TimeSeries`)
#########################################

.. currentmodule:: gwpy.data.timeseries

Gravitational-wave detectors are time-domain instruments, attempting to record gravitational wave amplitude as a strain. As a result, the basic data object is the `TimeSeries`.

Generating `TimeSeries`
=======================

`TimeSeries` can be generated from any data array, as long as you provide two key attributes:

  - `epoch`: the :class:`~gwpy.time.Time` marking the start of this `TimeSeries`
  - `sample_rate`: the number of data points per second

From this information, any new `TimeSeries` can be generated as follows::

    >>> from gwpy.time import Time
    >>> from gwpy.data import TimeSeries
    >>> mydata = TimeSeries([1,2,3,4,5,6,7,8,9,10], epoch=Time('2013-01-01', scale='utc'), sample_rate=1)
    >>> mydata
    <TimeSeries object: name='None' epoch=2013-01-01 00:00:00.000 dt=1.0 s>


===================
The GWF file format
===================

Each detector archives the time-streams of a huge number of channels in Gravitational-Wave Frame files (`.gwf` file extension). The format of these files is defined in `LIGO-T970130 <https://dcc.ligo.org/LIGO-T970130/public>`_.

Each of these frame files contains the amplitude time-series for a given `Channel`, and the important metadata, including:
  - `channel`: the name of the `~gwpy.detector.Channel` whose data have been recorded
  - `epoch`: the `~gwpy.time.Time` (in GPS format) marking the start of the data,
  - `sample_rate`: the number of samples per second
  - `units`: the units of the data, both for the time-stamps, and the amplitude

Data from one of these files can be read into a `TimeSeries` object as follows::

    >>> from gwpy.data import TimeSeries
    >>> hoft = TimeSeries.read('G-G1_RDS_C01_L3-1049587200-60.gwf', 'G1:DER_DATA_H')

The first argument, `'G-G1_RDS_C01_L3-1049587200-60.gwf'` is the path to the GWF file, and the second, `'G1:DER_DATA_H'` is the name of the data channel of interest. A quick examination of the output shows the extracted attributes::

    >>> hoft
    <TimeSeries object: name='G1:DER_DATA_H' epoch=1049587200.0 dt=6.103515625e-05 s>
    >>> hoft.channel
    Channel("G1:DER_DATA_H")
    >>> hoft.sample_rate
    <Quantity 16384.0 Hz>
    >>> hoft.data
    array([ -4.54581312e-12,  -4.54459216e-12,  -4.54321471e-12, ...,
            -5.42630392e-12,  -5.42534292e-12,  -5.42453795e-12])


===================
Network data access
===================

The LIGO and Virgo scientific collaborations use the Network Data Server (NDS) 2 to serve data, and instrument data can be fetched into a `TimeSeries` as follows::

    >>> from gwpy.data import TimeSeries
    >>> data = TimeSeries.fetch('H1:PSL-ISS_PDA_OUT_DQ', 1054684816, 1054684826)

=============
Reference/API
=============

.. autoclass:: TimeSeries
   :show-inheritance:
   :members: read, fetch, get_times, psd, asd, spectrogram, plot, fetch
