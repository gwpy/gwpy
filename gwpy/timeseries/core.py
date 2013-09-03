# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Array with metadata
"""

from __future__ import division

import numbers
import numpy
from math import modf
from scipy import signal

from astropy import units

import lal
from lal import utils as lalutils
from lalframe import frread

from .. import version
from ..time import Time
from ..detector import Channel
from ..segments import Segment
from ../data/nddata import NDData

LIGOTimeGPS = lal.LIGOTimeGPS

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["TimeSeries"]


class TimeSeries(NDData):
    """A data array holding some metadata to represent a time series of
    instrumental or analysis data.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Data values to initialise TimeSeries
    epoch : `float` GPS time, or :class:`~gwpy.time.Time`, optional
        TimeSeries start time
    channel : :class:`~gwpy.detector.Channel`, or `str`, optional
        Data channel for this TimeSeries
    unit : :class:`~astropy.units.Unit`, optional
        The units of the data

    Returns
    -------
    TimeSeries
        a new `TimeSeries`

    Notes
    -----
    Any regular array, i.e. any iterable collection of data, can be
    easily converted into a `TimeSeries`.

    >>> data = numpy.asarray([1,2,3])
    >>> series = TimeSeries(data)

    The necessary metadata to reconstruct timing information are recorded
    in the `epoch` and `sample_rate` attributes. This time-stamps can be
    calculated via the :meth:`~TimeSeries.get_times` method.

    Attributes
    ----------
    name
    epoch
    channel
    unit
    sample_rate

    Methods
    -------
    get_times
    psd
    asd
    spectrogram
    plot
    read
    fetch
    resample
    """
    def __init__(self, data, epoch=None, channel=None, unit=None,
                 sample_rate=None, name=None, **kwargs):
        """Generate a new TimeSeries.
        """
        super(TimeSeries, self).__init__(data, name=name, unit=unit, **kwargs)
        self.channel = Channel(channel)
        self.epoch = epoch
        self.unit = unit
        self.sample_rate = (sample_rate and sample_rate or
                            channel and channel.sample_rate or None)
        self.name = name and name or channel and channel.name or None
        """Test Name for this TimeSeries"""

    @property
    def epoch(self):
        """Starting GPS time epoch for this `TimeSeries`.

        This attribute is recorded as a `~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See `~astropy.time` for details on the `Time` object.
        """
        return self._meta['epoch']
    @epoch.setter
    def epoch(self, epoch):
        if epoch is not None and not isinstance(epoch, Time):
            if hasattr(epoch, "seconds"):
                epoch = [epoch.seconds, epoch.nanoseconds*1e-9]
            elif hasattr(epoch, "gpsSeconds"):
                epoch = [epoch.gpsSeconds, epoch.gpsNanoSeconds*1e-9]
            else:
                epoch = modf(epoch)[::-1]
            epoch = Time(*epoch, format='gps', precision=6)
        self._meta['epoch'] = epoch

    @property
    def unit(self):
        """Unit of the data in this `TimeSeries`.
        """
        return self._unit
    @unit.setter
    def unit(self, u):
        if u is None:
            self._unit = None
        else:
            self._unit = units.Unit(u)

    @property
    def dt(self):
        """Time between samples for this `TimeSeries`.
        """
        return (1 / self.sample_rate).decompose()
    @dt.setter
    def dt(self, val):
        if val is None:
            self.sample_rate = None
        else:
            self.sample_rate = 1 / float(val)

    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).
        """
        try:
            return self._meta['sample_rate']
        except:
            return self.channel.sample_rate
    @sample_rate.setter
    def sample_rate(self, val):
        if val is not None:
            val = units.Quantity(val, units.Hertz)
        self._meta['sample_rate'] = val

    @property
    def channel(self):
        """Data channel associated with this `TimeSeries`.
        """
        return self._channel
    @channel.setter
    def channel(self, ch):
        self._channel = Channel(ch)

    @property
    def span(self):
        """Time Segment encompassed by thie `TimeSeries`.
        """
        return Segment(self.epoch.gps,
                       self.epoch.gps + self.data.size * float(self.dt))

    def is_contiguous(self, other):
        """Check whether other is contiguous with self.
        """
        seg1 = self.span
        seg2 = other.span
        if seg1[1] == seg2[0]:
            return True
        else:
            return False

    def is_compatible(self, other):
        """Check whether metadata attributes for self and other match.
        """
        if not self.sample_rate == other.sample_rate:
            raise ValueError("TimeSeries sampling rates do not match.")
        if not self.unit == other.unit:
            raise ValueError("TimeSeries units do not match")
        if not self.unit == other.unit:
            raise ValueError("TimeSeries units do not match")
        return True

    def __iand__(self, other):
        """Append the other TimeSeries to self.
        """
        if not self.is_contiguous(other):
            raise ValueError("TimeSeries are not contiguous")
        self.is_compatible(other)
        shape = self.data.shape
        self.data.resize((shape[0], shape[1] + other.data.size))
        self.data[shape[1]:] = other.data
        return

    def __and__(self, other):
        """Return a `TimeSeries` from the combination of self and other
        """
        if not self.is_contiguous(other):
            raise ValueError("TimeSeries are not contiguous")
        self.is_compatible(other)
        return self.__class__(numpy.concantenate((self.data, other.data)),
                              epoch=self.epoch, sample_rate=self.sample_rate,
                              unit=self.unit, channel=self.channel)

    def get_times(self, dtype=LIGOTimeGPS):
        """Get the array of GPS times that accompany the data array

        Parameters
        ----------
        dtype : `type`, optional
            return data type, defaults to `LIGOTimeGPS` if available,
            otherwise, `~numpy.float64`

        Returns
        -------
        result : `~numpy.ndarray`
            1d array of GPS time floats
        """
        data = (numpy.arange(self.shape[0]) * self.dt +
                self.epoch.gps).astype(dtype)
        return NDData(data, unit=units.second)

    def psd(self, method='welch', **kwargs):
        """Calculate the power spectral density (PSD) `Spectrum` for this
        `TimeSeries`.

        The `method` argument can be one of
            * 'welch'
            * 'bartlett'
        and any keyword arguments will be passed to the relevant method
        in `gwpy.spectrum`.

        Parameters
        ----------
        method : `str`, defaults to `'welch'`
            name of average spectrum method
        **kwargs
            other keyword arguments passed to the average spectrum method,
            see the documentation for each method in `gwpy.spectrum` for
            details

        Returns
        -------
        psd :  `~gwpy.series.Spectrum`
            a data series containing the PSD.
        """
        from ..spectrum import psd
        psd_ = psd(self, method, **kwargs)
        if not hasattr(psd_.unit, "name"):
            psd_.unit.name = "Power spectral density"
        return psd_

    def asd(self, *args, **kwargs):
        """Calculate the amplitude spectral density (PSD)
        `Spectrum` for this `TimeSeries`.

        All `*args` and `**kwargs` are passed directly to the
        `Timeseries.psd` method, with the return converted into
        an amplitude series.

        Returns
        -------
        asd :  `~gwpy.series.Spectrum`
            a data series containing the ASD.
        """

        asd = self.psd(*args, **kwargs)
        asd.data **= 1/2.
        asd.unit **= 1/2.
        if not hasattr(asd.unit, "name"):
            asd.unit.name = "Amplitude spectral density"
        return asd

    def spectrogram(self, step, method='welch', **kwargs):
        """Calculate the power `Spectrogram` for this `TimeSeries`.

        This method wraps the `spectrogram` method.
        """
        from ..spectrum import spectrogram
        spec_ = spectrogram(self, method, step, **kwargs)
        if not hasattr(spec_.unit, 'name'):
            spec_.unit.name = "Power spectral density"
        return spec_

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    def __str__(self):
        return "TimeSeries('{0}', epoch={1})".format(self.name, self.epoch)

    def __repr__(self):
        return "<TimeSeries object: name='{0}' epoch={1} dt={2}>".format(
                   self.name, self.epoch, self.dt)

    def __getitem__(self, item):
        new = super(TimeSeries, self).__getitem__(item)
        new.epoch = self.epoch
        new.dt = self.dt
        new.name = self.name
        if item.start:
            new.epoch = LIGOTimeGPS(self.epoch.gps) + float(item.start*self.dt)
        if item.step:
            new.dt = self.dt * item.step
        return new

    @classmethod
    def from_lal(cls, lalts):
        """Generate a new TimeSeries from a LAL TimeSeries of any type
        """
        # write Channel
        channel = Channel(lalts.name, 1/lalts.deltaT,
                          unit=lal.UnitToString(lalts.sampleUnits),
                          dtype=lalts.data.data.dtype)
        return cls(lalts.data.data, channel=channel, epoch=lalts.epoch,
                   unit=lal.UnitToString(lalts.sampleUnits))

    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries
        """
        laltype = lalutils.LAL_TYPE_FROM_NUMPY[self.dtype.type]
        typestr = lalutils.LAL_TYPE_STR[laltype]
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       float(self.dt), lal.lalDimensionlessUnit, self.size)
        lalts.data.data = self.data
        return lalts

    @classmethod
    def read(cls, source, channel, epoch=None, duration=None, datatype=None,
             verbose=False):
        """Read data into a `TimeSeries` from files on disk

        Parameters
        ----------
        source : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
            source for data, one of:

            - a filepath for a GWF-format frame file,
            - a filepath for a LAL-format Cache file
            - a Cache object from GLUE or LAL

        channel : `str`, :class:`~gwpy.detector.channel.Channel`
            channel (name or object) to read
        epoch : :class:`~gwpy.time.Time`, optional
            start time of desired data
        duration : `float`, optional
            duration of desired data
        datatype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type
        verbose : `bool`, optional
            print verbose output

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from disk
        """
        if isinstance(channel, Channel):
            channel = channel.name
            if datatype is None:
                datatype = channel.dtype
        if epoch and isinstance(epoch, Time):
            epoch = epoch.gps
        lalts = frread.read_timeseries(source, channel, start=epoch,
                                       duration=duration, datatype=datatype,
                                       verbose=verbose)
        return cls.from_lal(lalts)

    @classmethod
    def fetch(cls, channel, start, end, host=None, port=None):
        """Fetch data from NDS into a TimeSeries

        Parameters
        ----------
        channel : :class:`~gwpy.detector.channel.Channel`, or `str`
            required data channel
        start : `~gwpy.time.Time`, or float
            GPS start time of data span
        end : `~gwpy.time.Time`, or float
            GPS end time of data span
        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host
        port : `int`, optional
            port number for NDS server query, must be given with `host`

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from NDS
        """
        from ..io import nds
        channel = Channel(channel)
        if not host or port:
            dhost,dport = nds.DEFAULT_HOSTS[channel.ifo]
            host = host or dhost
            port = port or dport
        with nds.NDSConnection(host, port) as connection:
            return connection.fetch(start, end, channel)

    def resample(self, rate, window=None):
        """Resample this TimeSeries to a new rate

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `TimeSeries`
        window : array_like, callable, string, float, or tuple, optional
            specifies the window applied to the signal in the Fourier
            domain.

        Returns
        -------
        resampled_timeseries
            a new TimeSeries with the resampling applied, and the same
            metadata
        """
        N = self.size / self.sample_rate * rate
        data = signal.resample(self.data, N, window=window)
        return TimeSeries(data, channel=self.channel, unit=self.unit,
                          sample_rate=rate, epoch=self.epoch)

    def highpass(self, frequency, amplitude=0.9, order=8):
        """Filter this `TimeSeries` with a Butterworth high-pass filter

        See (for example) :lalsuite:`XLALHighPassREAL8TimeSeries` for more
        information.

        Parameters
        ----------
        frequency : `float`
            minimum frequency for high-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter

        Returns
        -------
        TimeSeries
        """
        lalts = self.to_lal()
        highpass = getattr(lal, 'HighPass%s' % lalts.__class__.__name__)
        highpass(lalts, float(frequency), amplitude, order)
        return TimeSeries.from_lal(lalts)

    def lowpass(self, frequency, amplitude=0.9, order=8):
        """Filter this `TimeSeries` with a Butterworth low-pass filter

        See (for example) :lalsuite:`XLALLowPassREAL8TimeSeries` for more
        information.

        Parameters
        ----------
        frequency : `float`
            minimum frequency for low-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter

        Returns
        -------
        TimeSeries
        """
        lalts = self.to_lal()
        lowpass = getattr(lal, 'LowPass%s' % lalts.__class__.__name__)
        lowpass(lalts, float(frequency), amplitude, order)
        return TimeSeries.from_lal(lalts)

    def bandpass(self, flow, fhigh, amplitude=0.9, order=8):
        """Filter this `TimeSeries` by applying both low- and high-pass
        filters.

        See (for example) :lalsuite:`XLALLowPassREAL8TimeSeries` for more
        information.

        Parameters
        ----------
        flow : `float`
            minimum frequency for high-pass
        fhigh : `float`
            maximum frequency for low-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter

        Returns
        -------
        TimeSeries
        """
        high = self.highpass(flow, amplitude=amplitude, order=order)
        return high.lowpass(fhigh, amplitude=amplitude, order=order)



