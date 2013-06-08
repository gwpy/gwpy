# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Array with metadata
"""

from __future__ import division

import numbers
import numpy
from math import modf
from scipy import (interpolate, signal)

from astropy import units

from .. import version
from ..utils import lal
from ..time import Time
from ..detector import Channel
from ..segments import Segment
from .nddata import NDData

if lal.SWIG_LAL:
    LIGOTimeGPS = lal.swiglal.LIGOTimeGPS
else:
    LIGOTimeGPS = numpy.float64

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["TimeSeries", "Spectrum", "Spectrogram"]


class TimeSeries(NDData):
    """A data array holding some metadata to represent a time series of
    instrumental or analysis data.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Data values to initialise TimeSeries
    epoch : `float` GPS time, or `~gwpy.time.Time`, optional
        TimeSeries start time
    channel : `~gwpy.detector.Channel`, or `str`, optional
        Data channel for this TimeSeries
    unit : `~astropy.units.Unit`, optional
        The units of the data

    Returns
    -------
    result : `~gwpy.types.TimeSeries`
        A new TimeSeries

    Notes
    -----
    Any regular array, i.e. any iterable collection of data, can be
    easily converted into a `TimeSeries`.

    >>> data = numpy.asarray([1,2,3])
    >>> series = TimeSeries(data)

    The necessary metadata to reconstruct timing information are recorded
    in the `epoch` and `sample_rate` attributes. This array can be
    calculated via the `get_times` method.

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
        return Segment(epoch.gps, epoch.gps + self.data.size * self.dt)

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
    def fetch(cls, channel, start, end, host=None, port=None):
        """Fetch data from NDS into a TimeSeries

        Parameters
        ----------
        channel : `~gwpy.detector.Channel`, or `str`
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
        `TimeSeries`
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

class Spectrum(NDData):
    """A data array holding some metadata to represent a spectrum.

    """
    def __init__(self, data, f0=None, df=None, name=None, logscale=False,
                 unit=None, **kwargs):
        """Generate a new Spectrum.

        Parameters
        ----------
        data : numpy.ndarray, list
            Data values to initialise FrequencySeries
        f0 : `float`
            Starting frequency for this series
        df : float, optional
            Frequency resolution (Hertz)
        name : str, optional
            Name for this Spectrum
        unit : `~astropy.units.Unit`, optional
            The units of the data

        Returns
        -------
        result : `~gwpy.types.TimeSeries`
            A new TimeSeries
        """
        super(Spectrum, self).__init__(data, name=name, unit=unit, **kwargs)
        self.f0 = f0
        self.df = df
        self.logscale = logscale

    @property
    def f0(self):
        return self._meta['f0']
    @f0.setter
    def f0(self, val):
        if val is None:
            self._meta['f0'] = val
        else:
            self._meta['f0'] = units.Quantity(val, units.Hertz)

    @property
    def df(self):
        return self._meta['df']
    @df.setter
    def df(self, val):
        if val is None:
            self._meta['df'] = None
        else:
            self._meta['df'] = units.Quantity(val, units.Hertz)

    @property
    def logscale(self):
        return self._meta['logscale']
    @logscale.setter
    def logscale(self, val):
        self._meta['logscale'] = bool(val)

    def get_frequencies(self):
        """Get the array of frequencies that accompany the data array

        Returns
        -------
        result : `~gwpy.types.NDData`
            1d array of frequencies in Hertz
        """
        if self.logscale:
            logdf = numpy.log10(self.f0 + self.df) - numpy.log10(self.f0)
            logf1 = numpy.log10(self.f0) + self.shape[-1] * logdf
            data = numpy.logspace(numpy.log10(self.f0), logf1,
                                  num=self.shape[-1])
        else:
            data = numpy.arange(self.shape[-1]) * self.df + self.f0
        return NDData(data, unit=units.Hertz)

    def to_logscale(self, fmin=None, fmax=None, num=None):
        """Convert this Spectrum into logarithmic scale.
        """
        num = num or self.shape[-1]
        fmin = fmin or float(self.f0) or float(self.f0 + self.df)
        fmax = fmax or float(self.f0 + self.shape[-1] * self.df)
        linf = self.get_frequencies().data
        logf = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), num=num)
        logf = logf[logf<linf.max()]
        interpolator = interpolate.interp1d(linf, self.data, axis=0)
        new = self.__class__(interpolator(logf), unit=self.unit,
                             wcs=self.wcs, uncertainty=self.uncertainty,
                             flags=self.flags, meta=self.meta)
        for attr in self._getAttributeNames():
            setattr(new, attr, getattr(self, attr))
        new.f0 = logf[0]
        new.df = logf[1]-logf[0]
        new.logscale = True
        return new

    def plot(self, **kwargs):
        from ..plotter import SpectrumPlot
        return SpectrumPlot(self, **kwargs)

    def __str__(self):
        return "Spectrum('{0}')".format(self.name)

    def __repr__(self):
        return "<Spectrum object: name='{0}' f0={1} df={2}>".format(
                   self.name, self.f0, self.df)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        new = super(Spectrum, self).__getitem__(item)
        new.f0 = self.f0
        new.df = self.df
        new.name = self.name
        if item.start:
            new.f0 = self.f0 + item.start * self.df
        if item.step:
            new.df = self.df * item.step
        return new

class Spectrogram(Spectrum, TimeSeries):
    """A 2-dimensional array holding a spectrogram of time-frequency
    amplitude.
    """
    def __init__(self, data, epoch=None, f0=None, dt=None, df=None, name=None,
                 logscale=False, unit=None, **kwargs):
        """Generate a new Spectrum.

        Parameters
        ----------
        data : numpy.ndarray, list
            Data values to initialise FrequencySeries
        epoch : GPS time, or `~gwpy.time.Time`, optional
            TimeSeries start time
        f0 : `float`
            Starting frequency for this series
        dt : float, optional
            Time between samples (second)
        df : float, optional
            Frequency resolution (Hertz)
        name : str, optional
            Name for this Spectrum
        unit : `~astropy.units.Unit`, optional
            The units of the data

        Returns
        -------
        result : `~gwpy.types.TimeSeries`
            A new TimeSeries
        """
        super(Spectrogram, self).__init__(data,  name=name, unit=unit, **kwargs)
        self.epoch = epoch
        self.dt = dt
        self.f0 = f0
        self.df = df
        self.logscale = logscale

    def ratio(self, operand):
        """Calculate the ratio of this Spectrogram against a
        reference.

        Parameters
        ----------
        operand : str, Spectrum
            Spectrum against which to weight, or one of
            - 'mean' : weight against the mean of each spectrum
                       in this Spectrogram
            - 'median' : weight against the median of each spectrum
                       in this Spectrogram

        Returns
        -------
        R : `~gwpy.data.Spectrogram`
            A new spectrogram
        """
        if operand == 'mean':
            operand = self.mean(axis=0)
            unit = units.dimensionless_unscaled
        elif operand == 'median':
            operand = self.median(axis=0)
            unit = units.dimensionless_unscaled
        elif isinstance(operand, Spectrum):
            operand = Spectrum.data
            unit = self.unit / Spectrum.unit
        elif isinstance(operand, numbers.Number):
            unit = units.dimensionless_unscaled
        else:
            raise ValueError("operand '%s' unrecognised, please give Spectrum "
                             "or one of: 'mean', 'median'")
        return self.__class__(self.data/operand, unit=unit, epoch=self.epoch,
                              f0=self.f0, name=self.name, dt=self.dt,
                              df=self.df, logscale=self.logscale, wcs=self.wcs,
                              mask=self.mask, flags=self.flags, meta=self.meta)

    def __str__(self):
        return "Spectrogram('{0}')".format(self.name)

    def __repr__(self):
        return ("<Spectrogram object: name='{0}' epoch={1} f0={2} "
                "dt={3} df={4}>".format(
                    self.name, self.epoch, self.f0, self.dt, self.df))

    def plot(self, **kwargs):
        from ..plotter import SpectrogramPlot
        return SpectrogramPlot(self, **kwargs)

    def to_logscale(self, fmin=None, fmax=None, num=None):
        """Convert this Spectrum into logarithmic scale.
        """
        num = num or self.shape[-1]
        fmin = fmin or float(self.f0) or float(self.f0 + self.df)
        fmax = fmax or float(self.f0 + self.shape[-1] * self.df)
        linf = self.get_frequencies().data
        logf = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), num=num)
        logf = logf[logf<linf.max()]
        new = self.__class__(numpy.zeros((self.shape[0], logf.size)),
                             unit=self.unit,
                             wcs=self.wcs, uncertainty=self.uncertainty,
                             flags=self.flags, meta=self.meta)
        for i in range(self.shape[0]):
            interpolator = interpolate.interp1d(linf[-logf.size:],
                                                self.data[i,-logf.size:],
                                                axis=0)
            new.data[i,:] = interpolator(logf)
        for attr in self._getAttributeNames():
            setattr(new, attr, getattr(self, attr))
        new.f0 = logf[0]
        new.df = logf[1]-logf[0]
        new.logscale = True
        return new

    def __getitem__(self, item):
        if isinstance(item, int):
            return Spectrum(self.data[item], wcs=self.wcs, unit=self.unit,
                            name=self.name, f0=self.f0, df=self.df)
        elif isinstance(item, tuple):
            return self[item[0]][item[1]]
        new = super(Spectrogram, self).__getitem__(item)
        new.f0 = self.f0
        new.df = self.df
        new.name = self.name
        if isinstance(item, tuple):
            return new
        if item.start:
            new.f0 = self.f0 + item.start * self.df
        if item.step:
            new.df = self.df * item.step
        return new
