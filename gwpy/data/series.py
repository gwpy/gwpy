# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Array with metadata
"""

from __future__ import division

import numbers
import numpy
from math import modf
from scipy import interpolate

from astropy import units

from .. import version
from ..utils import lal
from ..time import Time
from .nddata import NDData

if lal.SWIG_LAL:
    LIGOTimeGPS = lal.swiglal.LIGOTimeGPS
else:
    LIGOTimeGPS = numpy.float64

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["TimeSeries", "Spectrum", "Spectrogram"]


class TimeSeries(NDData):
    """A data array holding some metadata to represent a time series.

    The data for the series are held, along with the 'epoch'
    (series start time) and 'dt' (time between successive samples)
    The array of times is constructed on request from these attribtutes.
    """
    def __init__(self, data, epoch=None, dt=None, name=None, unit=None,
                 **kwargs):
        """Generate a new TimeSeries.

        Parameters
        ----------
        data : numpy.ndarray, list
            Data values to initialise TimeSeries
        epoch : GPS time, or `~gwpy.time.Time`, optional
            TimeSeries start time
        dt : float, optional
            Number of samples per second
        unit : `~astropy.units.Unit`, optional
            The units of the data

        Returns
        -------
        result : `~gwpy.types.TimeSeries`
            A new TimeSeries
        """
        super(TimeSeries, self).__init__(data, name=name, unit=unit, **kwargs)
        self.epoch = epoch
        self.dt = dt
        self.name = name

    @property
    def epoch(self):
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
    def dt(self):
        return self._meta['dt']
    @dt.setter
    def dt(self, val):
        if val is None:
            self._meta['dt'] = val
        else:
            self._meta['dt'] = units.Quantity(val, units.second)

    @property
    def sample_rate(self):
        return 1 / self._meta['dt']
    @sample_rate.setter
    def sample_rate(self, val):
        self.dt = 1 / val

    def get_times(self, dtype=LIGOTimeGPS):
        """Get the array of GPS times that accompany the data array

        Returns
        -------
        result : `~numpy.ndarray`
            1d array of GPS time floats
        """
        data = (numpy.arange(self.shape[0]) * self.dt +
                self.epoch.gps).astype(dtype)
        return NDData(data, unit=units.second)

    def psd(self, method='welch', **kwargs):
        from ..spectrum import psd
        psd_ = psd(self, method, **kwargs)
        if not hasattr(psd_.unit, "name"):
            psd_.unit.name = "Power spectral density"
        return psd_

    def asd(self, *args, **kwargs):
        asd = self.psd(*args, **kwargs)
        asd.data **= 1/2.
        asd.unit **= 1/2.
        if not hasattr(asd.unit, "name"):
            asd.unit.name = "Amplitude spectral density"
        return asd

    def spectrogram(self, step, method='welch', **kwargs):
        from ..spectrum import spectrogram
        spec_ = spectrogram(self, method, step, **kwargs)
        if not hasattr(spec_.unit, 'name'):
            spec_.unit.name = "Power spectral density"
        return spec_

    def plot(self, **kwargs):
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
