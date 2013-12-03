# coding=utf-8
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Spectrogram object
"""

import bisect
import numbers
import warnings
from math import log10

import numpy
import scipy
from scipy import (interpolate, signal)
from astropy import units

from ..detector import Channel
from ..time import Time
from ..data import (Array2D, Series)
from ..timeseries import (TimeSeries, TimeSeriesList)
from ..spectrum import Spectrum

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org"
__version__ = version.version

__all__ = ['Spectrogram', 'SpectrogramList']


def as_spectrum(func):
    def decorated_func(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        if isinstance(out, Series):
            out = Spectrum(out.data, name=out.name, unit=out.unit,
                           epoch=out.epoch, channel=out.channel,
                           f0=out.x0.value, df=out.dx.value)
        return out
    decorated_func.__doc__ = func.__doc__
    return decorated_func


class Spectrogram(Array2D):
    """A 2-dimensional array holding a spectrogram of time-frequency
    amplitude.

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
    _metadata_slots = ['name', 'unit', 'epoch', 'dt', 'f0', 'df', 'logf']
    xunit = TimeSeries.xunit
    yunit = Spectrum.xunit

    def __new__(cls, data, name=None, unit=None, channel=None, epoch=None,
                f0=None, dt=None, df=None, logf=None, **kwargs):
        """Generate a new Spectrogram.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
            dt = dt or 1 / channel.sample_rate
        # generate Spectrogram
        return super(Spectrogram, cls).__new__(cls, data, name=name, unit=unit,
                                               channel=channel, epoch=epoch,
                                               f0=f0, dt=dt, df=df,
                                               logf=logf, **kwargs)

    # -------------------------------------------
    # Spectrogram properties

    epoch = property(TimeSeries.epoch.__get__, TimeSeries.epoch.__set__,
                     TimeSeries.epoch.__delete__,
                     """Starting GPS epoch for this `Spectrogram`""")

    span = property(TimeSeries.span.__get__, TimeSeries.span.__set__,
                    TimeSeries.span.__delete__,
                    """GPS [start, stop) span for this `Spectrogram`""")

    dt = property(TimeSeries.dt.__get__, TimeSeries.dt.__set__,
                  TimeSeries.dt.__delete__,
                  """Time-spacing for this `Spectrogram`""")

    f0 = property(Array2D.y0.__get__, Array2D.y0.__set__,
                  Array2D.y0.__delete__,
                  """Starting frequency for this `Spectrogram`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    df = property(Array2D.dy.__get__, Array2D.dy.__set__,
                  Array2D.dy.__delete__,
                  """Frequency spacing of this `Spectogram`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    times = property(fget=Array2D.xindex.__get__,
                     fset=Array2D.xindex.__set__,
                     fdel=Array2D.xindex.__delete__,
                     doc="""Series of GPS times for each sample""")

    frequencies = property(fget=Array2D.yindex.__get__,
                           fset=Array2D.yindex.__set__,
                           fdel=Array2D.yindex.__delete__,
                           doc="Series of frequencies for this Spectrogram")

    logf = property(fget=Array2D.logy.__get__,
                    fset=Array2D.logy.__set__,
                    fdel=Array2D.logy.__delete__,
                    doc="""Switch determining a logarithmic frequency scale""")

    # -------------------------------------------
    # Spectrogram methods

    def crop(self, gpsstart, gpsend):
        """Crop this `Spectrogram` to the given GPS ``[start, end)``
        `Segment`.

        Parameters
        ----------
        gpsstart : `Time`, `float`
            GPS start time to crop `TimeSeries` at left
        gpsend : `Time`, `float`
            GPS end time to crop `TimeSeries` at right

        Returns
        -------
        timeseries : `TimeSeries`
            A new `TimeSeries` with the same metadata but different GPS
            span

        Notes
        -----
        If either ``gpsstart`` or ``gpsend`` are outside of the original
        `TimeSeries` span, warnings will be printed and the limits will
        be restricted to the :attr:`TimeSeries.span`
        """
        if isinstance(gpsstart, Time):
            gpsstart = gpsstart.gps
        if isinstance(gpsend, Time):
            gpsend = gpsend.gps
        if gpsstart < self.span[0]:
            warnings.warn('TimeSeries.crop given GPS start earlier than '
                          'start time of the input TimeSeries. Crop will '
                          'begin when the TimeSeries actually starts.')
            gpsstart = self.span[0]
        if gpsend > self.span[1]:
            warnings.warn('TimeSeries.crop given GPS end later than '
                          'end time of the input TimeSeries. Crop will '
                          'end when the TimeSeries actually ends.')
            gpsend = self.span[1]
        times = self.times.data
        if gpsstart <= self.span[0]:
            idx0 = 0
        else:
            idx0 = bisect.bisect_right(times, gpsstart)
        if gpsend >= self.span[1]:
            idx1 = None
        else:
            idx1 = bisect.bisect_left(times, gpsend)
        return self[idx0:idx1]

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
            operand = self.mean(axis=0).data
            unit = units.dimensionless_unscaled
        elif operand == 'median':
            operand = self.median(axis=0).data
            unit = units.dimensionless_unscaled
        elif isinstance(operand, Spectrum):
            unit = self.unit / operand.unit
            operand = operand.data
        elif isinstance(operand, numbers.Number):
            unit = units.dimensionless_unscaled
        else:
            raise ValueError("operand '%s' unrecognised, please give Spectrum "
                             "or one of: 'mean', 'median'")
        return self.__class__(self.data / operand, unit=unit, epoch=self.epoch,
                              f0=self.f0, name=self.name, dt=self.dt,
                              df=self.df, logf=self.logf)

    def plot(self, **kwargs):
        """Plot the data for this `Spectrogram`
        """
        from ..plotter import SpectrogramPlot
        return SpectrogramPlot(self, **kwargs)

    def to_logf(self, fmin=None, fmax=None, num=None):
        """Convert this `Spectrogram`` into logarithmic scale.
        """
        num = num or self.shape[-1]
        fmin = (fmin or float(self.f0.value) or
                float(self.f0.value + self.df.value))
        fmax = fmax or float(self.f0.value + self.shape[-1] * self.df.value)
        linf = self.frequencies.data
        logf = numpy.logspace(log10(fmin), log10(fmax), num=num)
        logf = logf[logf < linf.max()]
        new = self.__class__(numpy.zeros((self.shape[0], logf.size)),
                             epoch=self.epoch, dt=self.dt, unit=self.unit)
        for i in range(self.shape[0]):
            interpolator = interpolate.interp1d(linf[-logf.size:],
                                                self.data[i, -logf.size:],
                                                axis=0)
            new.data[i, :] = interpolator(logf)
        new.metadata = self.metadata.copy()
        new.f0 = logf[0]
        new.df = logf[1] - logf[0]
        new.logf = True
        return new

    @classmethod
    def from_spectra(cls, *spectra, **kwargs):
        """Build a new `Spectrogram` from a list of spectra.

        Parameters
        ----------
        *spectra
            any number of :class:`~gwpy.spectrum.core.Spectrum` series
        dt : `float`, :class:`~astropy.units.quantity.Quantity`, optional
            stride between given spectra 

        Returns
        -------
        Spectrogram
            a new `Spectrogram` from a vertical stacking of the spectra
            The new object takes the metadata from the first given
            :class:`~gwpy.spectrum.core.Spectrum` if not given explicitly

        Notes
        -----
        Each :class:`~gwpy.spectrum.core.Spectrum` passed to this
        constructor must be the same length.
        """
        data = numpy.vstack([s.data for s in spectra])
        s1 = spectra[0]
        if not all(s.f0 == s1.f0 for s in spectra):
            raise ValueError("Cannot stack spectra with different f0")
        if not all(s.df == s1.df for s in spectra):
            raise ValueError("Cannot stack spectra with different df")
        if (not all(s.logf for s in spectra) and
                any(s.logf for s in spectra)):
            raise ValueError("Cannot stack spectra with different log-scaling")
        kwargs.setdefault('name', s1.name)
        kwargs.setdefault('epoch', s1.epoch)
        kwargs.setdefault('f0', s1.f0)
        kwargs.setdefault('df', s1.df)
        kwargs.setdefault('unit', s1.unit)
        if not ('dt' in kwargs or 'times' in kwargs):
            try:
                kwargs.setdefault('dt', spectra[1].epoch.gps - s1.epoch.gps)
            except AttributeError:
                raise ValueError("Cannot determine dt (time-spacing) for "
                                 "Spectrogram from inputs")
        return Spectrogram(data, logf=s1.logf, **kwargs)

    def percentile(self, percentile):
        """Calculate a given spectral percentile for this `SpectralVariance`

        Parameters
        ----------
        percentile : `float`
            percentile (0 - 100) of the bins to compute

        Returns
        -------
        spectrum : :class:`~gwpy.spectrum.core.Spectrum`
            the given percentile `Spectrum` calculated from this
            `SpectralVaraicence`
        """
        out = scipy.percentile(self.data, percentile, axis=0)
        name = '%s %s%% percentile' % (self.name, percentile)
        return Spectrum(out, epoch=self.epoch, channel=self.channel,
                        name=name, logf=self.logx, f0=self.f0, df=self.df,
                        frequencies=(hasattr(self, '_frequencies') and
                                     self.frequencies or None))

    def filter(self, *filt, **kwargs):
        """Apply the given `Filter` to this `Spectrum`

        Parameters
        ----------
        *filt
            one of:

            - a single :class:`scipy.signal.lti` filter
            - (numerator, denominator) polynomials
            - (zeros, poles, gain)
            - (A, B, C, D) 'state-space' representation
        inplace : `bool`, optional, default: `False`
            apply the filter directly on these data, without making a
            copy, default: `False`

        Returns
        -------
        fspectrum : `Spectrum`
            the filtered version of the input `Spectrum`

        See also
        --------
        :mod:`scipy.signal`
            for details on filtering and representations
        """
        # parse filter
        if len(filt) == 1 and isinstance(filt[0], signal.lti):
            filt = filt[0]
            a = filt.den
            b = filt.num
        elif len(filt) == 2:
            b, a = filt
        elif len(filt) == 3:
            b, a = signal.zpk2tf(*filt)
        elif len(filt) == 4:
            b, a = signal.ss2tf(*filt)
        else:
            raise ValueError("Cannot interpret filter arguments. Please give "
                             "either a signal.lti object, or a tuple in zpk "
                             "or ba format. See scipy.signal docs for "
                             "details.")
        # parse keyword args
        inplace = kwargs.pop('inplace', False)
        if kwargs:
            raise TypeError("Spectrogram.filter() got an unexpected keyword "
                            "argument '%s'" % list(kwargs.keys())[0])
        f = self.frequencies.data.copy()
        if f[0] == 0:
            f[0] = 1e-100
        fresp = numpy.nan_to_num(abs(signal.freqs(b, a, f)[1]))
        if inplace:
            self *= fresp
            return self
        else:
            new = self * fresp
            return new

    # -------------------------------------------
    # connectors

    def is_compatible(self, other):
        """Check whether metadata attributes for self and other match.
        """
        if not self.dt == other.dt:
            raise ValueError("Spectrogram time resolutions do not match.")
        if not self.df == other.df:
            raise ValueError("Spectrogram frequency resolutios do not match.")
        if not self.f0 == other.f0:
            raise ValueError("Spectrogram starting frequencies do not match.")
        if not self.unit == other.unit:
            raise ValueError("Spectrogram units do not match")
        return True

    def is_contiguous(self, other):
        """Check whether other is contiguous with self.
        """
        self.is_compatible(other)
        if self.span[1] == other.span[0]:
            return 1
        elif other.span[1] == self.span[0]:
            return -1
        else:
            return 0

    def append(self, other, pad=0.0, gap='raise', inplace=True):
        """Connect another `Spectrogram` onto the end of the current one

        Parameters
        ----------
        other : `Spectrogram`
            the second data set to connect to this one
        pad : `float`, optional, default: ``0.0``
            value with which to pad discontiguous `Spectrogram`
        gap : `str`, optional, default: ``'raise'``
            action to perform if there's a gap between the other series
            and this one. One of

                - ``'raise'`` - raise an `Exception`
                - ``'ignore'`` - remove gap and join data
                - ``'pad'`` - pad gap with zeros

        inplace : `bool`, optional, default: `True`
            perform operation in-place, modifying current `Spectrogram,
            otherwise copy data and return new `Spectrogram`

        Returns
        -------
        series : `Spectrogram`
            spectrogram containing joined data sets
        """
        # check metadata
        self.is_compatible(other)
        # make copy if needed
        if inplace:
            new = self
        else:
            new = self.copy()
        # fill gap
        if new.is_contiguous(other) != 1:
            if gap == 'pad':
                ngap = (other.span[0] - new.span[1]) / new.dt.value
                if ngap < 1:
                    raise ValueError("Cannot append Spectrogram that starts "
                                     "before this one.")
                gapshape = list(new.shape)
                gapshape[0] = ngap
                padding = numpy.ones(gapshape).view(new.__class__) * pad
                padding.epoch = new.span[1]
                padding.sample_rate = new.sample_rate
                new.append(padding, inplace=True)
            elif gap == 'ignore':
                pass
            else:
                raise ValueError("Cannot append discontiguous Spectrogram")
        # resize first
        s = list(new.shape)
        s[0] = new.shape[0] + other.shape[0]
        new.resize(s, refcheck=False)
        new[-other.shape[0]:] = other.data
        return new

    def prepend(self, other, pad=0.0, gap='raise', inplace=True):
        """Connect another `Spectrogram` onto the end of the current one

        Parameters
        ----------
        other : `Spectrogram`
            the second data set to connect to this one
        pad : `float`, optional, default: ``0.0``
            value with which to pad discontiguous `Spectrogram`
        gap : `str`, optional, default: ``'raise'``
            action to perform if there's a gap between the other series
            and this one. One of

                - ``'raise'`` - raise an `Exception`
                - ``'ignore'`` - remove gap and join data
                - ``'pad'`` - pad gap with zeros

        inplace : `bool`, optional, default: `True`
            perform operation in-place, modifying current `Spectrogram,
            otherwise copy data and return new `Spectrogram`

        Returns
        -------
        series : `Spectrogram`
            spectrogram containing joined data sets
        """
        # check metadata
        self.is_compatible(other)
        # make copy if needed
        if inplace:
            new = self
        else:
            new = self.copy()
        # fill gap
        if new.is_contiguous(other) != -1:
            if gap == 'pad':
                ngap = int((new.span[0]-other.span[1]) / new.dt.value)
                if ngap < 1:
                    raise ValueError("Cannot prepend Spectrogram that starts "
                                     "after this one.")
                gapshape = list(new.shape)
                gapshape[0] = ngap
                padding = numpy.ones(gapshape).view(new.__class__) * pad
                padding.epoch = other.span[1]
                padding.dt = new.dt
                new.prepend(padding, inplace=True)
            elif gap == 'ignore':
                pass
            else:
                raise ValueError("Cannot append discontiguous Spectrogram")
        # resize first
        N = new.shape[0]
        s = list(new.shape)
        s[0] = new.shape[0] + other.shape[0]
        new.resize(s, refcheck=False)
        new[-N:] = new.data[:N]
        new[:other.shape[0]] = other.data
        return new

    # -------------------------------------------
    # numpy.ndarray method modifiers
    # all of these try to return Spectra rather than simple numbers

    min = as_spectrum(Array2D.min)

    max = as_spectrum(Array2D.max)

    mean = as_spectrum(Array2D.mean)

    median = as_spectrum(Array2D.median)


class SpectrogramList(TimeSeriesList):
    """Fancy list representing a list of `Spectrogram`

    The `SpectrogramList` provides an easy way to collect and organise
    `Spectrogram` for a single `Channel` over multiple segments.

    Parameters
    ----------
    *items
        any number of `Spectrogram` series

    Returns
    -------
    list
        a new `SpectrogramList`

    Raises
    ------
    TypeError
        if any elements are not of type `Spectrogram`
    """
    EntryClass = Spectrogram
