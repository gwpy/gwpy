# -*- coding: utf-8 -*-
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

import numbers

import numpy
import scipy
from scipy import (interpolate, signal)
from astropy import units

from ..detector import Channel
from ..data import (Array2D, Series)
from ..timeseries import (TimeSeries, TimeSeriesList, common)
from ..spectrum import Spectrum
from ..utils import update_docstrings

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


@update_docstrings
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
    _metadata_slots = ['name', 'unit', 'epoch', 'dt', 'f0', 'df']
    xunit = TimeSeries.xunit
    yunit = Spectrum.xunit

    def __new__(cls, data, name=None, unit=None, channel=None, epoch=None,
                f0=None, dt=None, df=None, **kwargs):
        """Generate a new Spectrogram.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
            if channel.sample_rate:
                dt = dt or 1 / channel.sample_rate
        # generate Spectrogram
        return super(Spectrogram, cls).__new__(cls, data, name=name, unit=unit,
                                               channel=channel, epoch=epoch,
                                               f0=f0, dt=dt, df=df,
                                               **kwargs)

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

    # -------------------------------------------
    # Spectrogram methods

    def ratio(self, operand):
        """Calculate the ratio of this `Spectrogram` against a
        reference.

        Parameters
        ----------
        operand : `str`, `Spectrum`
            `Spectrum` against which to weight, or one of

            - ``'mean'`` : weight against the mean of each spectrum
              in this Spectrogram
            - ``'median'`` : weight against the median of each spectrum
              in this Spectrogram

        Returns
        -------
        spec : `~gwpy.data.Spectrogram`
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
        out = self / operand
        out.unit = unit
        return out

    def plot(self, **kwargs):
        """Plot the data for this `Spectrogram`
        """
        from ..plotter import SpectrogramPlot
        return SpectrogramPlot(self, **kwargs)

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
        return Spectrogram(data, **kwargs)

    def percentile(self, percentile):
        """Calculate a given spectral percentile for this `Spectrogram`.

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
                        name=name, f0=self.f0, df=self.df,
                        frequencies=(hasattr(self, '_frequencies') and
                                     self.frequencies or None))

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `Spectrogram`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `Spectrogram`.

        Parameters
        ----------
        *filt
            one of:

            - :class:`scipy.signal.lti`
            - ``(numerator, denominator)`` polynomials
            - ``(zeros, poles, gain)``
            - ``(A, B, C, D)`` 'state-space' representation

        Returns
        -------
        result : `Spectrogram`
            the filtered version of the input `Spectrogram`

        See also
        --------
        scipy.signal.zpk2tf
            for details on converting ``(zeros, poles, gain)`` into
            transfer function format
        scipy.signal.ss2tf
            for details on converting ``(A, B, C, D)`` to transfer function
            format
        scipy.signal.freqs
            for details on the filtering calculation

        Examples
        --------
        To apply a zpk filter with a pole at 0 Hz, a zero at 100 Hz and
        a gain of 25::

            >>> data2 = data.filter([100], [0], 25)

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly
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
        if isinstance(a, float):
            a = numpy.array([a])
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

    def variance(self, bins=None, low=None, high=None, nbins=500,
                 log=False, norm=False, density=False):
        """Calculate the `SpectralVariance` of this `Spectrogram`.

        Parameters
        ----------
        bins : :class:`~numpy.ndarray`, optional, default `None`
            array of histogram bin edges, including the rightmost edge
        low : `float`, optional, default: `None`
            left edge of lowest amplitude bin, only read
            if ``bins`` is not given
        high : `float`, optional, default: `None`
            right edge of highest amplitude bin, only read
            if ``bins`` is not given
        nbins : `int`, optional, default: `500`
            number of bins to generate, only read if ``bins`` is not
            given
        log : `bool`, optional, default: `False`
            calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given
        norm : `bool`, optional, default: `False`
            normalise bin counts to a unit sum
        density : `bool`, optional, default: `False`
            normalise bin counts to a unit integral

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts

        See Also
        --------
        :func:`numpy.histogram`
            for details on specifying bins and weights
        """
        from ..spectrum import SpectralVariance
        return SpectralVariance.from_spectrogram(
                   self, bins=bins, low=low, high=high, nbins=nbins, log=log,
                   norm=norm, density=density)

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
            raise ValueError("Spectrogram units do not match: %s vs %s."
                             % (self.unit, other.unit))
        return True

    is_contiguous = common.is_contiguous
    append = common.append
    prepend = common.prepend
    update = common.update
    crop = common.crop

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
