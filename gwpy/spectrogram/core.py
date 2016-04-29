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

import warnings

from six import string_types

import numpy

import scipy
from scipy import signal
from astropy import units

from ..detector import Channel
from ..data import (Array2D, Series)
from ..segments import Segment
from ..timeseries import (TimeSeries, TimeSeriesList)
from ..frequencyseries import FrequencySeries
from ..utils.docstring import interpolate_docstring

__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['Spectrogram', 'SpectrogramList']


@interpolate_docstring
class Spectrogram(Array2D):
    """A 2D array holding a spectrogram of time-frequency data

    Parameters
    ----------
    %(Array1)s

    %(time-axis)s

    %(frequency-axis)s

    %(Array2)s

    Notes
    -----
    Key methods:

    .. autosummary::

       ~Spectrogram.read
       ~Spectrogram.write
       ~Spectrogram.plot
       ~Spectrogram.zpk
    """
    _metadata_slots = ['name', 'channel', 'epoch', 'dt', 'f0', 'df']
    _default_xunit = TimeSeries._default_xunit
    _default_yunit = FrequencySeries._default_xunit

    def __new__(cls, data, unit=None, name=None, channel=None, epoch=None,
                dt=None, times=None, f0=None, df=None, frequencies=None,
                **kwargs):
        """Generate a new Spectrogram.
        """
        # parse Channel input
        if isinstance(channel, Channel):
            name = name or channel.name
            unit = unit or channel.unit
        # get axis-based params
        if epoch is None:
            epoch = kwargs.pop('x0', 0)
        if dt is None:
            dt = kwargs.pop('dx', 1)
        if f0 is None:
            f0 = kwargs.pop('y0', 0)
        if df is None:
            df = kwargs.pop('dy', 1)
        if times is None:
            times = kwargs.pop('xindex', None)
        if frequencies is None:
            frequencies = kwargs.pop('frequencies', None)
        # generate Spectrogram
        new = super(Spectrogram, cls).__new__(cls, data, unit=unit, name=name,
                                              channel=channel, y0=f0, dx=dt,
                                              dy=df, xindex=times,
                                              yindex=frequencies, **kwargs)
        if epoch is not None:
            new.epoch = epoch
        return new

    # -------------------------------------------
    # Spectrogram properties

    epoch = property(TimeSeries.epoch.__get__, TimeSeries.epoch.__set__,
                     TimeSeries.epoch.__delete__,
                     """Starting GPS epoch for this `Spectrogram`

                     :type:`~gwpy.segments.Segment`
                     """)

    span = property(TimeSeries.span.__get__, TimeSeries.span.__set__,
                    TimeSeries.span.__delete__,
                    """GPS [start, stop) span for this `Spectrogram`

                    :type:`~gwpy.segments.Segment`
                    """)

    dt = property(TimeSeries.dt.__get__, TimeSeries.dt.__set__,
                  TimeSeries.dt.__delete__,
                  """Time-spacing for this `Spectrogram`

                  :type:`~astropy.units.Quantity` in seconds
                  """)

    f0 = property(Array2D.y0.__get__, Array2D.y0.__set__,
                  Array2D.y0.__delete__,
                  """Starting frequency for this `Spectrogram`

                  :type: `~astropy.units.Quantity` in Hertz
                  """)

    df = property(Array2D.dy.__get__, Array2D.dy.__set__,
                  Array2D.dy.__delete__,
                  """Frequency spacing of this `Spectrogram`

                  :type: `~astropy.units.Quantity` in Hertz
                  """)

    times = property(fget=Array2D.xindex.__get__,
                     fset=Array2D.xindex.__set__,
                     fdel=Array2D.xindex.__delete__,
                     doc="""Series of GPS times for each sample""")

    frequencies = property(fget=Array2D.yindex.__get__,
                           fset=Array2D.yindex.__set__,
                           fdel=Array2D.yindex.__delete__,
                           doc="Series of frequencies for this Spectrogram")

    band = property(fget=Array2D.yspan.__get__,
                    fset=Array2D.yspan.__set__,
                    fdel=Array2D.yspan.__delete__,
                    doc="""Frequency band described by this `Spectrogram`""")

    # -------------------------------------------
    # Spectrogram methods

    def ratio(self, operand):
        """Calculate the ratio of this `Spectrogram` against a reference

        Parameters
        ----------
        operand : `str`, `~gwpy.frequencyseries.FrequencySeries`,
                  `~astropy.units.Quantity`
            `FrequencySeries` or `Quantity` to weight against, or one of

            - ``'mean'`` : weight against the mean of each spectrum
              in this Spectrogram
            - ``'median'`` : weight against the median of each spectrum
              in this Spectrogram

        Returns
        -------
        spectrogram : `Spectrogram`
            a new `Spectrogram`
        """
        if isinstance(operand, string_types):
            if operand == 'mean':
                operand = self.mean(axis=0)
            elif operand == 'median':
                operand = self.median(axis=0)
            else:
                raise ValueError("operand %r unrecognised, please give a "
                                 "Quantity or one of: 'mean', 'median'"
                                 % operand)
        out = self / operand
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
            any number of `~gwpy.frequencyseries.FrequencySeries` series
        dt : `float`, `~astropy.units.Quantity`, optional
            stride between given spectra

        Returns
        -------
        Spectrogram
            a new `Spectrogram` from a vertical stacking of the spectra
            The new object takes the metadata from the first given
            `~gwpy.frequencyseries.FrequencySeries` if not given explicitly

        Notes
        -----
        Each `~gwpy.frequencyseries.FrequencySeries` passed to this
        constructor must be the same length.
        """
        data = numpy.vstack([s.value for s in spectra])
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
            except (AttributeError, IndexError):
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
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            the given percentile `FrequencySeries` calculated from this
            `SpectralVaraicence`
        """
        out = scipy.percentile(self.value, percentile, axis=0)
        name = '%s %s%% percentile' % (self.name, percentile)
        return FrequencySeries(out, epoch=self.epoch, channel=self.channel,
                               name=name, f0=self.f0, df=self.df,
                               frequencies=(hasattr(self, '_frequencies') and
                                            self.frequencies or None))

    def zpk(self, zeros, poles, gain):
        """Filter this `Spectrogram` by applying a zero-pole-gain filter

        Parameters
        ----------
        zeros : `array-like`
            list of zero frequencies (in Hertz)
        poles : `array-like`
            list of pole frequencies (in Hertz)
        gain : `float`
            DC gain of filter

        Returns
        -------
        specgram : `Spectrogram`
            the frequency-domain filtered version of the input data

        See Also
        --------
        Spectrogram.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(zeros, poles, gain)

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `Spectrogram`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `Spectrogram`.

        .. note::

           Unlike the related
           :meth:`TimeSeries.filter <gwpy.timeseries.TimeSeries.filter>`
           method, here all frequency information (e.g. frequencies of
           poles or zeros in a ZPK) is assumed to be in Hertz.

        Parameters
        ----------
        *filt
            one of:

            - `scipy.signal.lti`
            - ``(numerator, denominator)`` polynomials
            - ``(zeros, poles, gain)``
            - ``(A, B, C, D)`` 'state-space' representation

        Returns
        -------
        result : `Spectrogram`
            the filtered version of the input `Spectrogram`

        See also
        --------
        FrequencySeries.zpk
            for information on filtering in zero-pole-gain format
        scipy.signal.zpk2tf
            for details on converting ``(zeros, poles, gain)`` into
            transfer function format
        scipy.signal.ss2tf
            for details on converting ``(A, B, C, D)`` to transfer function
            format
        scipy.signal.freqs
            for details on the filtering calculation

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
        f = self.frequencies.value.copy()
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
        bins : `~numpy.ndarray`, optional, default `None`
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
        from ..frequencyseries import SpectralVariance
        return SpectralVariance.from_spectrogram(
            self, bins=bins, low=low, high=high, nbins=nbins, log=log,
            norm=norm, density=density)

    # -------------------------------------------
    # connectors

    def is_compatible(self, other):
        """Check whether metadata attributes for self and other match.
        """
        if type(other) in [list, tuple, numpy.ndarray]:
            return True
        if not numpy.isclose(
                self.dt.decompose().value, other.dt.decompose().value):
            raise ValueError("Spectrogram time resolutions do not match: "
                             "%s vs %s." % (self.dt, other.dt))
        if not numpy.isclose(
                self.df.decompose().value, other.df.decompose().value):
            raise ValueError("Spectrogram frequency resolutions do not match:"
                             "%s vs %s." % (self.df, other.df))
        if not numpy.isclose(
                self.f0.decompose().value, other.f0.decompose().value):
            raise ValueError("Spectrogram starting frequencies do not match:"
                             "%s vs %s." % (self.f0, other.f0))
        if not self.unit == other.unit:
            raise ValueError("Spectrogram units do not match: %s vs %s."
                             % (self.unit, other.unit))
        return True

    def crop_frequencies(self, low=None, high=None, copy=False):
        """Crop this `Spectrogram` to the specified frequencies

        Parameters
        ----------
        low : `float`
            lower frequency bound for cropped `Spectrogram`
        high : `float`
            upper frequency bound for cropped `Spectrogram`
        copy : `bool`
            if `False` return a view of the original data, otherwise create
            a fresh memory copy

        Returns
        -------
        spec : `Spectrogram`
            A new `Spectrogram` with a subset of data from the frequency
            axis
        """
        if low is not None:
            low = units.Quantity(low, self._default_yunit)
        if high is not None:
            high = units.Quantity(high, self._default_yunit)
        # check low frequency
        if low is not None and low == self.f0:
            low = None
        elif low is not None and low < self.f0:
            warnings.warn('Spectrogram.crop_frequencies given low frequency '
                          'cutoff below f0 of the input Spectrogram. Low '
                          'frequency crop will have no effect.')
        # check high frequency
        if high is not None and high.value == self.band[1]:
            high = None
        elif high is not None and high.value > self.band[1]:
            warnings.warn('Spectrogram.crop_frequencies given high frequency '
                          'cutoff above cutoff of the input Spectrogram. High '
                          'frequency crop will have no effect.')
        # find low index
        if low is None:
            idx0 = None
        else:
            idx0 = int((low.value - self.f0.value) // self.df.value)
        # find high index
        if high is None:
            idx1 = None
        else:
            idx1 = int((high.value - self.f0.value) // self.df.value)
        # crop
        if copy:
            return self[:, idx0:idx1].copy()
        else:
            return self[:, idx0:idx1]

    # -------------------------------------------
    # ufunc modifier

    def _wrap_function(self, function, *args, **kwargs):
        out = super(Spectrogram, self)._wrap_function(
            function, *args, **kwargs)
        # requested frequency axis, return a FrequencySeries
        if out.ndim == 1 and out.x0.unit == self.y0.unit:
            return FrequencySeries(out.value, name=out.name, unit=out.unit,
                                   epoch=out.epoch, channel=out.channel,
                                   f0=out.x0.value, df=out.dx.value)
        # requested time axis, return a TimeSeries
        elif out.ndim == 1:
            return TimeSeries(out.value, name=out.name, unit=out.unit,
                              epoch=out.epoch, channel=out.channel, dx=out.dx)
        # otherwise return whatever we got back from super (Quantity)
        return out
    _wrap_function.__doc__ = Array2D._wrap_function.__doc__


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
