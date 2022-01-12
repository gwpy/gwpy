# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2022)
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

import numpy

from astropy import units
from astropy.io import registry as io_registry

from ..types import (Array2D, Series)
from ..timeseries import (TimeSeries, TimeSeriesList)
from ..timeseries.core import _format_time
from ..frequencyseries import FrequencySeries
from ..frequencyseries._fdcommon import fdfilter

__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['Spectrogram', 'SpectrogramList']


def _ordinal(n):
    """Returns the ordinal string for a given integer

    See https://stackoverflow.com/a/20007730/1307974

    Parameters
    ----------
    n : `int`
        the number to convert to ordinal

    Examples
    --------
    >>> _ordinal(11)
    '11th'
    >>> _ordinal(102)
    '102nd'
    """
    idx = int((n//10 % 10 != 1) * (n % 10 < 4) * n % 10)
    return '{}{}'.format(n, "tsnrhtdd"[idx::4])


class Spectrogram(Array2D):
    """A 2D array holding a spectrogram of time-frequency data

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        the rate of samples per second (Hertz)

    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `epoch` and `sample_rate` so should
        be given in place of these if relevant, not alongside

    f0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        starting frequency for these data

    df : `float`, `~astropy.units.Quantity`, optional, default: `1`
        frequency resolution for these data

    frequencies : `array-like`
        the complete array of frequencies indexing the data.
        This argument takes precedence over `f0` and `df` so should
        be given in place of these if relevant, not alongside

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    name : `str`, optional
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`, optional
        source data stream for these data

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator

    Notes
    -----
    Key methods:

    .. autosummary::

       ~Spectrogram.read
       ~Spectrogram.write
       ~Spectrogram.plot
       ~Spectrogram.zpk
    """
    _metadata_slots = Series._metadata_slots + ('y0', 'dy', 'yindex')
    _default_xunit = TimeSeries._default_xunit
    _default_yunit = FrequencySeries._default_xunit
    _rowclass = TimeSeries
    _columnclass = FrequencySeries

    def __new__(cls, data, unit=None, t0=None, dt=None, f0=None, df=None,
                times=None, frequencies=None,
                name=None, channel=None, **kwargs):
        """Generate a new Spectrogram.
        """
        # parse t0 or epoch
        epoch = kwargs.pop('epoch', None)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if epoch is None and t0 is not None:
            kwargs['x0'] = _format_time(t0)
        elif epoch is not None:
            kwargs['x0'] = _format_time(epoch)
        # parse sample_rate or dt
        if dt is not None:
            kwargs['dx'] = dt
        # parse times
        if times is not None:
            kwargs['xindex'] = times

        # parse y-axis params
        if f0 is not None:
            kwargs['y0'] = f0
        if df is not None:
            kwargs['dy'] = df
        if frequencies is not None:
            kwargs['yindex'] = frequencies

        # generate Spectrogram
        return super().__new__(cls, data, unit=unit, name=name,
                               channel=channel, **kwargs)

    # -- Spectrogram properties -----------------

    epoch = property(TimeSeries.epoch.__get__, TimeSeries.epoch.__set__,
                     TimeSeries.epoch.__delete__,
                     """Starting GPS epoch for this `Spectrogram`

                     :type: `~gwpy.segments.Segment`
                     """)

    t0 = property(TimeSeries.t0.__get__, TimeSeries.t0.__set__,
                  TimeSeries.t0.__delete__,
                  """GPS time of first time bin

                  :type: `~astropy.units.Quantity` in seconds
                  """)

    dt = property(TimeSeries.dt.__get__, TimeSeries.dt.__set__,
                  TimeSeries.dt.__delete__,
                  """Time-spacing for this `Spectrogram`

                  :type: `~astropy.units.Quantity` in seconds
                  """)

    span = property(TimeSeries.span.__get__, TimeSeries.span.__set__,
                    TimeSeries.span.__delete__,
                    """GPS [start, stop) span for this `Spectrogram`

                    :type: `~gwpy.segments.Segment`
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

    # -- Spectrogram i/o ------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `Spectrogram`

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        *args
            Other arguments are (in general) specific to the given
            ``format``.

        format : `str`, optional
            Source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        **kwargs
            Other keywords are (in general) specific to the given ``format``.

        Returns
        -------
        specgram : `Spectrogram`

        Notes
        -----"""
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `Spectrogram` to a file

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        target : `str`
            output filename

        format : `str`, optional
            output format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        Notes
        -----"""
        return io_registry.write(self, target, *args, **kwargs)

    # -- Spectrogram methods --------------------

    def ratio(self, operand):
        """Calculate the ratio of this `Spectrogram` against a reference

        Parameters
        ----------
        operand : `str`, `FrequencySeries`, `Quantity`
            a `~gwpy.frequencyseries.FrequencySeries` or
            `~astropy.units.Quantity` to weight against, or one of

            - ``'mean'`` : weight against the mean of each spectrum
              in this Spectrogram
            - ``'median'`` : weight against the median of each spectrum
              in this Spectrogram

        Returns
        -------
        spectrogram : `Spectrogram`
            a new `Spectrogram`

        Raises
        ------
        ValueError
            if ``operand`` is given as a `str` that isn't supported
        """
        if isinstance(operand, str):
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

    def plot(
        self,
        method="pcolormesh",
        figsize=(12, 6),
        xscale="auto-gps",
        **kwargs,
    ):
        """Plot the data for this `Spectrogram`

        Parameters
        ----------
        method : `str`, optional
            which plotting method to use to render this spectrogram,
            either ``'pcolormesh'`` (default) or ``'imshow'``

        figsize : `tuple` of `float`, optional
            ``(width, height)`` (inches) of the output figure

        xscale : `str`, optional
            the X-axis scale

        **kwargs
            all keyword arguments are passed along to underlying
            functions, see below for references

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            the `Plot` containing the data

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.Axes.imshow
        gwpy.plot.Axes.pcolormesh
            for documentation of keyword arguments used in rendering the
            `Spectrogram` data
        """
        return super().plot(
            method=method,
            figsize=figsize,
            xscale=xscale,
            **kwargs,
        )

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
        spec1 = list(spectra)[0]
        if not all(s.f0 == spec1.f0 for s in spectra):
            raise ValueError("Cannot stack spectra with different f0")
        if not all(s.df == spec1.df for s in spectra):
            raise ValueError("Cannot stack spectra with different df")
        kwargs.setdefault('name', spec1.name)
        kwargs.setdefault('channel', spec1.channel)
        kwargs.setdefault('epoch', spec1.epoch)
        kwargs.setdefault('f0', spec1.f0)
        kwargs.setdefault('df', spec1.df)
        kwargs.setdefault('unit', spec1.unit)
        if not ('dt' in kwargs or 'times' in kwargs):
            try:
                kwargs.setdefault('dt', spectra[1].epoch.gps - spec1.epoch.gps)
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
        out = numpy.percentile(self.value, percentile, axis=0)
        if self.name is not None:
            name = '{}: {} percentile'.format(self.name, _ordinal(percentile))
        else:
            name = None
        return FrequencySeries(
            out, epoch=self.epoch, channel=self.channel, name=name,
            f0=self.f0, df=self.df, unit=self.unit, frequencies=(
                hasattr(self, '_frequencies') and self.frequencies or None))

    def zpk(self, zeros, poles, gain, analog=True):
        """Filter this `Spectrogram` by applying a zero-pole-gain filter

        Parameters
        ----------
        zeros : `array-like`
            list of zero frequencies (in Hertz)

        poles : `array-like`
            list of pole frequencies (in Hertz)

        gain : `float`
            DC gain of filter

        analog : `bool`, optional
            type of ZPK being applied, if `analog=True` all parameters
            will be converted in the Z-domain for digital filtering

        Returns
        -------
        specgram : `Spectrogram`
            the frequency-domain filtered version of the input data

        See also
        --------
        Spectrogram.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(zeros, poles, gain, analog=analog)

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `Spectrogram`.

        Parameters
        ----------
        *filt : filter arguments
            1, 2, 3, or 4 arguments defining the filter to be applied,

                - an ``Nx1`` `~numpy.ndarray` of FIR coefficients
                - an ``Nx6`` `~numpy.ndarray` of SOS coefficients
                - ``(numerator, denominator)`` polynomials
                - ``(zeros, poles, gain)``
                - ``(A, B, C, D)`` 'state-space' representation

        analog : `bool`, optional
            if `True`, filter definition will be converted from Hertz
            to Z-domain digital representation, default: `False`

        inplace : `bool`, optional
            if `True`, this array will be overwritten with the filtered
            version, default: `False`

        Returns
        -------
        result : `Spectrogram`
            the filtered version of the input `Spectrogram`,
            if ``inplace=True`` was given, this is just a reference to
            the modified input array

        Raises
        ------
        ValueError
            if ``filt`` arguments cannot be interpreted properly
        """
        return fdfilter(self, *filt, **kwargs)

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

        See also
        --------
        numpy.histogram
            for details on specifying bins and weights
        """
        from ..frequencyseries import SpectralVariance
        return SpectralVariance.from_spectrogram(
            self, bins=bins, low=low, high=high, nbins=nbins, log=log,
            norm=norm, density=density)

    # -- Spectrogram connectors -----------------

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
            idx0 = int(float(low.value - self.f0.value) // self.df.value)
        # find high index
        if high is None:
            idx1 = None
        else:
            idx1 = int(float(high.value - self.f0.value) // self.df.value)
        # crop
        if copy:
            return self[:, idx0:idx1].copy()
        return self[:, idx0:idx1]

    # -- Spectrogram ufuncs ---------------------

    def _wrap_function(self, function, *args, **kwargs):
        out = super()._wrap_function(function, *args, **kwargs)
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

    # -- other ----------------------------------

    def __getitem__(self, item):
        out = super().__getitem__(item)

        # set epoch manually, because Spectrogram doesn't store self._epoch
        if isinstance(out, self._columnclass):
            out.epoch = self.epoch

        return out


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
