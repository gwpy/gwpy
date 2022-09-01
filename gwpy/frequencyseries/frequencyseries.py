# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Representation of a frequency series
"""

import warnings

import numpy
from numpy import fft as npfft

from astropy import units
from astropy.io import registry as io_registry

from ..types import Series

__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['FrequencySeries']


class FrequencySeries(Series):
    """A data array holding some metadata to represent a frequency series

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

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

       ~FrequencySeries.read
       ~FrequencySeries.write
       ~FrequencySeries.plot
       ~FrequencySeries.zpk
    """
    _default_xunit = units.Unit('Hz')
    _print_slots = ['f0', 'df', 'epoch', 'name', 'channel']

    def __new__(cls, data, unit=None, f0=None, df=None, frequencies=None,
                name=None, epoch=None, channel=None, **kwargs):
        """Generate a new FrequencySeries.
        """
        if f0 is not None:
            kwargs['x0'] = f0
        if df is not None:
            kwargs['dx'] = df
        if frequencies is not None:
            kwargs['xindex'] = frequencies

        # generate FrequencySeries
        return super().__new__(
            cls, data, unit=unit, name=name, channel=channel,
            epoch=epoch, **kwargs)

    # -- FrequencySeries properties -------------

    f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
                  """Starting frequency for this `FrequencySeries`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
                  """Frequency spacing of this `FrequencySeries`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    frequencies = property(fget=Series.xindex.__get__,
                           fset=Series.xindex.__set__,
                           fdel=Series.xindex.__delete__,
                           doc="""Series of frequencies for each sample""")

    # -- FrequencySeries i/o --------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `FrequencySeries`

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

        Raises
        ------
        IndexError
            if ``source`` is an empty list

        Notes
        -----"""
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `FrequencySeries` to a file

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

    # -- FrequencySeries methods ----------------

    def plot(self, xscale='log', **kwargs):
        # use log y-scale for ASD, PSD
        u = self.unit
        try:
            hzpow = u.powers[u.bases.index(units.Hz)]
        except ValueError:
            pass
        else:
            if hzpow < 0:
                kwargs.setdefault('yscale', 'log')

        kwargs.update(xscale=xscale)
        return super().plot(**kwargs)

    def ifft(self):
        """Compute the one-dimensional discrete inverse Fourier
        transform of this `FrequencySeries`.

        Returns
        -------
        out : :class:`~gwpy.timeseries.TimeSeries`
            the normalised, real-valued `TimeSeries`.

        See also
        --------
        numpy.fft.irfft : The inverse (real) FFT function

        Notes
        -----
        This method applies the necessary normalisation such that the
        condition holds:

        >>> timeseries = TimeSeries([1.0, 0.0, -1.0, 0.0], sample_rate=1.0)
        >>> timeseries.fft().ifft() == timeseries
        """
        from ..timeseries import TimeSeries
        nout = (self.size - 1) * 2
        # Undo normalization from TimeSeries.fft
        # The DC component does not have the factor of two applied
        # so we account for it here
        dift = npfft.irfft(self.value * nout) / 2
        new = TimeSeries(dift, epoch=self.epoch, channel=self.channel,
                         unit=self.unit, dx=1/self.dx/nout)
        return new

    def zpk(self, zeros, poles, gain, analog=True):
        """Filter this `FrequencySeries` by applying a zero-pole-gain filter

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
        spectrum : `FrequencySeries`
            the frequency-domain filtered version of the input data

        See also
        --------
        FrequencySeries.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(zeros, poles, gain, analog=analog)

    def interpolate(self, df):
        """Interpolate this `FrequencySeries` to a new resolution.

        Parameters
        ----------
        df : `float`
            desired frequency resolution of the interpolated `FrequencySeries`,
            in Hz

        Returns
        -------
        out : `FrequencySeries`
            the interpolated version of the input `FrequencySeries`

        See also
        --------
        numpy.interp
            for the underlying 1-D linear interpolation scheme
        """
        f0 = self.f0.decompose().value
        N = (self.size - 1) * (self.df.decompose().value / df) + 1
        fsamples = numpy.arange(0, numpy.rint(N),
                                dtype=self.real.dtype) * df + f0
        out = type(self)(numpy.interp(fsamples, self.frequencies.value,
                                      self.value))
        out.__array_finalize__(self)
        out.f0 = f0
        out.df = df
        return out

    def filter(self, *filt, **kwargs):
        """Apply a filter to this `FrequencySeries`.

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
        result : `FrequencySeries`
            the filtered version of the input `FrequencySeries`,
            if ``inplace=True`` was given, this is just a reference to
            the modified input array

        Raises
        ------
        ValueError
            if ``filt`` arguments cannot be interpreted properly
        """
        from ._fdcommon import fdfilter
        return fdfilter(self, *filt, **kwargs)

    def filterba(self, *args, **kwargs):
        warnings.warn("filterba will be removed soon, please use "
                      "FrequencySeries.filter instead, with the same "
                      "arguments", DeprecationWarning)
        return self.filter(*args, **kwargs)

    @classmethod
    def from_lal(cls, lalfs, copy=True):
        """Generate a new `FrequencySeries` from a LAL `FrequencySeries`
        of any type.
        """
        # convert units
        from ..utils.lal import from_lal_unit
        try:
            unit = from_lal_unit(lalfs.sampleUnits)
        except TypeError:
            unit = None

        # create a new series
        return cls(
            lalfs.data.data,
            name=lalfs.name or None,
            unit=unit,
            f0=lalfs.f0,
            df=lalfs.deltaF,
            epoch=lalfs.epoch,
            channel=None,
            copy=copy,
        )

    def to_lal(self):
        """Convert this `FrequencySeries` into a LAL FrequencySeries.

        Returns
        -------
        lalspec : `FrequencySeries`
            an XLAL-format FrequencySeries of a given type, e.g.
            `REAL8FrequencySeries`

        Notes
        -----
        Currently, this function is unable to handle unit string
        conversion.
        """
        import lal
        from ..utils.lal import (find_typed_function, to_lal_unit)

        # map unit
        try:
            unit, scale = to_lal_unit(self.unit)
        except ValueError as exc:
            warnings.warn(f"{exc}, defaulting to lal.DimensionlessUnit")
            unit = lal.DimensionlessUnit
            scale = 1

        # convert epoch
        epoch = lal.LIGOTimeGPS(0 if self.epoch is None else self.epoch.gps)

        # create FrequencySeries
        create = find_typed_function(self.dtype, 'Create', 'FrequencySeries')
        lalfs = create(self.name, epoch, self.f0.value, self.df.value,
                       unit, self.shape[0])
        lalfs.data.data = self.value * scale

        return lalfs

    @classmethod
    def from_pycbc(cls, fs, copy=True):
        """Convert a `pycbc.types.frequencyseries.FrequencySeries` into
        a `FrequencySeries`

        Parameters
        ----------
        fs : `pycbc.types.frequencyseries.FrequencySeries`
            the input PyCBC `~pycbc.types.frequencyseries.FrequencySeries`
            array

        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        spectrum : `FrequencySeries`
            a GWpy version of the input frequency series
        """
        return cls(fs.data, f0=0, df=fs.delta_f, epoch=fs.epoch, copy=copy)

    def to_pycbc(self, copy=True):
        """Convert this `FrequencySeries` into a
        `~pycbc.types.frequencyseries.FrequencySeries`

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        frequencyseries : `pycbc.types.frequencyseries.FrequencySeries`
            a PyCBC representation of this `FrequencySeries`
        """
        from pycbc import types

        if self.epoch is None:
            epoch = None
        else:
            epoch = self.epoch.gps
        if self.f0.to('Hz').value:
            raise ValueError(
                f"Cannot convert FrequencySeries to PyCBC with f0 = {self.f0}."
                " Starting frequency must be equal to 0 Hz."
            )
        return types.FrequencySeries(self.value,
                                     delta_f=self.df.to('Hz').value,
                                     epoch=epoch, copy=copy)
