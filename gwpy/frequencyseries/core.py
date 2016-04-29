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

"""Representation of a frequency series
"""

import warnings
from copy import deepcopy

from numpy import fft as npfft
from scipy import signal

from astropy import units

from ..data import (Array, Series)
from ..detector import Channel
from ..utils import with_import
from ..utils.docstring import interpolate_docstring


__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['FrequencySeries', 'Spectrum']

interpolate_docstring.update({
    'frequency-axis': (
        """f0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        starting frequency for these data
    df : `float`, `~astropy.units.Quantity`, optional, default: `1`
        frequency resolution for these data
    frequencies : `array-like`
        the complete array of frequencies indexing the data.
        This argument takes precedence over `f0` and `df` so should
        be given in place of these if relevant, not alongside"""),
})


@interpolate_docstring
class FrequencySeries(Series):
    """A data array holding some metadata to represent a frequency series

    Parameters
    ----------
    %(Array1)s

    %(frequency-axis)s

    %(Array2)s

    Notes
    -----
    Key methods:

    .. autosummary::

       ~FrequencySeries.read
       ~FrequencySeries.write
       ~FrequencySeries.plot
       ~FrequencySeries.zpk
    """
    _metadata_slots = Array._metadata_slots + ['f0', 'df']
    _default_xunit = units.Unit('Hz')

    def __new__(cls, data, unit=None, frequencies=None, name=None,
                epoch=None, f0=0, df=1, channel=None,
                **kwargs):
        """Generate a new FrequencySeries.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
        if frequencies is None and 'xindex' in kwargs:
            frequencies = kwargs.pop('xindex')
        # allow use of x0 and dx
        f0 = kwargs.pop('x0', f0)
        df = kwargs.pop('dx', df)
        # generate Spectrum
        return super(FrequencySeries, cls).__new__(
            cls, data, name=name, unit=unit, channel=channel, x0=f0, dx=df,
            epoch=epoch, xindex=frequencies, **kwargs)

    # -------------------------------------------
    # FrequencySeries properties

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

    # -------------------------------------------
    # FrequencySeries methods

    def plot(self, **kwargs):
        """Display this `FrequencySeries` in a figure

        All arguments are passed onto the
        `~gwpy.plotter.FrequencySeriesPlot` constructor

        Returns
        -------
        plot : `~gwpy.plotter.FrequencySeriesPlot`
            a new `FrequencySeriesPlot` rendering of this `FrequencySeries`
        """
        from ..plotter import FrequencySeriesPlot
        return FrequencySeriesPlot(self, **kwargs)

    def ifft(self):
        """Compute the one-dimensional discrete inverse Fourier
        transform of this `Spectrum`.

        Returns
        -------
        out : :class:`~gwpy.timeseries.TimeSeries`
            the normalised, real-valued `TimeSeries`.

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.

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
        dift = npfft.irfft(self.value * nout)
        dift[1:] /= 2
        new = TimeSeries(dift, epoch=self.epoch, channel=self.channel,
                       unit=self.unit * units.Hertz, dx=1/self.dx/nout)
        return new

    def zpk(self, zeros, poles, gain):
        """Filter this `FrequencySeries` by applying a zero-pole-gain filter

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
        spectrum : `FrequencySeries`
            the frequency-domain filtered version of the input data

        See Also
        --------
        FrequencySeries.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(zeros, poles, gain)

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `FrequencySeries`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `FrequencySeries`.

        .. note::

           Unlike the related
           :meth:`TimeSeries.filter <gwpy.timeseries.TimeSeries.filter>`
           method, here all frequency information (e.g. frequencies of
           poles or zeros in a ZPK) is assumed to be in Hertz.

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
        result : `FrequencySeries`
            the filtered version of the input `FrequencySeries`

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
        # parse keyword args
        inplace = kwargs.pop('inplace', False)
        if kwargs:
            raise TypeError("FrequencySeries.filter() got an unexpected keyword "
                            "argument '%s'" % list(kwargs.keys())[0])
        fresp = abs(signal.freqs(b, a, self.frequencies.value)[1])
        if inplace:
            self.value *= fresp
            return self
        else:
            new = (self.value * fresp).view(type(self))
            new.__dict__ = deepcopy(self.__dict__)
            return new

    def filterba(self, *args, **kwargs):
        warnings.warn("filterba will be removed soon, please use "
                      "FrequencySeries.filter instead, with the same arguments",
                      DeprecationWarning)
        return self.filter(*args, **kwargs)

    @classmethod
    @with_import('lal')
    def from_lal(cls, lalfs, copy=True):
        """Generate a new `FrequencySeries` from a LAL `FrequencySeries` of any type
        """
        from ..utils.lal import from_lal_unit
        try:
            unit = from_lal_unit(lalfs.sampleUnits)
        except TypeError:
            unit = None
        channel = Channel(lalfs.name, unit=unit,
                          dtype=lalfs.data.data.dtype)
        return cls(lalfs.data.data, channel=channel, f0=lalfs.f0,
                   df=lalfs.deltaF, epoch=float(lalfs.epoch),
                   dtype=lalfs.data.data.dtype, copy=copy)

    @with_import('lal')
    def to_lal(self):
        """Convert this `FrequencySeries` into a LAL FrequencySeries.

        Returns
        -------
        lalspec : `FrequencySeries`
            an XLAL-format FrequencySeries of a given type, e.g.
            :lal:`REAL8FrequencySeries`

        Notes
        -----
        Currently, this function is unable to handle unit string
        conversion.
        """
        from ..utils.lal import (LAL_TYPE_STR_FROM_NUMPY, to_lal_unit)
        typestr = LAL_TYPE_STR_FROM_NUMPY[self.dtype.type]
        try:
            unit = to_lal_unit(self.unit)
        except TypeError:
            unit = lal.lalDimensionlessUnit
        create = getattr(lal, 'Create%sFrequencySeries' % typestr.upper())
        if self.epoch is None:
            epoch = 0
        else:
            epoch = self.epoch.gps
        lalfs = create(self.name, lal.LIGOTimeGPS(epoch),
                       self.f0.value, self.df.value, unit, self.size)
        lalfs.data.data = self.value
        return lalfs

    @classmethod
    def from_pycbc(cls, fs):
        """Convert a `pycbc.types.frequencyseries.FrequencySeries` into
        a `FrequencySeries`

        Parameters
        ----------
        fs : `pycbc.types.frequencyseries.FrequencySeries`
            the input PyCBC `~pycbc.types.frequencyseries.FrequencySeries`
            array

        Returns
        -------
        spectrum : `FrequencySeries`
            a GWpy version of the input frequency series
        """
        return cls(fs.data, f0=0, df=fs.delta_f, epoch=fs.epoch)

    @with_import('pycbc.types')
    def to_pycbc(self, copy=True):
        """Convert this `FrequencySeries` into a
        `pycbc.types.frequencyseries.FrequencySeries`

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        frequencyseries : `pycbc.types.frequencyseries.FrequencySeries`
            a PyCBC representation of this `FrequencySeries`
        """
        if self.epoch is None:
            epoch = None
        else:
            epoch = self.epoch.gps
        return types.FrequencySeries(self.data, delta_f=self.df.to('Hz').value,
                                     epoch=epoch, copy=copy)


class Spectrum(FrequencySeries):
    def __new__(cls, *args, **kwargs):
        warnings.warn("The gwpy.spectrum.Spectrum was renamed "
                      "gwpy.frequencyseries.FrequencySeries",
                      DeprecationWarning)
        return super(Spectrum, cls).__new__(cls, *args, **kwargs)
