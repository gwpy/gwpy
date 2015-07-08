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

"""Representation of a frequency-series spectrum
"""

import warnings
from copy import deepcopy

from scipy import signal

from astropy import units

from ..data import (Array, Series)
from ..detector import Channel
from ..utils import with_import
from ..utils.docstring import interpolate_docstring


from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['Spectrum']

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
class Spectrum(Series):
    """A data array holding some metadata to represent a spectrum.

    Parameters
    ----------
    %(Array1)s

    %(frequency-axis)s

    %(Array2)s

    Notes
    -----
    Key methods:

    .. autosummary::

       ~Spectrum.read
       ~Spectrum.write
       ~Spectrum.plot
       ~Spectrum.zpk
    """
    _metadata_slots = Array._metadata_slots + ['f0', 'df']
    _default_xunit = units.Unit('Hz')

    def __new__(cls, data, unit=None, frequencies=None, name=None,
                epoch=None, f0=0, df=1, channel=None,
                **kwargs):
        """Generate a new Spectrum.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
        if frequencies is None and 'xindex' in kwargs:
            frequencies = kwargs.pop('xindex')
        # generate Spectrum
        return super(Spectrum, cls).__new__(cls, data, name=name, unit=unit,
                                            channel=channel, x0=f0, dx=df,
                                            xindex=frequencies, **kwargs)

    # -------------------------------------------
    # Spectrum properties

    f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
                  """Starting frequency for this `Spectrum`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
                  """Frequency spacing of this `Spectrum`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    frequencies = property(fget=Series.xindex.__get__,
                           fset=Series.xindex.__set__,
                           fdel=Series.xindex.__delete__,
                           doc="""Series of frequencies for each sample""")

    # -------------------------------------------
    # Spectrum methods

    def plot(self, **kwargs):
        """Display this `Spectrum` in a figure

        All arguments are passed onto the
        :class:`~gwpy.plotter.SpectrumPlot` constructor

        Returns
        -------
        SpectrumPlot
            a new :class:`~gwpy.plotter.SpectrumPlot` rendering
            of this `Spectrum`
        """
        from ..plotter import SpectrumPlot
        return SpectrumPlot(self, **kwargs)

    def zpk(self, zeros, poles, gain):
        """Filter this `Spectrum` by applying a zero-pole-gain filter

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
        spectrum : `Spectrum`
            the frequency-domain filtered version of the input data

        See Also
        --------
        Spectrum.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(zeros, poles, gain)

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `Spectrum`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `Spectrum`.

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
        result : `Spectrum`
            the filtered version of the input `Spectrum`

        See also
        --------
        Spectrum.zpk
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
            raise TypeError("Spectrum.filter() got an unexpected keyword "
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
                      "Spectrum.filter instead, with the same arguments",
                      DeprecationWarning)
        return self.filter(*args, **kwargs)

    @classmethod
    @with_import('lal')
    def from_lal(cls, lalfs, copy=True):
        """Generate a new `Spectrum` from a LAL `FrequencySeries` of any type
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
        """Convert this `Spectrum` into a LAL FrequencySeries.

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
        lalfs = create(self.name, lal.LIGOTimeGPS(self.epoch.gps),
                       self.f0.value, self.df.value, unit, self.size)
        lalfs.data.data = self.value
        return lalfs

    @classmethod
    def from_pycbc(cls, fs):
        """Convert a `pycbc.types.frequencyseries.FrequencySeries` into
        a `Spectrum`

        Parameters
        ----------
        fs : `pycbc.types.frequencyseries.FrequencySeries`
            the input PyCBC `~pycbc.types.frequencyseries.FrequencySeries`
            array

        Returns
        -------
        spectrum : `Spectrum`
            a GWpy version of the input frequency series
        """
        return cls(fs.data, f0=0, df=fs.delta_f, epoch=fs.epoch)

    @with_import('pycbc.types')
    def to_pycbc(self, copy=True):
        """Convert this `Spectrum` into a
        `pycbc.types.frequencyseries.FrequencySeries`

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        frequencyseries : `pycbc.types.frequencyseries.FrequencySeries`
            a PyCBC representation of this `Spectrum`
        """
        return types.FrequencySeries(self.data, delta_f=self.df.to('Hz').value,
                                     epoch=self.epoch.gps, copy=copy)
