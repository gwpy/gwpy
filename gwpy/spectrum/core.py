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
from math import pi

from scipy import signal

from astropy import units

from ..data import Series
from ..detector import Channel
from ..utils import (update_docstrings, with_import)


from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['Spectrum']


@update_docstrings
class Spectrum(Series):
    """A data array holding some metadata to represent a spectrum.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        array to initialise `Spectrum`
    f0 : `float`, optional
        starting frequency for this `Spectrum`
    df : `float`, optional
        frequency resolution
    name : `str`, optional
        name for this `Spectrum`
    unit : :class:`~astropy.units.Unit`, optional
        The units of the data

    Returns
    -------
    Spectrum
        a new Spectrum holding the given data
    """
    _metadata_slots = ['name', 'unit', 'epoch', 'channel', 'f0', 'df']
    xunit = units.Unit('Hz')

    def __new__(cls, data, frequencies=None, name=None, unit=None,
                epoch=None, f0=None, df=None, channel=None,
                **kwargs):
        """Generate a new Spectrum.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
        # generate Spectrum
        return super(Spectrum, cls).__new__(cls, data, name=name, unit=unit,
                                            f0=f0, df=df, channel=channel,
                                            frequencies=frequencies,
                                            epoch=epoch, **kwargs)

    # -------------------------------------------
    # Spectrum properties

    f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
                  """Starting frequency for this `Spectrum`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
                  """Frequency spacing of this `Spectrum`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    frequencies = property(fget=Series.index.__get__,
                           fset=Series.index.__set__,
                           fdel=Series.index.__delete__,
                           doc="""Series of frequencies for each sample""")

    # -------------------------------------------
    # Spectrum methods

    def plot(self, **kwargs):
        """Display this `Spectrum` in a figure

        All arguments are passed onto the
        :class:`~gwpy.plotter.spectrum.SpectrumPlot` constructor

        Returns
        -------
        SpectrumPlot
            a new :class:`~gwpy.plotter.spectrum.SpectrumPlot` rendering
            of this `Spectrum`
        """
        from ..plotter import SpectrumPlot
        return SpectrumPlot(self, **kwargs)

    def filter(self, *filt, **kwargs):
        """Apply the given filter to this `Spectrum`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `Spectrum`.

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
        # parse keyword args
        inplace = kwargs.pop('inplace', False)
        if kwargs:
            raise TypeError("Spectrum.filter() got an unexpected keyword "
                            "argument '%s'" % list(kwargs.keys())[0])
        fresp = abs(signal.freqs(b, a, self.frequencies * 2 * pi)[1])
        if inplace:
            self *= fresp
            return self
        else:
            new = self * fresp
            return new

    def filterba(self, *args, **kwargs):
        warnings.warn("filterba will be removed soon, please use "
                      "Spectrum.filter instead, with the same arguments",
                      DeprecationWarning)
        return self.filter(*args, **kwargs)

    @classmethod
    @with_import('lal')
    def from_lal(cls, lalfs):
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
                   df=lalfs.deltaF, epoch=lalfs.epoch)

    @with_import('lal')
    def to_lal(self):
        """Convert this `Spectrum` into a LAL FrequencySeries.

        Returns
        -------
        lalspec : `FrequencySeries`
            an XLAL-format FrequencySeries of a given type, e.g.
            :lalsuite:`REAL8FrequencySeries`

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
        lalfs.data.data = self.data
        return lalfs
