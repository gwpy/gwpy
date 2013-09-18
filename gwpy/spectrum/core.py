# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Representation of a frequency-series spectrum
"""

import numpy
from scipy import (interpolate, signal)
from astropy import units
import lal

from ..data import Series
from ..detector import Channel
from ..time import Time
from ..timeseries import TimeSeries

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

__all__ = ['Spectrum']


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

    Attributes
    ----------
    name
    epoch
    f0
    df
    logf
    unit
    frequencies

    Methods
    -------
    plot
    filter
    to_lal
    from_lal
    """
    _metadata_slots = ['name', 'unit', 'epoch', 'channel', 'f0', 'df', 'logf']
    xunit = units.Unit('Hz')

    def __new__(cls, data, frequencies=None, name=None, unit=None,
                epoch=None, f0=None, df=None, channel=None, logf=False,
                **kwargs):
        """Generate a new Spectrum.
        """
        # parse Channel input
        if channel:
            channel = Channel(channel)
            name = name or channel.name
            unit = unit or channel.unit
        # generate Spectrum
        return super(Spectrum, cls).__new__(cls, data, name=name, unit=unit,
                                            f0=f0, df=df, channel=channel,
                                            frequencies=frequencies,
                                            logf=logf, **kwargs)

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


    logf = property(Series.logx.__get__, Series.logx.__set__,
                    Series.logx.__delete__,
                    """Boolean telling whether this `Spectrum` has a
                    logarithmic frequency scale
                    """)

    frequencies = property(fget=Series.index.__get__,
                           fset=Series.index.__set__,
                           fdel=Series.index.__delete__,
                           doc="""Series of frequencies for each sample""")

    # -------------------------------------------
    # Spectrum methods

    def to_logf(self, fmin=None, fmax=None, num=None):
        """Convert this Spectrum into logarithmic scale.

        Parameters
        ----------
        fmin : `float`, optional
            minimum frequency for new `Spectrum`
        fmax : `float, optional
            maxmimum frequency for new `Spectrum`
        num : `int`, optional
            length of new `Spectrum`

        Notes
        -----
        All arguments to this function default to the corresponding
        parameters of the existing `Spectrum`
        """
        num = num or self.shape[-1]
        fmin = fmin or float(self.f0) or float(self.f0 + self.df)
        fmax = fmax or float(self.f0 + self.shape[-1] * self.df)
        linf = self.frequencies.data
        logf = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), num=num)
        logf = logf[logf<linf.max()]
        interpolator = interpolate.interp1d(linf, self.data, axis=0)
        new = self.__class__(interpolator(logf), unit=self.unit,
                             wcs=self.wcs, uncertainty=self.uncertainty,
                             flags=self.flags)
        for attr in self._getAttributeNames():
            setattr(new, attr, getattr(self, attr))
        new.f0 = logf[0]
        new.df = logf[1]-logf[0]
        new.logf = True
        return new

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

    def filter(self, zeros=[], poles=[], gain=1, inplace=True):
        """Apply a filter to this `Spectrum` in zero-pole-gain format.

        Parameters
        ----------
        zeros : `list`, optional
            list of zeros for the transfer function
        poles : `list`, optional
            list of poles for the transfer function
        gain : `float`, optional
            amplitude gain factor
        inplace : `bool`, optional
            modify this `Spectrum` in-place, default `True`

        Returns
        -------
        Spectrum
            either a view of the current `Spectrum` with filtered data,
            or a new `Spectrum` with the filtered data
        """
        # generate filter
        f = self.frequencies.data
        if not zeros and not poles:
            fresp = numpy.ones_like(f) * gain
        else:
            lti = signal.lti(numpy.asarray(zeros), numpy.asarray(poles), gain)
            fresp = abs(lti.freqresp(f)[1])
        # filter in-place
        if inplace:
            self.data *= fresp
            return self
        # otherwise copy and filter
        else:
            out = self.copy()
            out *= fresp
            return out

    @classmethod
    def from_lal(cls, lalfs):
        """Generate a new `Spectrum` from a LAL `FrequencySeries` of any type
        """
        # write Channel
        channel = Channel(lalfs.name,
                          unit=lal.UnitToString(lalfs.sampleUnits),
                          dtype=lalfs.data.data.dtype)
        return cls(lalfs.data.data, channel=channel, f0=lalfs.f0,
                   df=lalfs.deltaF, unit=lal.UnitToString(lalfs.sampleUnits))

    def to_lal(self):
        """Convert this `Spectrum` into a LAL FrequencySeries

        Returns
        -------
        FrequencySeries
            an XLAL-format FrequencySeries of a given type, e.g.
            :lalsuite:`XLALREAL8FrequencySeries`
        """
        laltype = lalutils.LAL_TYPE_FROM_NUMPY[self.dtype.type]
        typestr = lalutils.LAL_TYPE_STR[laltype]
        create = getattr(lal, 'Create%sFrequencySeries' % typestr.upper())
        lalfs = create(self.name, lal.LIGOTimeGPS(self.epoch.gps),
                       float(self.f0), float(self.dt),
                       lal.lalDimensionlessUnit, self.size)
        lalfs.data.data = self.data
        return lalfs

