# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Representation of a frequency-series spectrum
"""

import numpy
from scipy import interpolate
from astropy import units

from ..data import NDData

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"
__version__ = version.version

__all__ = ['Spectrum']


class Spectrum(NDData):
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
    logscale
    unit

    Methods
    -------
    get_frequencies
    plot
    filter
    to_lal
    from_lal
    """
    def __init__(self, data, epoch=None, f0=None, df=None, name=None,
                 logscale=False, unit=None, **kwargs):
        """Generate a new Spectrum.
        """
        super(Spectrum, self).__init__(data, name=name, unit=unit, **kwargs)
        self.epoch = epoch
        self.f0 = f0
        self.df = df
        self.logscale = logscale

    @property
    def epoch(self):
        """Starting GPS time epoch for this `Spectrum`

        This attribute is recorded as a :class:`~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See :mod:`~astropy.time` for details on the `Time` object.
        """
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
    def f0(self):
        """Starting frequency for this `Spectrum`

        This attributes is recorded as a
        :class:`~astropy.units.quantity.Quantity` object, assuming a
        unit of 'Hertz'.
        """
        return self._meta['f0']
    @f0.setter
    def f0(self, val):
        if val is None:
            self._meta['f0'] = val
        else:
            self._meta['f0'] = units.Quantity(val, units.Hertz)

    @property
    def df(self):
        """Frequency spacing of this `Spectrum`

        This attributes is recorded as a
        :class:`~astropy.units.quantity.Quantity` object, assuming a
        unit of 'Hertz'.
        """
        return self._meta['df']
    @df.setter
    def df(self, val):
        if val is None:
            self._meta['df'] = None
        else:
            self._meta['df'] = units.Quantity(val, units.Hertz)

    @property
    def logscale(self):
        """Boolean telling whether this `Spectrum` has a logarithmic
        frequency scale
        """
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
        f = self.get_frequencies().data
        if not zeros or poles:
            fresp = numpy.ones_like(f) * gain
        else:
            lti = signal.lti(numpy.asarray(zeros), numpy.asarray(poles), gain)
            try:
                fresp = map(lambda w: numpy.polyval(lti.num, w*1j)/\
                                      numpy.polyval(lti.den, w*1j), f)
            except TypeError:
                fresp = map(lambda w: numpy.polyval(lti.num, w*1j)/\
                                      numpy.polyval([lti.den], w*1j), f)
            fresp = numpy.asarray(fresp)
        # filter in-place
        if inplace:
            self.data *= fresp
            return self
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
                   df=lalfs.deltaF, unit=lal.UnitToString(lalts.sampleUnits))

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

