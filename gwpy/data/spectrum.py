# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Spectrum object
"""

import numpy
from scipy import interpolate
from astropy import units

from .nddata import NDData

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"
__version__ = version.version

__all__ = ['Spectrum']


class Spectrum(NDData):
    """A data array holding some metadata to represent a spectrum.

    """
    def __init__(self, data, f0=None, df=None, name=None, logscale=False,
                 unit=None, **kwargs):
        """Generate a new Spectrum.

        Parameters
        ----------
        data : numpy.ndarray, list
            Data values to initialise FrequencySeries
        f0 : `float`
            Starting frequency for this series
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
        super(Spectrum, self).__init__(data, name=name, unit=unit, **kwargs)
        self.f0 = f0
        self.df = df
        self.logscale = logscale

    @property
    def f0(self):
        return self._meta['f0']
    @f0.setter
    def f0(self, val):
        if val is None:
            self._meta['f0'] = val
        else:
            self._meta['f0'] = units.Quantity(val, units.Hertz)

    @property
    def df(self):
        return self._meta['df']
    @df.setter
    def df(self, val):
        if val is None:
            self._meta['df'] = None
        else:
            self._meta['df'] = units.Quantity(val, units.Hertz)

    @property
    def logscale(self):
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
