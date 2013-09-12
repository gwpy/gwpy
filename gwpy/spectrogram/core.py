# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Spectrogram object
"""

import numpy
from scipy import interpolate
from astropy import units

from ..data import NDData
from ..timeseries import TimeSeries
from ..spectrum import Spectrum

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org"
__version__ = version.version

__all__ = ['Spectrogram']


class Spectrogram(Spectrum, TimeSeries):
    """A 2-dimensional array holding a spectrogram of time-frequency
    amplitude.
    """
    def __init__(self, data, epoch=None, f0=None, dt=None, df=None, name=None,
                 logscale=None, unit=None, **kwargs):
        """Generate a new Spectrum.

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
        super(Spectrogram, self).__init__(data, name=name, unit=unit, **kwargs)
        if isinstance(data, self.__class__):
            epoch = epoch or data.epoch
            f0 = f0 is None and data.f0 or f0
            dt = dt is None and data.dt or dt
            df = df is None and data.df or df
            logscale = logscale is None and data.logscale or logscale
        self.epoch = epoch
        self.f0 = f0
        self.dt = dt
        self.df = df
        self.logscale = logscale

    def ratio(self, operand):
        """Calculate the ratio of this Spectrogram against a
        reference.

        Parameters
        ----------
        operand : str, Spectrum
            Spectrum against which to weight, or one of
            - 'mean' : weight against the mean of each spectrum
                       in this Spectrogram
            - 'median' : weight against the median of each spectrum
                       in this Spectrogram
 
        Returns
        -------
        R : `~gwpy.data.Spectrogram`
            A new spectrogram
        """
        if operand == 'mean':
            operand = self.mean(axis=0)
            unit = units.dimensionless_unscaled
        elif operand == 'median':
            operand = self.median(axis=0)
            unit = units.dimensionless_unscaled
        elif isinstance(operand, Spectrum):
            operand = Spectrum.data
            unit = self.unit / Spectrum.unit
        elif isinstance(operand, numbers.Number):
            unit = units.dimensionless_unscaled
        else:
            raise ValueError("operand '%s' unrecognised, please give Spectrum "
                             "or one of: 'mean', 'median'")
        return self.__class__(self.data/operand, unit=unit, epoch=self.epoch,
                              f0=self.f0, name=self.name, dt=self.dt,
                              df=self.df, logscale=self.logscale)

    def __str__(self):
        return "Spectrogram('{0}')".format(self.name)

    def __repr__(self):
        return ("<Spectrogram object: name='{0}' epoch={1} f0={2} "
                "dt={3} df={4}>".format(
                    self.name, self.epoch, self.f0, self.dt, self.df))

    def plot(self, **kwargs):
        from ..plotter import SpectrogramPlot
        return SpectrogramPlot(self, **kwargs)

    def to_logscale(self, fmin=None, fmax=None, num=None):
        """Convert this Spectrum into logarithmic scale.
        """
        num = num or self.shape[-1]
        fmin = fmin or float(self.f0) or float(self.f0 + self.df)
        fmax = fmax or float(self.f0 + self.shape[-1] * self.df)
        linf = self.get_frequencies().data
        logf = numpy.logspace(numpy.log10(fmin), numpy.log10(fmax), num=num)
        logf = logf[logf<linf.max()]
        new = self.__class__(numpy.zeros((self.shape[0], logf.size)),
                             epoch=self.epoch, dt=self.dt, unit=self.unit)
        for i in range(self.shape[0]):
            interpolator = interpolate.interp1d(linf[-logf.size:],
                                                self.data[i,-logf.size:],
                                                axis=0)
            new.data[i,:] = interpolator(logf)
        for attr in self._getAttributeNames():
            setattr(new, attr, getattr(self, attr))
        new.f0 = logf[0]
        new.df = logf[1]-logf[0]
        new.logscale = True
        return new

    def __getitem__(self, item):
        if isinstance(item, int):
            return Spectrum(self.data[item], unit=self.unit,
                            name=self.name, f0=self.f0, df=self.df)
        elif isinstance(item, tuple):
            return self[item[0]][item[1]]
        new = super(Spectrogram, self).__getitem__(item)
        new.f0 = self.f0
        new.df = self.df
        new.name = self.name
        if isinstance(item, tuple):
            return new
        if item.start:
            new.f0 = self.f0 + item.start * self.df
        if item.step:
            new.df = self.df * item.step
        return new

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
        data = numpy.vstack(spectra)
        s1 = spectra[0]
        if not all(s.f0==s1.f0 for s in spectra):
            raise ValueError("Cannot stack spectra with different f0")
        if not all(s.df==s1.df for s in spectra):
            raise ValueError("Cannot stack spectra with different df")
        if (not all(s.logscale for s in spectra) and
            any(s.logscale for s in spectra)):
            raise ValueError("Cannot stack spectra with different log-scaling")
        kwargs.setdefault('name', s1.name)
        kwargs.setdefault('epoch', s1.epoch)
        kwargs.setdefault('f0', s1.f0)
        kwargs.setdefault('df', s1.df)
        kwargs.setdefault('unit', s1.unit)
        return Spectrogram(data, logscale=s1.logscale, **kwargs)
