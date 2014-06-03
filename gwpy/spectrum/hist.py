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

"""This module provides a spectral-variation histogram class
"""

import numpy
import sys

if sys.version_info[0] < 3:
    range = xrange

from .. import version
__version__ = version.version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

from ..data import (Array, Array2D)
from .core import Spectrum
from ..detector import Channel

__all__ = ['SpectralVariance']


class SpectralVariance(Array2D):
    """A 2-dimensional array containing the variance histogram of a
    frequency-series `Spectrum`
    """
    _metadata_slots = ['name', 'unit', 'epoch', 'df', 'f0', 'bins']
    xunit = Spectrum.xunit
    def __new__(cls, data, name=None, channel=None, epoch=None, unit=None,
                f0=None, df=None, bins=None, yunit=None, **kwargs):
        """Generate a new SpectralVariance
        """
        # parse Channel input
        if channel:
            channel = Channel(channel)
            name = name or channel.name
            unit = unit or channel.unit
        # generate Spectrogram
        return super(SpectralVariance, cls).__new__(cls, data, name=name,
                                                    yunit=yunit,
                                                    channel=channel,
                                                    epoch=epoch,
                                                    f0=f0, df=df,
                                                    bins=bins,
                                                    **kwargs)

    # -------------------------------------------
    # SpectralVariance properties

    @property
    def normed(self):
        return self._normed

    @property
    def density(self):
        return self._density

    @property
    def bins(self):
        return self.metadata['bins']

    @bins.setter
    def bins(self, bins):
        if not isinstance(bins, Array):
            bins = Array(bins, name='%s bins' % self.name, unit=self.yunit,
                         epoch=self.epoch, channel=self.channel)
        assert bins.size == self.shape[1] + 1,\
               ("SpectralVariance.bins must be given as a list of bin edges, "
                "including the rightmost edge, and have length 1 greater than "
                "the y-axis of the SpectralVariance data")
        self.metadata['bins'] = bins

    # over-write yindex to communicate with bins
    @property
    def yindex(self):
        return self.bins[:-1]

    @property
    def dy(self):
        return self.bins[1] - self.bins[0]

    @property
    def y0(self):
        return self.bins[0]

    f0 = property(Array2D.x0.__get__, Array2D.x0.__set__,
                  Array2D.x0.__delete__,
                  """Starting frequency for this `Spectrogram`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    df = property(Array2D.dx.__get__, Array2D.dx.__set__,
                  Array2D.dx.__delete__,
                  """Frequency spacing of this `Spectogram`

                  This attributes is recorded as a
                  :class:`~astropy.units.quantity.Quantity` object, assuming a
                  unit of 'Hertz'.
                  """)

    frequencies = property(fget=Array2D.xindex.__get__,
                           fset=Array2D.xindex.__set__,
                           fdel=Array2D.xindex.__delete__,
                           doc="""Array of frequencies for each sample""")

    # -------------------------------------------
    # SpectralVariance methods

    @classmethod
    def from_spectrogram(cls, *spectrograms, **kwargs):
        """Calculate a new `SpectralVariance` from a
        :class:`~gwpy.spectrogram.core.Spectrogram`

        Parameters
        ----------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            input `Spectrogram` data
        bins : :class:`~numpy.ndarray`, optional, default `None`
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
        # parse args and kwargs
        assert len(spectrograms) >= 1,\
               "Must give at least one Spectrogram"
        bins = kwargs.pop('bins', None)
        low = kwargs.pop('low', None)
        high = kwargs.pop('high', None)
        nbins = kwargs.pop('nbins', 500)
        log = kwargs.pop('log', False)
        norm = kwargs.pop('norm', False)
        density = kwargs.pop('density', False)
        assert not (norm and density),\
               "Cannot give both norm=True and density=True, please pick one"

        # get data and bins
        spectrogram = spectrograms[0]
        data = numpy.vstack(s.data for s in spectrograms)
        ubins = (bins is not None)
        if bins is None:
            if low is None and log:
                low = numpy.log10(data.min() / 2)
            elif low is None:
                low = data.min()/2
            elif log:
                low = numpy.log10(low)
            if high is None and log:
                high = numpy.log10(data.max() * 2)
            elif high is None:
                high = data.max() * 2
            elif log:
                high = numpy.log10(high)
            if log:
                bins = numpy.logspace(low, high, num=nbins+1)
            else:
                bins = numpy.linspace(low, high, num=nbins+1)
        nbins = bins.size-1

        # loop over frequencies
        out = numpy.zeros((data.shape[1], nbins))
        for i in range(data.shape[1]):
            out[i,:], bins = numpy.histogram(data[:,i], bins, density=density)
            if norm and out[i,:].sum():
                out[i,:] /= out[i,:].sum()

        # return SpectralVariance
        name = '%s variance' % spectrogram.name
        new = SpectralVariance(out, epoch=spectrogram.epoch,
                               yunit=spectrogram.unit, name=name,
                               channel=spectrogram.channel,
                               f0=spectrogram.f0, df=spectrogram.df,
                               logy=log, bins=bins)
        new._normed = norm
        new._density = density
        return new

    def percentile(self, percentile):
        """Calculate a given spectral percentile for this `SpectralVariance`

        Parameters
        ----------
        percentile : `float`
            percentile (0 - 100) of the bins to compute

        Returns
        -------
        spectrum : :class:`~gwpy.spectrum.core.Spectrum`
            the given percentile `Spectrum` calculated from this
            `SpectralVaraicence`
        """
        rows, columns = self.data.shape
        out = numpy.zeros(rows)
        # Loop over frequencies
        for i in range(rows):
            # Calculate cumulative sum for array
            cumsumvals = numpy.cumsum(self.data[i, :])

            # Find value nearest requested percentile
            abs_cumsumvals_minus_percentile = numpy.abs(cumsumvals -
                                                        percentile)
            minindex = abs_cumsumvals_minus_percentile.argmin()
            val = self.bins[minindex]
            out[i] = val
        name = '%s %s%% percentile' % (self.name, percentile)
        return Spectrum(out, epoch=self.epoch, frequencies=self.bins[:-1],
                        channel=self.channel, name=name)

    def plot(self, **kwargs):
        """Plot this `SpectralVariance`.
        """
        from ..plotter import SpectrumPlot
        return SpectrumPlot(self, **kwargs)
