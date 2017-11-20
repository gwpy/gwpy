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

from six.moves import range

import numpy

from astropy.io import registry as io_registry

from ..types import (Quantity, Array2D)
from ..segments import Segment
from .core import FrequencySeries

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['SpectralVariance']


class SpectralVariance(Array2D):
    """A 2-dimensional array containing the variance histogram of a
    frequency-series `FrequencySeries`
    """
    _metadata_slots = FrequencySeries._metadata_slots + ('bins',)
    _default_xunit = FrequencySeries._default_xunit
    _rowclass = FrequencySeries

    def __new__(cls, data, bins, unit=None,
                f0=None, df=None, frequencies=None,
                name=None, channel=None, epoch=None, **kwargs):
        """Generate a new SpectralVariance histogram
        """
        # parse x-axis params
        if f0 is not None:
            kwargs['x0'] = f0
        if df is not None:
            kwargs['dx'] = df
        if frequencies is not None:
            kwargs['xindex'] = frequencies

        # generate SpectralVariance using the Series constructor
        new = super(Array2D, cls).__new__(cls, data, unit=unit, name=name,
                                          channel=channel, epoch=epoch,
                                          **kwargs)

        # set bins
        new.bins = bins

        return new

    # -- properties -----------------------------

    @property
    def bins(self):
        """Array of bin edges, including the rightmost edge

        :type: `astropy.units.Quantity`
        """
        return self._bins

    @bins.setter
    def bins(self, bins):
        if bins is None:
            del self.bins
            return
        bins = Quantity(bins)
        if bins.size != self.shape[1] + 1:
            raise ValueError(
                "SpectralVariance.bins must be given as a list of bin edges, "
                "including the rightmost edge, and have length 1 greater than "
                "the y-axis of the SpectralVariance data")
        self._bins = bins

    @bins.deleter
    def bins(self):
        try:
            del self._bins
        except AttributeError:
            pass

    # over-write yindex and yspan to communicate with bins
    @property
    def yindex(self):
        """List of left-hand amplitude bin edges
        """
        return self.bins[:-1]

    def yspan(self):
        """Amplitude range (low, high) spanned by this array
        """
        return Segment(self.bins[0], self.bins[-1])

    @property
    def dy(self):
        """Size of the first (lowest value) amplitude bin
        """
        return self.bins[1] - self.bins[0]

    @property
    def y0(self):
        """Starting value of the first (lowert value) amplitude bin
        """
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

    # -- i/O ------------------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `SpectralVariance`

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        source : `str`, :class:`~glue.lal.Cache`
            source of data, any of the following:

            - `str` path of single data file
            - `str` path of LAL-format cache file
            - :class:`~glue.lal.Cache` describing one or more data files,

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats

        Notes
        -----"""
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `SpectralVariance` to a file

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

    # -- methods --------------------------------

    def __getitem__(self, item):
        if isinstance(item, int):
            out = self.value[item].view(self._columnclass)
            out.unit = self.unit
            try:
                out.xindex = self.yindex
            except AttributeError:
                pass
            return out
        if isinstance(item, tuple) and item[0] == slice(None, None, None):
            out = self.value.__getitem__(item).view(self._rowclass)
            out.xindex = self.xindex
            return out
        if isinstance(item, tuple) and len(item) == 2:
            return self[item[0]][item[1]]
        return super(SpectralVariance, self).__getitem__(item)
    __getitem__.__doc__ = Array2D.__getitem__.__doc__

    @classmethod
    def from_spectrogram(cls, *spectrograms, **kwargs):
        """Calculate a new `SpectralVariance` from a
        :class:`~gwpy.spectrogram.core.Spectrogram`

        Parameters
        ----------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            input `Spectrogram` data

        bins : `~numpy.ndarray`, optional
            array of histogram bin edges, including the rightmost edge

        low : `float`, optional
            left edge of lowest amplitude bin, only read
            if ``bins`` is not given

        high : `float`, optional
            right edge of highest amplitude bin, only read
            if ``bins`` is not given

        nbins : `int`, optional
            number of bins to generate, only read if ``bins`` is not
            given, default: `500`

        log : `bool`, optional
            calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given, default: `False`

        norm : `bool`, optional
            normalise bin counts to a unit sum, default: `False`

        density : `bool`, optional
            normalise bin counts to a unit integral, default: `False`

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
        if not spectrograms:
            raise ValueError("Must give at least one Spectrogram")
        bins = kwargs.pop('bins', None)
        low = kwargs.pop('low', None)
        high = kwargs.pop('high', None)
        nbins = kwargs.pop('nbins', 500)
        log = kwargs.pop('log', False)
        norm = kwargs.pop('norm', False)
        density = kwargs.pop('density', False)
        if norm and density:
            raise ValueError("Cannot give both norm=True and density=True, "
                             "please pick one")

        # get data and bins
        spectrogram = spectrograms[0]
        data = numpy.vstack(s.value for s in spectrograms)
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
        bins = bins * spectrogram.unit

        # loop over frequencies
        out = numpy.zeros((data.shape[1], nbins))
        for i in range(data.shape[1]):
            out[i, :], bins = numpy.histogram(data[:, i], bins,
                                              density=density)
            if norm and out[i, :].sum():  # normalise
                out[i, :] /= out[i, :].sum()

        # return SpectralVariance
        name = '%s variance' % spectrogram.name
        new = cls(out, bins, epoch=spectrogram.epoch, name=name,
                  channel=spectrogram.channel, f0=spectrogram.f0,
                  df=spectrogram.df)
        return new

    def percentile(self, percentile):
        """Calculate a given spectral percentile for this `SpectralVariance`

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
        rows, columns = self.shape
        out = numpy.zeros(rows)
        # Loop over frequencies
        for i in range(rows):
            # Calculate cumulative sum for array
            cumsumvals = numpy.cumsum(self.value[i, :])

            # Find value nearest requested percentile
            abs_cumsumvals_minus_percentile = numpy.abs(cumsumvals -
                                                        percentile)
            minindex = abs_cumsumvals_minus_percentile.argmin()
            val = self.bins[minindex]
            out[i] = val

        name = '%s %s%% percentile' % (self.name, percentile)
        return FrequencySeries(out, epoch=self.epoch, channel=self.channel,
                               frequencies=self.bins[:-1], name=name)

    def plot(self, **kwargs):
        """Plot this `SpectralVariance`
        """
        from ..plotter import FrequencySeriesPlot
        return FrequencySeriesPlot(self, **kwargs)
