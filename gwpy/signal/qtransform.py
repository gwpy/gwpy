# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016)
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

"""Python implementation of the tiled Q-transform scan.

This is a re-implementation of the original Q-transform scan from the Omega
pipeline, all credits for the original algorithm go to its
authors.
"""

from __future__ import division

import warnings
from math import (log, ceil, pi, isinf, exp)

from six import string_types
from six.moves import xrange

import numpy
from numpy import fft as npfft

from ..timeseries import TimeSeries

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['QTiling', 'QPlane', 'QTile']


class QObject(object):
    """Base class for Q-transform objects

    This object exists just to provide basic methods for all other
    Q-transform objects.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, duration, sampling, mismatch=.2):
        self.duration = float(duration)
        self.sampling = float(sampling)
        self.mismatch = float(mismatch)

    @property
    def deltam(self):
        """Fractional mismatch between neighbouring tiles

        :type: `float`
        """
        return 2 * (self.mismatch / 3.) ** (1/2.)


class QBase(QObject):
    """Base class for Q-transform objects with fixed Q

    This class just provides a property for Q-prime = Q / sqrt(11)
    """
    def __init__(self, q, duration, sampling, mismatch=.2):
        super(QBase, self).__init__(duration, sampling, mismatch=mismatch)
        self.q = float(q)

    @property
    def qprime(self):
        """Normalized Q `(q/sqrt(11))`
        """
        return self.q / 11**(1/2.)


class QTiling(QObject):
    """Iterable constructor of `QPlane` objects

    For a given Q-range, each of the resulting `QPlane` objects can
    be iterated over.

    Parameters
    ----------
    duration : `float`
        the duration of the data to be Q-transformed
    qrange : `tuple` of `float`
        `(low, high)` pair of Q extrema
    frange : `tuple` of `float`
        `(low, high)` pair of frequency extrema
    sampling : `float`
        sampling rate (in Hertz) of data to be Q-transformed
    mismatch : `float`
        maximum fractional mismatch between neighbouring tiles
    """
    def __init__(self, duration, sampling,
                 qrange=(4, 64), frange=(0, numpy.inf), mismatch=.2):
        super(QTiling, self).__init__(duration, sampling, mismatch=mismatch)
        self.qrange = (float(qrange[0]), float(qrange[1]))
        self.frange = [float(frange[0]), float(frange[1])]

        qlist = list(self._iter_qs())
        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * max(qlist) / (2 * pi * self.duration)
        maxf = self.sampling / 2 / (1 + 11**(1/2.) / min(qlist))
        if isinf(self.frange[1]):
            self.frange[1] = maxf
        elif self.frange[1] > maxf:  # truncate upper frequency to maximum
            warnings.warn('upper frequency of %.2f is too high for the given '
                          'Q range, resetting to %.2f'
                          % (self.frange[1], maxf))
            self.frange[1] = maxf

    @property
    def qs(self):  # pylint: disable=invalid-name
        """Array of Q values for this `QTiling`

        :type: `numpy.ndarray`
        """
        return numpy.array(list(self._iter_qs()))

    @property
    def whitening_duration(self):
        """The recommended data duration required for whitening
        """
        return max(t.whitening_duration for t in self)

    def _iter_qs(self):
        """Iterate over the Q values
        """
        # work out how many Qs we need
        cumum = log(self.qrange[1] / self.qrange[0]) / 2**(1/2.)
        nplanes = int(max(ceil(cumum / self.deltam), 1))
        dq = cumum / nplanes  # pylint: disable=invalid-name
        for i in xrange(nplanes):
            yield self.qrange[0] * exp(2**(1/2.) * dq * (i + .5))

    def __iter__(self):
        """Iterate over this `QTiling`

        Yields a `QPlane` at each Q value
        """
        for q in self._iter_qs():
            yield QPlane(q, self.frange, self.duration, self.sampling,
                         mismatch=self.mismatch)


class QPlane(QBase):
    """Iterable representation of a Q-transform plane

    For a given Q, an array of frequencies can be iterated over, yielding
    a `QTile` each time.

    Parameters
    ----------
    q : `float`
        the Q-value for this plane
    frange : `tuple` of `float`
        `(low, high)` range of frequencies for this plane
    duration : `float`
        the duration of the data to be Q-transformed
    sampling : `float`
        sampling rate (in Hertz) of data to be Q-transformed
    mismatch : `float`
        maximum fractional mismatch between neighbouring tiles
    """
    def __init__(self, q, frange, duration, sampling, mismatch=.2):
        super(QPlane, self).__init__(q, duration, sampling, mismatch=mismatch)
        self.frange = [float(frange[0]), float(frange[1])]

        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * pi * self.duration)
        if isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = self.sampling / 2 / (1 + 1/self.qprime)

    def __iter__(self):
        """Iterate over this `QPlane`

        Yields a `QTile` at each frequency
        """
        # for each frequency, yield a QTile
        for freq in self._iter_frequencies():
            yield QTile(self.q, freq, self.duration, self.sampling,
                        mismatch=self.mismatch)

    def _iter_frequencies(self):
        """Iterate over the frequencies of this `QPlane`
        """
        # work out how many frequencies we need
        minf, maxf = self.frange
        fcum_mismatch = log(maxf / minf) * (2 + self.q**2)**(1/2.) / 2.
        nfreq = int(max(1, ceil(fcum_mismatch / self.deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1 / self.duration
        # for each frequency, yield a QTile
        for i in xrange(nfreq):
            yield (minf *
                   exp(2 / (2 + self.q**2)**(1/2.) * (i + .5) * fstep) //
                   fstepmin * fstepmin)

    @property
    def frequencies(self):
        """Array of central frequencies for this `QPlane`

        :type: `numpy.ndarray`
        """
        return numpy.array(list(self._iter_frequencies()))

    @property
    def farray(self):
        """Array of frequencies for the lower-edge of each frequency bin

        :type: `numpy.ndarray`
        """
        bandwidths = 2 * pi ** (1/2.) * self.frequencies / self.q
        return self.frequencies - bandwidths / 2.

    @property
    def whitening_duration(self):
        """The recommended data duration required for whitening
        """
        return 2 ** (round(log(self.q / (2 * self.frange[0]), 2)))

    def transform(self, fseries, norm=True, epoch=None):
        """Calculate the energy `TimeSeries` for the given fseries

        Parameters
        ----------
        fseries : `~gwpy.frequencyseries.FrequencySeries`
            the complex FFT of a time-series data set
        norm : `bool`, `str`, optional
            normalize the energy of the output by the median (if `True` or
            ``'median'``) or the ``'mean'``, if `False` the output
            is the complex `~numpy.fft.ifft` output of the Q-tranform
        epoch : `~gwpy.time.LIGOTimeGPS`, `float`, optional
            the epoch of these data, only used for metadata in the output
            `TimeSeries`, and not requires if the input `fseries` has the
            epoch populated.

        Returns
        -------
        frequencies : `numpy.ndarray`
            array of frequencies for this `QPlane`
        transforms : `list` of `~gwpy.timeseries.TimeSeries`
            the complex energies of the Q-transform of the input `fseries`
            at each frequency

        See Also
        --------
        QTile.transform
            for details on the transform for a single `(Q, frequency)` tile
        """
        out = []
        for qtile in self:
            # get energy from transform
            out.append(qtile.transform(fseries, norm=norm,
                                       epoch=epoch))
        return self.frequencies, out


class QTile(QBase):
    """Representation of a tile with fixed Q and frequency
    """
    def __init__(self, q, frequency, duration, sampling, mismatch=.2):
        super(QTile, self).__init__(q, duration, sampling, mismatch=mismatch)
        self.frequency = frequency

    @property
    def bandwidth(self):
        """The bandwidth for tiles in this row

        :type: `float`
        """
        return 2 * pi ** (1/2.) * self.frequency / self.q

    @property
    def ntiles(self):
        """The number of tiles in this row

        :type: `int`
        """
        tcum_mismatch = self.duration * 2 * pi * self.frequency / self.q
        return next_power_of_two(tcum_mismatch / self.deltam)

    @property
    def windowsize(self):
        """The size of the frequency-domain window for this row

        :type: `int`
        """
        return 2 * int(self.frequency / self.qprime * self.duration) + 1

    def _get_indices(self):
        half = int((self.windowsize - 1) / 2)
        return numpy.arange(-half, half + 1)

    def get_window(self):
        """Generate the bi-square window for this row

        Returns
        -------
        window : `numpy.ndarray`
        """
        # real frequencies
        wfrequencies = self._get_indices() / self.duration
        # dimensionless frequencies
        xfrequencies = wfrequencies * self.qprime / self.frequency
        # normalize and generate bi-square window
        norm = self.ntiles / (self.duration * self.sampling) * (
            315 * self.qprime / (128 * self.frequency)) ** (1/2.)
        return (1 - xfrequencies ** 2) ** 2 * norm

    def get_data_indices(self):
        """Returns the index array of interesting frequencies for this row
        """
        return numpy.round(self._get_indices() + 1 +
                           self.frequency * self.duration).astype(int)

    @property
    def padding(self):
        """The `(left, right)` padding required for the IFFT

        :type: `tuple` of `int`
        """
        pad = self.ntiles - self.windowsize
        return (int((pad - 1)/2.), int((pad + 1)/2.))

    def transform(self, fseries, norm=True, epoch=None):
        """Calculate the energy `TimeSeries` for the given fseries

        Parameters
        ----------
        fseries : `~gwpy.frequencyseries.FrequencySeries`
            the complex FFT of a time-series data set
        norm : `bool`, `str`, optional
            normalize the energy of the output by the median (if `True` or
            ``'median'``) or the ``'mean'``, if `False` the output
            is the complex `~numpy.fft.ifft` output of the Q-tranform
        epoch : `~gwpy.time.LIGOTimeGPS`, `float`, optional
            the epoch of these data, only used for metadata in the output
            `TimeSeries`, and not requires if the input `fseries` has the
            epoch populated.

        Returns
        -------
        energy : `~gwpy.timeseries.TimeSeries`
            a `TimeSeries` of the complex energy from the Q-transform of
            this tile against the data. Basically just the raw output
            of the :meth:`~numpy.fft.ifft`
        """
        windowed = fseries[self.get_data_indices()] * self.get_window()
        # pad data, move negative frequencies to the end, and IFFT
        padded = numpy.pad(windowed, self.padding, mode='constant')
        wenergy = npfft.ifftshift(padded)
        # return a `TimeSeries`
        if epoch is None:
            epoch = fseries.epoch
        tdenergy = npfft.ifft(wenergy)
        cenergy = TimeSeries(tdenergy, x0=epoch,
                             dx=self.duration/tdenergy.size, copy=False)
        if norm:
            if isinstance(norm, string_types):
                norm = norm.lower()
            energy = type(cenergy)(
                cenergy.value.real ** 2. + cenergy.value.imag ** 2.,
                x0=cenergy.x0, dx=cenergy.dx, copy=False)
            if norm in (True, 'median'):
                meanenergy = energy.median()
            elif norm in ('mean',):
                meanenergy = energy.mean()
            else:
                raise ValueError("Invalid normalisation %r" % norm)
            return energy / meanenergy
        else:
            return cenergy


def next_power_of_two(x):
    """Return the smallest power of two greater than or equal to `x`
    """
    return 2**(ceil(log(x, 2)))
