# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""GWpy API to the scipy.signal FFT routines
"""

import numpy

import scipy.signal

from ...frequencyseries import FrequencySeries
from ._utils import scale_timeseries_unit
from . import _registry as fft_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- density scaling methods --------------------------------------------------

def _spectral_density(timeseries, segmentlength, noverlap=None, name=None,
                      sdfunc=scipy.signal.welch, **kwargs):
    """Calculate a generic spectral density of this `TimeSeries`
    """
    # compute spectral density
    freqs, psd_ = sdfunc(
        timeseries.value,
        noverlap=noverlap,
        fs=timeseries.sample_rate.decompose().value,
        nperseg=segmentlength,
        **kwargs
    )
    # generate FrequencySeries and return
    unit = scale_timeseries_unit(
        timeseries.unit,
        kwargs.get('scaling', 'density'),
    )
    return FrequencySeries(
        psd_,
        unit=unit,
        frequencies=freqs,
        name=(name or timeseries.name),
        epoch=timeseries.epoch,
        channel=timeseries.channel,
    )


def welch(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Welch's method
    """
    kwargs.setdefault('average', 'mean')
    return _spectral_density(timeseries, segmentlength, **kwargs)


def bartlett(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Bartlett's method
    """
    kwargs.pop('noverlap', None)
    return _spectral_density(timeseries, segmentlength, noverlap=0, **kwargs)


def median(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Welch's method with a median average
    """
    kwargs.setdefault('average', 'median')
    return _spectral_density(timeseries, segmentlength, **kwargs)


# register
for func in (welch, bartlett, median):
    fft_registry.register_method(func, name=func.__name__)

    # DEPRECATED:
    fft_registry.register_method(func, name='scipy-{}'.format(func.__name__))


# -- others -------------------------------------------------------------------

def rayleigh(timeseries, segmentlength, noverlap=0, window='hann'):
    """Calculate a Rayleigh statistic spectrum

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, passing `None` will
        choose based on the window method, default: ``0``

    window : `str`, `numpy.ndarray`, optional
        window function to apply to ``timeseries`` prior to FFT,
        see :func:`scipy.signal.get_window` for details on acceptable
        formats

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`
    """
    stepsize = segmentlength - noverlap
    if noverlap:
        numsegs = 1 + int((timeseries.size - segmentlength) / float(noverlap))
    else:
        numsegs = int(timeseries.size // segmentlength)
    tmpdata = numpy.ndarray((numsegs, int(segmentlength//2 + 1)))
    for i in range(numsegs):
        tmpdata[i, :] = welch(
            timeseries[i*stepsize:i*stepsize+segmentlength],
            segmentlength, window=window)
    std = tmpdata.std(axis=0)
    mean = tmpdata.mean(axis=0)
    return FrequencySeries(std/mean, unit='', copy=False, f0=0,
                           epoch=timeseries.epoch,
                           df=timeseries.sample_rate.value/segmentlength,
                           channel=timeseries.channel,
                           name='Rayleigh spectrum of %s' % timeseries.name)


def csd(timeseries, other, segmentlength, noverlap=None, **kwargs):
    """Calculate the CSD of two `TimeSeries` using Welch's method

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        time-series of data

    other : `~gwpy.timeseries.TimeSeries`
        time-series of data

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

    **kwargs
        other keyword arguments are passed to :meth:`scipy.signal.csd`

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`

    See also
    --------
    scipy.signal.csd
    """
    # calculate CSD
    kwargs.setdefault('y', other.value)
    return _spectral_density(
        timeseries, segmentlength, noverlap=noverlap,
        name=str(timeseries.name)+'---'+str(other.name),
        sdfunc=scipy.signal.csd, **kwargs)


def coherence(timeseries, other, segmentlength, noverlap=None, **kwargs):
    """Calculate the coherence between two `TimeSeries` using Welch's method

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        time-series of data

    other : `~gwpy.timeseries.TimeSeries`
        time-series of data

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

    **kwargs
        other keyword arguments are passed to :meth:`scipy.signal.coherence`

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`

    See also
    --------
    scipy.signal.coherence
    """
    # calculate CSD
    kwargs.setdefault('y', other.value)
    out = _spectral_density(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        name="Coherence between {} and {}".format(timeseries.name, other.name),
        sdfunc=scipy.signal.coherence,
        **kwargs,
    )
    out.override_unit("coherence")
    return out
