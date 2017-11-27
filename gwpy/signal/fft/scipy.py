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

"""GWpy API to the scipy.signal FFT routines
"""

from __future__ import absolute_import

import numpy

import scipy.signal

from ...frequencyseries import FrequencySeries
from .utils import scale_timeseries_unit
from . import registry as fft_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- density scaling methods --------------------------------------------------

def welch(timeseries, segmentlength, noverlap=None, **kwargs):
    """Calculate a PSD of this `TimeSeries` using Welch's method.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`

    See also
    --------
    scipy.signal.welch
    """
    # calculate PSD
    freqs, psd_ = scipy.signal.welch(
        timeseries.value, noverlap=noverlap,
        fs=timeseries.sample_rate.decompose().value,
        nperseg=segmentlength, **kwargs)
    # generate FrequencySeries and return
    unit = scale_timeseries_unit(timeseries.unit,
                                 kwargs.get('scaling', 'density'))
    return FrequencySeries(psd_, unit=unit, frequencies=freqs,
                           name=timeseries.name, epoch=timeseries.epoch,
                           channel=timeseries.channel)


def bartlett(timeseries, segmentlength, **kwargs):
    """Calculate a PSD of this `TimeSeries` using Bartlett's method

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`

    See also
    --------
    scipy.signal.welch
    """
    kwargs.pop('noverlap', None)
    return welch(timeseries, segmentlength, noverlap=0, **kwargs)


# register
for func in (welch, bartlett,):
    fft_registry.register_method(func, name='scipy-{}'.format(func.__name__),
                                 scaling='density')


# -- other scaling methods ----------------------------------------------------

def rayleigh(timeseries, segmentlength, noverlap=0):
    """Calculate a Rayleigh statistic spectrum

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

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
            segmentlength)
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
    try:
        freqs, csd_ = scipy.signal.csd(
            timeseries.value, other.value, noverlap=noverlap,
            fs=timeseries.sample_rate.decompose().value,
            nperseg=segmentlength, **kwargs)
    except AttributeError as exc:
        exc.args = ('{}, scipy>=0.16 is required'.format(str(exc)),)
        raise

    # generate FrequencySeries and return
    unit = scale_timeseries_unit(timeseries.unit,
                                 kwargs.get('scaling', 'density'))
    return FrequencySeries(
        csd_, unit=unit, frequencies=freqs,
        name=str(timeseries.name)+'---'+str(other.name),
        epoch=timeseries.epoch, channel=timeseries.channel)


# register
for func in (rayleigh, csd,):
    try:
        fft_registry.register_method(func, scaling='other')
    except KeyError:
        pass
    fft_registry.register_method(func, name='scipy-{}'.format(func.__name__),
                                 scaling='other')
