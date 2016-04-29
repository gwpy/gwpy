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

"""`FrequencySeries` calculation methods using the SciPy module.
"""

import numpy

from astropy import units

from .core import FrequencySeries
from .registry import register_method
from ..utils import import_method_dependency
from .utils import scale_timeseries_units

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def welch(timeseries, segmentlength, noverlap=None, **kwargs):
    """Calculate the PSD using the scipy Welch method.
    """
    # get module
    signal = import_method_dependency('scipy.signal')
    # calculate PSD
    f, psd_ = signal.welch(timeseries.value, noverlap=noverlap,
                           fs=timeseries.sample_rate.decompose().value,
                           nperseg=segmentlength, **kwargs)
    # generate FrequencySeries and return
    unit = scale_timeseries_units(timeseries.unit,
                                  kwargs.get('scaling', 'density'))
    return FrequencySeries(psd_, unit=unit, frequencies=f,
                           name=timeseries.name, epoch=timeseries.epoch,
                           channel=timeseries.channel)

register_method(welch)


def bartlett(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using the Bartlett average method.
    """
    import_method_dependency('scipy.signal')
    kwargs.pop('noverlap', None) 
    return welch(timeseries, segmentlength, noverlap=0, **kwargs)

register_method(bartlett)


def rayleigh(timeseries, segmentlength, noverlap=0, **kwargs):
    """Calculate a Rayleigh statistic spectrum
    """
    stepsize = segmentlength - noverlap
    if noverlap:
        numsegs = 1 + int((timeseries.size - segmentlength) / float(noverlap))
    else:
        numsegs = int(timeseries.size // segmentlength)
    tmpdata = numpy.ndarray((numsegs, int(segmentlength//2 + 1)))
    for i in range(numsegs):
        ts = timeseries[i*stepsize:i*stepsize+segmentlength]
        tmpdata[i, :] = welch(ts, segmentlength)
    std = tmpdata.std(axis=0)
    mean = tmpdata.mean(axis=0)
    return FrequencySeries(std/mean, unit='', copy=False, f0=0,
                           df=timeseries.sample_rate.value/segmentlength,
                           channel=timeseries.channel,
                           name='Rayleigh spectrum of %s' % timeseries.name)

register_method(rayleigh)


def csd(timeseries, othertimeseries, segmentlength, noverlap=None, **kwargs):
    """Calculate the CSD using scipy's csd method (which uses Welch's method)
    """
    # get module
    signal = import_method_dependency('scipy.signal')
    # calculate CSD
    f, csd_ = signal.csd(timeseries.value, othertimeseries.value,
                         noverlap=noverlap,
                         fs=timeseries.sample_rate.decompose().value,
                         nperseg=segmentlength, **kwargs)
    # generate FrequencySeries and return
    unit = scale_timeseries_units(timeseries.unit,
                                  kwargs.get('scaling', 'density'))
    return FrequencySeries(
       csd_, unit=unit, frequencies=f,
       name=str(timeseries.name)+'---'+str(othertimeseries.name),
       epoch=timeseries.epoch, channel=timeseries.channel)

register_method(csd)
