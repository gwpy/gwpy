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

"""`Spectrum` calculation methods using the SciPy module.
"""

from astropy import units

from .core import Spectrum
from .registry import register_method
from ..utils import import_method_dependency
from .utils import scale_timeseries_units
from .. import version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def welch(timeseries, segmentlength, noverlap=None, **kwargs):
    """Calculate the PSD using the scipy Welch method.
    """
    # get module
    signal = import_method_dependency('scipy.signal')
    # calculate PSD
    f, psd_ = signal.welch(timeseries.data, noverlap=noverlap,
                           fs=timeseries.sample_rate.decompose().value,
                           nperseg=segmentlength, **kwargs)
    # generate Spectrum and return
    spec = psd_.view(Spectrum)
    spec.frequencies = f
    spec.name = timeseries.name
    spec.epoch = timeseries.epoch
    spec.channel = timeseries.channel
    spec.unit = scale_timeseries_units(timeseries.unit,
                                       kwargs.get('scaling', 'density'))
    return spec

register_method(welch)


def bartlett(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using the Bartlett average method.
    """
    import_method_dependency('scipy.signal')
    return welch(timeseries, segmentlength, noverlap=0, **kwargs)

register_method(bartlett)
