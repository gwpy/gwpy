# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""Simple registrations for the FFT registry

`scipy.signal` doesn't provide alternatives to a simple mean average, so
this method maps the names of the other averages to an actual method provided
by one of the other FFT-API libraries (e.g. `pycbc`).
"""

import warnings
from functools import wraps

from . import registry as fft_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def map_fft_method(func):
    name = func.__name__

    @wraps(func)
    def mapped_method(*args, **kwargs):
        for (scaling, regname) in ((k, v) for k in fft_registry.METHODS for
                                   v in fft_registry.METHODS[k]):
            if regname.endswith('_{}'.format(name)):
                api_func = fft_registry.METHODS[scaling][regname]
                try:
                    return api_func(*args, **kwargs)
                except ImportError:
                    pass
                finally:
                    warnings.warn('no FFT method registered as {!r}, '
                                  'using {!r}, please consider specifying '
                                  'this directly in the future'.format(
                                      name, regname))
        raise RuntimeError("no underlying API method available for FFT method "
                           "{!r}, consider installing one of the extra "
                           "FFT-API libraries, see the online docs for full "
                           "details".format(name))

    return mapped_method


@map_fft_method
def welch(timeseries, segmentlength, noverlap=None, **kwargs):
    """Calculate a PSD of this `TimeSeries using a mean average of FFTs

    This method is dynamically linked at runtime to an underlying method
    provided by a third-party library.

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
    pass  # pragma: nocover


@map_fft_method
def bartlett(timeseries, segmentlength, **kwargs):
    pass  # pragma: nocover


bartlett.__doc__ = welch.__doc__.replace('mean average',
                                         'zero-overlap mean average')


@map_fft_method
def median(timeseries, segmentlength, noverlap=None, **kwargs):
    pass  # pragma: nocover


median.__doc__ = welch.__doc__.replace('mean average', 'median average')


@map_fft_method
def median_mean(timeseries, segmentlength, noverlap=None, **kwargs):
    pass  # pragma: nocover


median_mean.__doc__ = welch.__doc__.replace('mean average',
                                            'median-mean average')


for func in (welch, bartlett, median, median_mean,):
    fft_registry.register_method(func, scaling='density')
