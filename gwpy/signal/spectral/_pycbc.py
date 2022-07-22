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

"""GWpy API to the pycbc.psd FFT routines

This module is deprecated and will be removed in a future release.
"""

from contextlib import nullcontext

from ...frequencyseries import FrequencySeries
from ._utils import scale_timeseries_unit
from . import _registry as fft_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def welch(timeseries, segmentlength, noverlap=None, scheme=None, **kwargs):
    """Calculate a PSD using Welch's method with a mean average

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.

    segmentlength : `int`
        number of samples in single average.

    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.

    scheme : `pycbc.scheme.Scheme`, optional
        processing scheme in which to execute FFT, default: `None`

    **kwargs
        other keyword arguments to pass to :func:`pycbc.psd.welch`

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`

    See also
    --------
    pycbc.psd.welch
    """
    from pycbc.psd import welch as pycbc_welch

    # default to 'standard' welch
    kwargs.setdefault('avg_method', 'mean')

    # get scheme
    if scheme is None:
        scheme = nullcontext()

    # generate pycbc FrequencySeries
    with scheme:
        pycbc_fseries = pycbc_welch(timeseries.to_pycbc(copy=False),
                                    seg_len=segmentlength,
                                    seg_stride=segmentlength-noverlap,
                                    **kwargs)

    # return GWpy FrequencySeries
    fseries = FrequencySeries.from_pycbc(pycbc_fseries, copy=False)
    fseries.name = timeseries.name
    fseries.override_unit(scale_timeseries_unit(
        timeseries.unit, scaling='density'))
    return fseries


def bartlett(*args, **kwargs):  # pylint: disable=missing-docstring
    kwargs['avg_method'] = 'mean'
    kwargs['noverlap'] = 0
    return welch(*args, **kwargs)


bartlett.__doc__ = welch.__doc__.replace('mean average',
                                         'non-overlapping mean average')


def median(*args, **kwargs):  # pylint: disable=missing-docstring
    kwargs['avg_method'] = 'median'
    return welch(*args, **kwargs)


median.__doc__ = welch.__doc__.replace('mean average', 'median average')


def median_mean(*args, **kwargs):  # pylint: disable=missing-docstring
    kwargs['avg_method'] = 'median-mean'
    return welch(*args, **kwargs)


median_mean.__doc__ = welch.__doc__.replace('mean average',
                                            'median-mean average')


# register new functions
for func in (welch, bartlett, median, median_mean):
    fft_registry.register_method(
        func,
        name='pycbc-{}'.format(func.__name__),
        deprecated=True,
    )
