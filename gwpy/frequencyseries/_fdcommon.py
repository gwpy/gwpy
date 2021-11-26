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

"""Common utilities for frequency-domain operations

This module holds code used by both the `FrequencySeries` and `Spectrogram`.
"""

import numpy

from scipy import signal

from ..signal.filter_design import parse_filter

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def fdfilter(data, *filt, **kwargs):
    """Filter a frequency-domain data object

    See also
    --------
    gwpy.frequencyseries.FrequencySeries.filter
    gwpy.spectrogram.Spectrogram.filter
    """
    # parse keyword args
    inplace = kwargs.pop('inplace', False)
    analog = kwargs.pop('analog', False)
    fs = kwargs.pop('sample_rate', None)
    if kwargs:
        raise TypeError(
            "filter() got an unexpected keyword argument "
            f"'{list(kwargs).pop()}'"
        )

    # parse filter
    if fs is None:
        fs = 2 * (data.shape[-1] * data.df).to('Hz').value
    form, filt = parse_filter(filt, analog=analog, sample_rate=fs)
    lti = signal.lti(*filt)

    # generate frequency response
    freqs = data.frequencies.value.copy()
    fresp = numpy.nan_to_num(abs(lti.freqresp(w=freqs)[1]))

    # apply to array
    if inplace:
        data *= fresp
        return data
    new = data * fresp
    return new
