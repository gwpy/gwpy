# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Hacky registration of median-mean

`scipy.signal` doesn't support median-mean averages, so this module
maps that name to an actual method provided by one of the other
FFT-API libraries (e.g. `pycbc`).

This module is deprecated, and will likely be removed before too long.
"""

import warnings

from . import _registry as fft_registry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def median_mean(*args, **kwargs):
    for api_name, api_func in filter(lambda x: x[0].endswith('_median_mean'),
                                     fft_registry.METHODS.items()):
        try:
            result = api_func(*args, **kwargs)
        except ImportError:
            continue
        # warn about deprecated name match
        warnings.warn(
            "no FFT method registered as 'median_mean', used {!r}; "
            "in the future this will raise a ValueError.".format(api_name),
            DeprecationWarning,
        )
        return result
    # try and call normally, mainly to raise the standard error
    raise KeyError("no PSD method registered with name 'median-mean'")


fft_registry.register_method(median_mean)
