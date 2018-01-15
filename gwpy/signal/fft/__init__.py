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

"""FFT routines for GWpy

This sub-package provides wrappers of a given average spectrum package,
with each API registering relevant FrequencySeries-generation methods, so
that they can be accessed via the ``'method'`` keyword argument of the given
`TimeSeries` instance method, e.g.`TimeSeries.psd`.

See the online docs for full details.

For developers, to add another method to an existing API, simply write the
function into the sub-module and call

>>> fft_registry.register_method(my_new_method)

to register it.

To add another API from scratch, copy the format of the `gwpy.signal.fft.scipy`
module, and then add it to the import below. The imports are ordered by
preference (after `basic`).
"""

from importlib import import_module

from . import (
    basic,
    pycbc,  # <- PyCBC is preferred (better optimisation)
    lal,
    scipy,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def get_default_fft_api():
    """Return the preferred FFT-API library

    This is referenced to set the default methods for
    `~gwpy.timeseries.TimeSeries` methods (amongst others)

    Examples
    --------
    If you have :mod:`pycbc` installed:

    >>> from gwpy.signal.fft import get_default_fft_api
    >>> get_default_fft_api()
    'pycbc.psd'

    If you just have a basic installation (from `pip install gwpy`):

    >>> get_default_fft_api()
    'scipy'
    """
    for lib in ('pycbc.psd', 'lal',):
        try:
            import_module(lib)
        except ImportError:
            pass
        else:
            return lib
    return 'scipy'
