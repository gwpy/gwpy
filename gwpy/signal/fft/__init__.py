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
module.
"""

from . import (  # pylint: disable=unused-import
    scipy,
    lal,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
