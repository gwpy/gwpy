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

"""FFT routines for GWpy

This sub-package provides PSD estimation routines based on functionality
provided by :mod:`scipy.signal`.

The methods provided here aren't meant to be called directly by the user,
but rather via instance methods of the :class:`~gwpy.timeseries.TimeSeries`
object.
"""

from ...utils.decorators import deprecated_function
from ._registry import (get_method, register_method)
from ._scipy import (
    bartlett,
    coherence,
    csd,
    median,
    rayleigh,
    welch,
)
from ._ui import (psd, spectrogram, average_spectrogram)

# register deprecated methods
from . import (
    _pycbc,  # deprecated
    _lal,  # deprecated
    _median_mean,  # deprecated
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@deprecated_function
def get_default_fft_api():
    """Return the preferred FFT-API library

    As of gwpy > 0.14.0|1.0.0 this always returns 'scipy'
    This is referenced to set the default methods for
    `~gwpy.timeseries.TimeSeries` methods (amongst others)

    Examples
    --------
    >>> get_default_fft_api()
    'scipy'
    """
    return 'scipy'
