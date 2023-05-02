# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
#
# This file is part of PDpy.
#
# PDpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PDpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PDpy.  If not, see <http://www.gnu.org/licenses/>.

"""This module has been renamed `pdpy.signal.spectral`.

DO NOT USE THIS MODULE.
"""

import warnings

# import things as they were named before
from .spectral import (  # noqa: F401
    get_default_fft_api,
    _lal as lal,
    _pycbc as pycbc,
    _registry as registry,
    _scipy as scipy,
    _ui as ui,
    _utils as utils,
)

warnings.warn(
    "this module has been renamed pdpy.signal.spectral and will be "
    "removed in a future release",
    DeprecationWarning,
)
