#!/usr/bin/env python
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

"""GWpy tests.
"""

# XXX: HACK for astropy breaking test suite
from astropy.utils.data import _deltemps
import atexit
try:
    idx = zip(*atexit._exithandlers)[0].index(_deltemps)
except ValueError:
    pass
else:
    atexit._exithandlers.pop(idx)

# run tests
from . import (timeseries, detector)
