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

"""Shared GWF-file identifier
"""

from glue.lal import CacheEntry

from astropy.io import registry

from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)
from .... import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def identify_gwf(*args, **kwargs):
    """Determine an input file as written in GWF-format.
    """
    filename = args[1]
    ce = args[3]
    if isinstance(ce, CacheEntry):
        filename = ce.path
    if isinstance(filename, str) and filename.endswith('gwf'):
        return True
    else:
        return False

registry.register_identifier('gwf', TimeSeries, identify_gwf)
registry.register_identifier('gwf', TimeSeriesDict, identify_gwf)
registry.register_identifier('gwf', StateVector, identify_gwf)
registry.register_identifier('gwf', StateVectorDict, identify_gwf)
