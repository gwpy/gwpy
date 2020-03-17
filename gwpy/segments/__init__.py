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

"""This module provides classes for generating and manipulating
data segments of the form [gps_start, gps_end).

The core of this module is adapted from |ligo-segments|_.
"""

from .segments import (Segment, SegmentList, SegmentListDict)
from .flag import (DataQualityFlag, DataQualityDict)

from . import io

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    'Segment',
    'SegmentList',
    'SegmentListDict',
    'DataQualityFlag',
    'DataQualityDict',
]
