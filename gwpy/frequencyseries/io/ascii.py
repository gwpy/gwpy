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

"""ASCII I/O registrations for gwpy.frequencyseries objects
"""

from ...types.io.ascii import register_ascii_series_io
from .. import FrequencySeries

# -- registration -------------------------------------------------------------

register_ascii_series_io(FrequencySeries, format='txt')
register_ascii_series_io(FrequencySeries, format='csv', delimiter=',')
