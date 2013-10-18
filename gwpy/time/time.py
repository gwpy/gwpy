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

"""Convenience Time representations, used mainly in plotting
"""

from astropy.time import TimeISO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_HMS = ("hms", "%H:%M:%S", '{hour:02d}:{min:02d}:{sec:02d} ')
_HM = ("hm", "%H:%M", '{hour:02d}:{min:02d} ')
TimeISO.subfmts = tuple(TimeISO.subfmts + (_HMS, _HM))
