# Copyright (c) 2026 Cardiff University
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

"""Example setup for gwpy.

Helpers for when the example tests are run with pytest.
"""

import contextlib

# -- gpstime compatibility
# import gpstime now so that the first import is never during a
# pytest run using xdist where the multiple workers try and download
# the leap seconds file at the same time, and then fall over each other

with contextlib.suppress(ImportError):
    import gpstime  # noqa: F401
