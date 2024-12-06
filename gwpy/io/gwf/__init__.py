# Copyright (C) Cardiff University (2024-)
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

"""I/O utilities for GWF files.

Most of the actual GWF file I/O is delegated to one of the 'backend'
implementations that rely on a third-party library.
"""

# support for GWF I/O backends
from .backend import *

# basic functionality that should be underpinned by a backend implementation
from .core import *

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
