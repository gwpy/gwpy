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

"""Input/output methods for tabular data.
"""

from glue.ligolw.ligolw import LIGOLWContentHandler

from .. import lsctables
lsctables.use_in(LIGOLWContentHandler)

from ... import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

# import LIGO_LW I/O
from .ligolw import *

# try importing ROOT-based I/O
from .omicron import *

# import cache I/O
from .cache import *

