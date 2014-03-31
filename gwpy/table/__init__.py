# coding=utf-8
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

"""This module extends the functionality of the :mod:`glue.ligolw`
library for reading/writing/manipulating LIGO_LW format XML tables.
"""

import warnings
warnings.filterwarnings('ignore', 'column name', UserWarning)

from glue.ligolw.ligolw import (LIGOLWContentHandler, Column, Document)
from glue.ligolw.table import Table
from glue.ligolw import lsctables

lsctables.use_in(LIGOLWContentHandler)

from .rate import (event_rate, binned_event_rates)
from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Kipp Cannon <kipp.cannon@ligo.org>'
__version__ = version.version
__all__ = ['Column', 'Document', 'Table', 'lsctables']
