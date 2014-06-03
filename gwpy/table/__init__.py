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

"""This module extends the functionality of the :mod:`glue.ligolw`
library for reading/writing/manipulating LIGO_LW format XML tables.

The :mod:`glue.ligolw.lsctables` library defines a number of specific
tables used in GW data analysis; this module extends their functionality
with the unified input/output system (the .read() method).

Additionally, for event tables (burst, inspiral, and ringdown), methods
to calculate event rate are also attached.

Users can make the extensions available by either importing the
:mod:`~glue.ligolw.lsctables` module from gwpy as follows::

    >>> from gwpy.table import lsctables

or simply importing the :mod:`gwpy.table` module at any point before
using the lsctables module (even after importing it from glue)::

    >>> from glue.ligolw import lsctables
    >>> import gwpy.table

"""

import warnings
warnings.filterwarnings('ignore', 'column name', UserWarning)

from glue.ligolw.ligolw import (Column, Document)
from glue.ligolw.table import Table

# import all tables
from . import lsctables


# attach unified I/O
from .io import *

# attach rate methods
from .rate import (event_rate, binned_event_rates)

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Kipp Cannon <kipp.cannon@ligo.org>'
__version__ = version.version
__all__ = ['Column', 'Document', 'Table', 'lsctables']
