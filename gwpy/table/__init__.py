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

"""This module extends the functionality of the :mod:`astropy.table`
library for reading/writing/manipulating hetergeneous data tables.

Importing the `~astropy.table.Table` object from here via

    >>> from gwpy.table import Table

loads extra input/output definitions available for
:meth:`~astropy.table.Table.read` and :meth:`~astropy.table.Table.write`.

Additionally, the `EventTable` object is provided to simplify working with
tables of time-stamped GW (or other) events.
"""

# load tables
from astropy.table import (Column, Table)
from .table import EventTable
from .gravityspy import GravitySpyTable

# attach unified I/O
from . import io

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
