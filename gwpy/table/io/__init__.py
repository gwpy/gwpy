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

"""Input/output methods for tabular data.
"""

from .. import _TABLES

from ... import version
from ...io import reader

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

# add the unified input/output system to each of the lsctables.
for name, table in _TABLES.iteritems():
    # define the read classmethod with docstring
    table.read = classmethod(reader(doc="""
        Read data into a `{0}`.

        Parameters
        ----------
        f : `file`, `str`
            open `file` in memory, or path to file on disk.
        columns : `list`, optional
            list of column name strings to read, default all.
        contenthandler : :class:`~glue.ligolw.ligolw.LIGOLWContentHandler`
            SAX content handler for parsing LIGO_LW documents.

        Returns
        -------
        table : :class:`~glue.ligolw.lsctables.{0}`
            `{0}` of data with given columns filled
        """.format(name)))

# import LIGO_LW I/O
from .ligolw import *

# import cache I/O
from .cache import *

# try importing ROOT-based I/O
try:
    from .omicron import *
except ImportError:
    pass
