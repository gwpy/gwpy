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

from glue.ligolw.lsctables import TableByName

from ... import version
from ...io import reader

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

# add the unified input/output system to each of the lsctables.
for table in TableByName.itervalues():
    # define the read classmethod with docstring
    table.read = classmethod(reader(doc="""
        Read data into a `{0}`.

        Parameters
        ----------
        f : `file`, `str`, `CacheEntry`, `list`, `Cache`
            object representing one or more files. One of

            - an open `file`
            - a `str` pointing to a file path on disk
            - a formatted :class:`~glue.lal.CacheEntry` representing one file
            - a `list` of `str` file paths
            - a formatted :class:`~glue.lal.Cache` representing many files

        columns : `list`, optional
            list of column name strings to read, default all.

        nproc : `int`, optional, default: ``1``
            number of parallel processes with which to distribute file I/O,
            default: serial process.

            .. warning::

               This keyword argument is only applicable when reading a
               `list` (or `Cache`) of files.

        contenthandler : :class:`~glue.ligolw.ligolw.LIGOLWContentHandler`
            SAX content handler for parsing LIGO_LW documents.

            .. warning::

               This keyword argument is only applicable when reading from
               LIGO_LW-scheme XML files.

        Returns
        -------
        table : :class:`~glue.ligolw.lsctables.{0}`
            `{0}` of data with given columns filled
        """.format(table.__name__)))

# import LIGO_LW I/O
from .ligolw import *

# try importing ROOT-based I/O
try:
    from .omicron import *
except ImportError:
    pass

# import cache I/O
from .cache import *

