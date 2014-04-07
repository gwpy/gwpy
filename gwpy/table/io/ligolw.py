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

"""Read LIGO_LW documents into glue.ligolw.table.Table objects.
"""

import inspect

from glue.ligolw.table import Table
from glue.ligolw import lsctables

from astropy.io import registry

from ...io.ligolw import (table_from_file, identify_ligolw_file)
from ...io import reader
from ... import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version
__all__ = []


def _read_factory(table_):
    """Define a custom function to read this table from a LIGO_LW file.
    """
    def _read(f, **kwargs):
        return table_from_file(f, table_.tableName, **kwargs)
    return _read


# add the unified input/output system to each of the lsctables.
for name, table in inspect.getmembers(
        lsctables, lambda t: inspect.isclass(t) and issubclass(t, Table)):
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

    # register reader and auto-id for LIGO_LW
    registry.register_reader('ligolw', table, _read_factory(table))
    registry.register_identifier('ligolw', table, identify_ligolw_file)
