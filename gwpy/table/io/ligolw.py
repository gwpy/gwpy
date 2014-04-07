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

from astropy.io import registry

from .. import _TABLES
from ...io.ligolw import (table_from_file, identify_ligolw_file)
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

# register reader and auto-id for LIGO_LW
for name, table in _TABLES.iteritems():
    registry.register_reader('ligolw', table, _read_factory(table))
    registry.register_identifier('ligolw', table, identify_ligolw_file)
