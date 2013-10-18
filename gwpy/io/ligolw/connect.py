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

"""Test converter for LIGO_LW tables to ATpy
"""

import re
import numpy

from glue.ligolw import (utils as ligolw_utils, table as ligolw_table,
                         lsctables)

from astropy.table import Table
from astropy.io import registry

from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_ligolw(filepath, table_name, columns=None):
    from . import utils
    # read table into GLUE LIGO_LW
    if columns:
        TableType = lsctables.TableByName[table_name]
        _oldcols = TableType.loadcolumns
        TableType.loadcolumns = columns
    if isinstance(filepath, basestring):
        xmldoc = ligolw_utils.load_filename(filepath)
    else:
        xmldoc,_ = ligolw_utils.load_fileobj(filepath)
    out = ligolw_table.get_table(xmldoc, table_name)
    if columns:
        TableType.loadcolumns = _oldcols
    return utils.to_table(out, columns=columns)


def identify_ligolw(*args, **kwargs):
    filename = args[1][0]
    if not isinstance(filename, basestring):
        filename = filename.name
    if filename.endswith('xml') or filename.endswith('xml.gz'):
        return True
    else:
        return False


registry.register_reader('ligolw', Table, read_ligolw, force=True)
registry.register_identifier('ligolw', Table, identify_ligolw)
