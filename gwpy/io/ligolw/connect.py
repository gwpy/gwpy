# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Test converter for LIGO_LW tables to ATpy
"""

import re
import numpy

from glue.ligolw import (utils as ligolw_utils, table as ligolw_table,
                         lsctables)

from astropy.io import registry

from ... import version
from ...table import Table

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
