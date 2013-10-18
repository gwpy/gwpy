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

"""Read Omicron-format ROOT files into Tables
"""

import sys
import numpy

from astropy.io import registry

from .utils import ROOTFile
from ...table import Table
from .. import version

if sys.version_info[0] < 3:
    range = xrange


__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version



def read_triggers(filename, tree='triggers', columns=None):
    if not isinstance(filename, basestring):
        filename = filename.name
    f = ROOTFile(filename)
    treename = tree
    tree = f.get_tree(tree)
    if not columns:
        columns = f.branches[treename]
    size = (int(tree.GetEntries()), len(columns))
    t = Table(data=numpy.empty(size), names=columns)
    for nrow in range(size[0]):
        tree.GetEntry(nrow)
        for ncol,col in enumerate(columns):
            t[col][nrow] = getattr(tree, col)
    return t


def identify_omicron(*args, **kwargs):
    filename = args[1][0]
    if not isinstance(filename, basestring):
        filename = filename.name
    if not filename.endswith('root'):
        return False
    t = ROOTFile(filename)
    if sorted(t.trees) == ['segments', 'triggers']:
        return True
    else:
        return False

registry.register_reader('omicron', Table, read_triggers, force=True)
registry.register_identifier('omicron', Table, identify_omicron)

