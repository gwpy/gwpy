# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read Omicron-format ROOT files into Tables
"""

import sys
import numpy

from astropy.io import registry

from .utils import ROOTFile
from ...table import Table
from .. import version

if sys.version_info.major < 3:
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

