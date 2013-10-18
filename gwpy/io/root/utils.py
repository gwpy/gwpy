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

"""Utilties for reading data from ROOT files
"""


from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class ROOTFile(object):
    """Base ROOT data reader.
    """
    def __init__(self, filename):
        from ROOT import TFile
        self._root = TFile(filename)

    @property
    def name(self):
        return self._root.GetName()

    @property
    def trees(self):
        return [tree.GetName() for tree in self._root.GetListOfKeys()]

    def get_tree(self, tree):
        if not tree in self.trees:
            raise ValueError("Tree '%s' not found in file '%s'"
                             % (tree, self.name))
        return self._root.Get(tree)

    @property
    def branches(self):
        out = dict()
        for name in self.trees:
            tree = self.get_tree(name)
            out[name] = [b.GetName() for b in tree.GetListOfBranches()]
        return out

    def get_branch(self, tree, branch):
        if branch not in self.branches[tree]:
            raise ValueError("Branch '%s' not found in tree '%s' in file '%s'"
                             % (branch, tree, self.name))
        return self.get_tree(tree).Get(branch)
