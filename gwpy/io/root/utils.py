# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
