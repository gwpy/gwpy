# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides an extension to the NDData class from astropy
with dynamic access to metadata as class attributes
"""

import numpy

from astropy.nddata import NDData as AstroData
from astropy.io import registry as io_registry

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class NDData(AstroData):
    """A subclass of the numpy array with added metadata access
    """
    def __init__(self, data, name=None, **kwargs):
        super(NDData, self).__init__(data, **kwargs)
        self.name = name

    @property
    def name(self):
        return self._meta['name']
    @name.setter
    def name(self, val):
        self._meta['name'] = val
    @name.deleter
    def name(self):
        del self._meta['name']

    def copy(self):
        """Make a copy of this series

        Returns
        -------
        new
            A copy of this series
        """
        out = self.__class__(self)
        out.data = numpy.array(out.data, subok=True, copy=True)
        return out

    def min(self, **kwargs):
        return self.data.min(**kwargs)
    min.__doc__ = numpy.ndarray.min.__doc__

    def max(self, **kwargs):
        return self.data.max(**kwargs)
    max.__doc__ = numpy.ndarray.max.__doc__

    def mean(self, **kwargs):
        return self.data.mean(**kwargs)
    mean.__doc__ = numpy.ndarray.mean.__doc__

    def median(self, **kwargs):
        return numpy.median(self.data, **kwargs)
    median.__doc__ = median.__doc__

    def _getAttributeNames(self):
        return self._meta.keys()

    def __getattr__(self, attr):
        if attr in self._meta.keys():
            return self._meta[attr]
        else:
            return self.__getattribute__(attr)

    read = classmethod(io_registry.read)
    write = io_registry.write

