# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides an extension to the NDData class from astropy
with dynamic access to metadata as class attributes
"""

import numpy

from astropy.nddata import NDData as AstroData
from astropy.units import Quantity
from astropy.io import registry as io_registry

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class NDData(AstroData):
    """A subclass of the numpy array with added metadata access
    """
    def __init__(self, data, name=None, **kwargs):
        kwargs.setdefault('mask', None)
        super(NDData, self).__init__(numpy.asarray(data), **kwargs)
        if isinstance(data, self.__class__):
            name = name or data.name
        self.name = name

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
    median.__doc__ = numpy.median.__doc__

    def __pow__(self, y, z=None):
        out = self.copy()
        out.data **= y
        if out.unit is not None:
            out.unit **= y
        return out
    __pow__.__doc__ = numpy.ndarray.__pow__.__doc__

    def __mul__(self, other):
        out = self.copy()
        out.data *= other
        return out
    __mul__.__doc__ = numpy.ndarray.__mul__.__doc__

    def __div__(self, other):
        out = self.copy()
        out.data /= other
        return out
    __div__.__doc__ = numpy.ndarray.__div__.__doc__

    def _getAttributeNames(self):
        return self.meta.keys()

    def __gt__(self, val):
        return self.data > val
    __gt__.__doc__ = numpy.ndarray.__gt__.__doc__

    def __lt__(self, val):
        return self.data < val
    __lt__.__doc__ = numpy.ndarray.__lt__.__doc__

    def __ge__(self, val):
        return self.data >= val
    __ge__.__doc__ = numpy.ndarray.__ge__.__doc__

    def __le__(self, val):
        return self.data <= val
    __le__.__doc__ = numpy.ndarray.__le__.__doc__

    def __eq__(self, val):
        return self.data == val
    __eq__.__doc__ = numpy.ndarray.__eq__.__doc__

    def __ne__(self, val):
        return self.data != val
    __ne__.__doc__ = numpy.ndarray.__ne__.__doc__

    def __neg__(self, val):
        out = self.copy()
        out.data = -out.data
        return out
    __neg__.__doc__ = numpy.ndarray.__neg__.__doc__

    def __pos__(self, val):
        out = self.copy()
        out.data = +out.data
        return out
    __pos__.__doc__ = numpy.ndarray.__pos__.__doc__

    def __abs__(self, val):
        out = self.copy()
        out.data = out.data.abs()
        return out
    __abs__.__doc__ = numpy.ndarray.__abs__.__doc__

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, int) and self.unit:
            return Quantity(self.data[item], unit=self.unit)
        elif isinstance(item, int):
            return self.data[item]
        else:
            return super(NDData, self).__getitem__(item)

    read = classmethod(io_registry.read)
    write = io_registry.write
