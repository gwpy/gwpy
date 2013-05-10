# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Convenience module to return the correct function for a given data
type
"""

from ..lal import atomic
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def function_factory(stub, type_,):
    """Find the `LAL` function that performs the given operation stub
    for the given type

    >>> function_factory("create", "REAL8")
    lal.CreateREAL8FrequencySeries

    @returns a function
    """
    if isinstance(type_, int):
        type_ = atomic.LAL_TYPE_MAP[type_]
    name = "%s%sFrequencySeries" % (stub.title(), type_.upper())
    return getattr(lal, name)
