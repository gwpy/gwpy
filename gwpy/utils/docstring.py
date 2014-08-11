# -*- coding: utf-8 -*-
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

"""Utility to update parent class references in docstring headers
"""

from inspect import getmembers, ismethod
from types import FunctionType


def update_docstrings(cls):
    """Update member docstrings to reference this class instead of parents

    Parameters
    ----------
    cls : `type`
       `class object` to update

    Notes
    -----
    This function is a class decorator, and should be used as such.
    """
    for name, func in getmembers(cls):
        try:
            d1 = func.__doc__.split('\n')[0]
        except AttributeError:
            continue
        for parent in cls.__mro__[1:]:
            if '`%s`' % parent.__name__ in d1:
                doc_ = func.__doc__.replace(parent.__name__, cls.__name__, 1)
                if ismethod(func):
                    # docstring replacement requires a new copy of the method
                    setattr(cls, name,
                            FunctionType(func.func_code, func.func_globals,
                                         name, func.func_defaults,
                                         func.func_closure))
                    func = getattr(cls, name)
                    func.__func__.__doc__ = doc_
                elif isinstance(func, property):
                    setattr(cls, name, property(doc=doc_, fget=func.fget,
                                                fset=func.fset, fdel=func.fdel))
    return cls
