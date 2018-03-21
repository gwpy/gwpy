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

"""Decorators for GWpy
"""

import warnings
from functools import wraps

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class deprecated_property(property):  # pylint: disable=invalid-name
    """sub-class of `property` that invokes DeprecationWarning on every call
    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        # get name  of property
        pname = fget.__name__

        # build a wrapper that will spawn a DeprecationWarning for all calls
        def _warn(func):
            @wraps(func)
            def _wrapped(self, *args, **kwargs):
                parent = type(self).__name__  # parent class name
                warnings.warn('the {0}.{1} property is deprecated, and will '
                              'be removed in a future release, please stop '
                              'using it.'.format(parent, pname),
                              DeprecationWarning)
                return func(self, *args, **kwargs)

            return _wrapped

        # wrap the property methods
        if fdel:
            fdel = _warn(fdel)
        if fset:
            fset = _warn(fset)
        if not fset and not fdel:  # only wrap once
            fget = _warn(fget)

        super(deprecated_property, self).__init__(fget, fset, fdel, doc)
