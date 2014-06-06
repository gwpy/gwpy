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

"""This module provides a few utilities for handling optional
dependencies within GWpy code.
"""

import inspect
from functools import wraps

from .. import version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def import_method_dependency(module, stacklevel=1):
    """Import the given module, with a more useful `ImportError` message.

    Parameters
    ----------
    module : `str`
        name of the module to import

    Returns
    -------
    object : `object`
        the imported `object`

    Raises
    ------
    ImportError
        if the given object cannot be imported
    """
    try:
        return __import__(module, fromlist=[''])
    except ImportError as e:
        # get method name
        caller = inspect.stack()[stacklevel][3]
        # raise a better exception
        if not e.args: 
           e.args=('',)
        e.args = ("Cannot import %s required by the %s() method: %r"
                  % (module, caller, str(e)),) + e.args[1:]
        raise


def with_import(module):
    """Decorate a given method to import an optional dependency.

    Parameters
    ----------
    module : `str`
        name of the module object to import

    Returns
    -------
    decorator : `function`
        a decorator to apply to a method with the optional import
    """
    modname = module.split('.')[-1]
    def decorate_method(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func.func_globals[modname] = import_method_dependency(module,
                                                                  stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorate_method
