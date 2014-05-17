# coding=utf-8
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

"""This target provides a few utilities for handling optional
dependencies within GWpy code.
"""

import inspect

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def import_method_dependency(target):
    """Import the given target, with a more useful `ImportError` message.

    Parameters
    ----------
    target : `str`
        name of the target object (module, method, class, or variable)
        to import

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
        return __import__(target, fromlist=[''])
    except ImportError:
        caller = inspect.stack()[1][3]
        raise ImportError("Cannot import %r required by the %s() method."
                          % (target, caller))
