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

"""Utilities for data input/output in standard GW formats.
"""

from types import FunctionType

from astropy.io.registry import (read, write)

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def reader(name=None, doc=None):
    """Construct a new unified input/output reader.

    This method is required to create a new copy of the
    :func:`astropy.io.registry.read` with a dynamic docstring.

    Returns
    -------
    read : `function`
        A copy of the :func:`astropy.io.registry.read` function
    """
    func = FunctionType(read.func_code, read.func_globals,
                        name or read.func_name, read.func_defaults,
                        read.func_closure)
    if doc is not None:
        func.__doc__ = doc.strip('\n ')
    return func


def writer():
    """Construct a new unified input/output writeer.

    This method is required to create a new copy of the
    :func:`astropy.io.registry.write` with a dynamic docstring.

    Returns
    -------
    write : `function`
        A copy of the :func:`astropy.io.registry.write` function
    """
    return FunctionType(write.func_code, write.func_globals,
                        write.func_name, write.func_defaults,
                        write.func_closure)
