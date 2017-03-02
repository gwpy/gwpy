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

from .registry import (read, write)
from .mp import with_nproc

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def reader(name=None, doc=None, mp_flattener=False):
    """Construct a new unified input/output reader.

    This method is required to create a new copy of the
    :func:`astropy.io.registry.read` with a dynamic docstring.

    Returns
    -------
    read : `function`
        a copy of the :func:`astropy.io.registry.read` function

    doc : `str`
        custom docstring for this reader

    mp_flattener : `function`
        the function to flatten multiple instances of the parent object,
        enabling multiprocessed reading via the `nproc` argument
    """
    func = FunctionType(read.func_code, read.func_globals,
                        name or read.func_name, read.func_defaults,
                        read.func_closure)
    if doc is not None:
        func.__doc__ = doc.strip('\n ')
    if mp_flattener:
        return with_nproc(func, mp_flattener)
    else:
        return func


def writer(doc=None):
    """Construct a new unified input/output writeer.

    This method is required to create a new copy of the
    :func:`astropy.io.registry.write` with a dynamic docstring.

    Returns
    -------
    write : `function`
        A copy of the :func:`astropy.io.registry.write` function
    """
    func = FunctionType(write.func_code, write.func_globals,
                        write.func_name, write.func_defaults,
                        write.func_closure)
    if doc is not None:
        func.__doc__ = doc.strip('\n ')
    return func
