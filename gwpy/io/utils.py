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

"""Utilities for unified input/output
"""

from six import string_types

from astropy.utils.compat.gzip import GzipFile

from glue.lal import CacheEntry

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def identify_factory(*extensions):
    def identify(origin, path, fileobj, *args, **kwargs):
        """Identify the given extensions in a file object/path
        """
        if isinstance(path, string_types) and path.endswith(extensions):
            return True
        else:
            return False
    return identify


def gopen(name, *args, **kwargs):
    """Open a file handling optional gzipping

    Parameters
    ----------
    name : `str`
        path (name) of file to open
    *args, **kwargs
        other arguments to pass to either `open` for regular files, or
        `gzip.open` for files with a `name` ending in `.gz`
    """
    if name.endswith('.gz'):
        return GzipFile(name, *args, **kwargs)
    else:
        return open(name, *args, **kwargs)
