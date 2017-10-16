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

import gzip

from six import string_types

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

GZIP_SIGNATURE = '\x1f\x8b\x08'


def identify_factory(*extensions):
    def identify(origin, filepath, fileobj, *args, **kwargs):
        """Identify the given extensions in a file object/path
        """
        if (isinstance(filepath, string_types) and
                filepath.endswith(extensions)):
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
    if name.endswith('.gz'):  # filename declares gzip
        return gzip.open(name, *args, **kwargs)
    else:  # open regular file
        fobj = open(name, *args, **kwargs)
        sig = fobj.read(3)
        fobj.seek(0)
        is_gzip = (sig == bytes(GZIP_SIGNATURE) if isinstance(sig, bytes) else
                   sig == GZIP_SIGNATURE)
        if is_gzip:   # file signature declares gzip
            return gzip.GzipFile(fileobj=fobj)
        return fobj
