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

from gzip import GzipFile

from astropy.utils.compat.gzip import GzipFile as AstroGzipFile

from glue.lal import CacheEntry

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def identify_factory(*extensions):
    def identify(*args, **kwargs):
        """Identify the given extensions in a file object/path
        """
        fp = args[3]
        if isinstance(fp, (file, GzipFile, AstroGzipFile)):
            fp = fp.name
        elif isinstance(fp, CacheEntry):
            fp = fp.path
        # identify string
        if isinstance(fp, (unicode, str)) and fp.endswith(extensions):
            return True
        else:
            return False
    return identify
