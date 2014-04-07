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

"""Input/Output utilities for LAL Cache files.
"""

from glue.lal import Cache

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def open_cache(lcf):
    """Read a LAL-format cache file into memory as a
    :class:`glue.lal.Cache`.
    """
    if isinstance(lcf, file):
        return Cache.fromfile(lcf)
    else:
        with open(lcf, 'r') as f:
            return Cache.fromfile(f)


def identify_cache_file(*args, **kwargs):
    """Determine an input object as either a LAL-format cache file.
    """
    cachefile = args[1]
    if isinstance(cachefile, file):
        cachefile = cachefile.name
    # identify string
    if (isinstance(cachefile, (unicode, str)) and
            cachefile.endswith(('.lcf', '.cache'))):
        return True
    # identify cache object
    else:
        return False


def identify_cache(*args, **kwargs):
    """Determine an input object as a :class:`glue.lal.Cache` or a
    :lalsuite:`LALCache`.
    """
    cacheobj = args[3]
    if isinstance(cacheobj, Cache):
        return True
    else:
        return False
