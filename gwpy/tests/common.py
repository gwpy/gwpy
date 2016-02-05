# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Common methods for GWpy unit tests
"""

from gwpy.io.cache import (Cache, CacheEntry)
from gwpy.io.registry import identify_format
from gwpy import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def test_io_identify(cls, extensions, modes=['read', 'write']):
    for mode in modes:
        for ext in extensions:
            p = 'X-TEST_CACHE_ENTRY-0-1.%s' % ext
            c = Cache([CacheEntry.from_T050017(p)])
            for path in [p, [p], c[0], c]:
                formats = identify_format(
                    mode, cls, path, None, [], {})
                if len(formats) == 0:
                    raise RuntimeError("No %s.%s method identified for "
                                       "file-format %r in form %r"
                                       % (cls.__name__, mode, ext,
                                          type(path).__name__))
                elif len(formats) != 1:
                    raise RuntimeError("Multiple %s.%s methods identified for "
                                       "file-format %r in form %r"
                                       % (cls.__name__, mode, ext,
                                          type(path).__name__))
